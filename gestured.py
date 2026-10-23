#!/usr/bin/env python3
"""gestured - hand-gesture control for Sway.

Webcam -> MediaPipe Hands -> gesture detectors -> Sway IPC (i3ipc).
All gesture -> command mappings live in config.toml.

Run `gestured.py --help` for options. Start/stop normally goes through toggle.sh.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import sys  # noqa: E402

if "--preview" in sys.argv:
    # OpenCV's bundled Qt has no Wayland plugin; use XWayland and skip the warning.
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import argparse
from collections import deque
import logging
import math
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

import cv2
import numpy as np
import i3ipc
import mediapipe as mp

log = logging.getLogger("gestured")

# Config

DEFAULTS = {
    "camera": {
        "index": 0,
        "width": 640,
        "height": 480,
        "fourcc": "MJPG",
        "device_fps": 30,
        "flip": True,
        "controls": {
            "exposure_dynamic_framerate": 0,
        },
    },
    "loop": {
        "fps_active": 30,
        "fps_idle": 6,
        "idle_after_s": 2.0,
        "lost_grace_s": 0.25,
        "hold_grace_s": 0.15,
        "two_grace_s": 0.3,
        "label_votes": 3,
        "cooldown_s": 0.3,
        "stats_every_s": 10.0,
        "notifications": True,
        "notify_hand": True,
        "hand_notify_min_absent_s": 1.0,
        "hand_notify_cooldown_s": 3.0,
        "notify_pose": True,
        "pose_notify_after_s": 0.12,
        "pose_notify_cooldown_s": 0.8,
        "hud": True,
        "hud_min_interval_s": 0.12,
    },
    "detector": {
        "model_complexity": 0,
        "min_detection_confidence": 0.45,
        "min_tracking_confidence": 0.4,
        "swap_handedness": False,
    },
    "pose": {
        "pinch_on": 0.35,
        "pinch_off": 0.55,
        "finger_extend": 0.95,
        "thumb_extend": 0.75,
        "fist_curl_max": 0.27,
        "palm_min_spread": 0.92,
        "fists_require_open_first": True,
        "palm_min_fingers": 3,
        "thumbs_up_min": 0.45,
    },
    "swipe": {
        "arm_s": 0.05,
        "still_max": 0.06,
        "rearm_still_max": 0.04,
        "palm_grace_s": 0.2,
        "min_distance": 0.09,
        "wave_distance": 0.08,
        "window_s": 0.5,
        "axis_ratio": 1.5,
        "rearm_s": 0.25,
    },
    "three_finger_swipe": {
        "arm_s": 0.05,
        "still_max": 0.06,
        "rearm_still_max": 0.04,
        "min_distance": 0.08,
        "wave_distance": 0.06,
        "window_s": 0.5,
        "axis_ratio": 1.5,
        "rearm_s": 0.25,
    },
    "pinch_drag": {
        "settle_s": 0.06,
        "min_distance": 0.10,
        "wave_distance": 0.08,
        "axis_ratio": 1.5,
    },
    "resize": {
        "step_norm": 0.04,
        "min_interval_s": 0.12,
    },
    "workspaces": {
        "count": 10,
        "wrap": True,
    },
}

SCRUB_DEFAULTS = {
    "scrub": True,
    "scrub_delay_s": 0.4,
    "scrub_interval_s": 0.3,
    "scrub_pause_s": 0.2,
    "scrub_eps": 0.004,
    "scrub_reverse": 0.06,
}
for _t in ("swipe", "pinch_drag"):
    DEFAULTS[_t] = {**SCRUB_DEFAULTS, **DEFAULTS[_t]}

HOLD_GESTURES = {"three_finger_hold", "fists_hold", "thumbs_up_hold"}
KNOWN_GESTURES = HOLD_GESTURES | {
    "palm_swipe_left",
    "palm_swipe_right",
    "three_finger_swipe_left",
    "three_finger_swipe_right",
    "pinch_drag_left",
    "pinch_drag_right",
    "resize_spread_h",
    "resize_squeeze_h",
    "resize_spread_v",
    "resize_squeeze_v",
}
VALID_HANDS = {"left", "right", "any", "both"}


class ConfigError(Exception):
    pass


class Binding:
    def __init__(self, name, gesture, hand, command, hold_s, notify):
        self.name = name
        self.gesture = gesture
        self.hand = hand
        self.command = command
        self.hold_s = hold_s
        self.notify = notify

    def __repr__(self):
        return f"<{self.name}: {self.gesture}/{self.hand} -> {self.command!r}>"


def deep_merge(base: dict, over: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path) -> dict:
    log.info("loading config: %s", path)
    with open(path, "rb") as f:
        user = tomllib.load(f)
    raw_bindings = user.pop("binding", [])
    cfg = deep_merge(DEFAULTS, user)

    bindings = []
    for i, b in enumerate(raw_bindings):
        gesture = b.get("gesture")
        hand = b.get("hand", "any")
        command = b.get("command")
        name = b.get("name", f"binding{i}")
        if gesture not in KNOWN_GESTURES:
            raise ConfigError(
                f"[{name}] unknown gesture {gesture!r}; known: {sorted(KNOWN_GESTURES)}"
            )
        if hand not in VALID_HANDS:
            raise ConfigError(
                f"[{name}] hand must be one of {sorted(VALID_HANDS)}, got {hand!r}"
            )
        if not command:
            raise ConfigError(f"[{name}] missing 'command'")
        hold_s = float(b.get("hold_s", 0.7 if gesture in HOLD_GESTURES else 0.0))
        bindings.append(
            Binding(name, gesture, hand, command, hold_s, b.get("notify", ""))
        )
    if not bindings:
        raise ConfigError("no [[binding]] entries found in config")
    cfg["bindings"] = bindings
    log.info("config ok: %d bindings", len(bindings))
    for b in bindings:
        log.info(
            "  binding %-30s %-24s hand=%-5s hold=%.2fs -> %s",
            b.name,
            b.gesture,
            b.hand,
            b.hold_s,
            b.command,
        )
        if "wtype" in b.command and not shutil.which("wtype"):
            log.warning(
                "  binding %r needs wtype, which is not installed "
                "(sudo dnf install wtype)",
                b.name,
            )
    return cfg


def find_config(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    else:
        xdg = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        candidates.append(xdg / "gestured" / "config.toml")
        candidates.append(Path(__file__).resolve().parent / "config.toml")
    for c in candidates:
        if c.is_file():
            return c
    raise ConfigError(
        "config not found; looked in: " + ", ".join(str(c) for c in candidates)
    )


class Sway:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.conn = None

    def _connect(self):
        if self.conn is None:
            self.conn = i3ipc.Connection()
            log.info("connected to Sway IPC")
        return self.conn

    def current_ws(self) -> int:
        try:
            for w in self._connect().get_workspaces():
                if w.focused:
                    return w.num
        except Exception as e:  # noqa: BLE001
            self.conn = None
            log.warning("could not read current workspace (%s); assuming 1", e)
        return 1

    def run(self, command: str) -> None:
        if self.dry_run:
            log.info("[dry-run] sway command: %s", command)
            return
        for attempt in (1, 2):
            try:
                replies = self._connect().command(command)
                for r in replies:
                    if r.success:
                        log.info("sway ok: %s", command)
                    else:
                        log.error("sway rejected %r: %s", command, r.error)
                return
            except Exception as e:  # noqa: BLE001
                log.warning("sway IPC failure on attempt %d: %s", attempt, e)
                self.conn = None
        log.error("giving up on sway command %r", command)


class Notifier:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def send(
        self,
        body: str,
        title: str = "Gestures",
        timeout_ms: int = 1500,
        tag: str = "gestured",
        wait: bool = True,
    ) -> None:
        if not self.enabled:
            return
        cmd = [
            "notify-send",
            "-a",
            "gestured",
            "-t",
            str(timeout_ms),
            "-h",
            f"string:x-canonical-private-synchronous:{tag}",
            "-h",
            f"string:x-dunst-stack-tag:{tag}",
            title,
            body,
        ]
        try:
            if wait:
                subprocess.run(
                    cmd,
                    timeout=2,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            log.debug("notify-send unavailable")


class Hud:
    def __init__(self, enabled: bool, min_interval: float, dry_run: bool = False):
        self.enabled = enabled
        self.min_interval = min_interval
        self.title = "Gesture control" + (" (dry-run)" if dry_run else "")
        self.nid = None
        self.shown = None
        self.last_sent = -1e9

    def _cmd(self, timeout_ms: int, tail: list) -> list:
        cmd = [
            "notify-send",
            "-a",
            "gestured",
            "-t",
            str(timeout_ms),
            "-h",
            "string:x-canonical-private-synchronous:gestured-hud",
            "-h",
            "string:x-dunst-stack-tag:gestured-hud",
        ]
        if self.nid:
            cmd += ["-r", str(self.nid)]
        return cmd + tail

    def _send(self, body: str, timeout_ms: int = 0, wait: bool = False) -> None:
        quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        try:
            if self.nid is None:
                r = subprocess.run(
                    self._cmd(timeout_ms, ["-p", self.title, body]),
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                out = r.stdout.strip()
                if r.returncode == 0 and out.isdigit():
                    self.nid = int(out)
                    log.info("status notification shown (id %d)", self.nid)
                else:
                    self.nid = 0
                    if r.returncode != 0:
                        subprocess.run(
                            self._cmd(timeout_ms, [self.title, body]),
                            timeout=2,
                            check=False,
                            **quiet,
                        )
            elif wait:
                subprocess.run(
                    self._cmd(timeout_ms, [self.title, body]),
                    timeout=2,
                    check=False,
                    **quiet,
                )
            else:
                subprocess.Popen(
                    self._cmd(timeout_ms, [self.title, body]),
                    start_new_session=True,
                    **quiet,
                )
        except FileNotFoundError:
            log.warning(
                "notify-send not found; status notification disabled "
                "(sudo dnf install libnotify)"
            )
            self.enabled = False
        except (subprocess.SubprocessError, OSError) as e:
            log.debug("status notification failed: %s", e)

    def show(self, body: str, now: float) -> None:
        if (
            not self.enabled
            or body == self.shown
            or now - self.last_sent < self.min_interval
        ):
            return
        self.shown = body
        self.last_sent = now
        self._send(body)

    def close(self, body: str) -> None:
        if self.enabled:
            self._send(body, timeout_ms=1500, wait=True)


def ws_targets(cur: int, count: int, wrap: bool) -> tuple[int, int]:
    if not 1 <= cur <= count:
        cur = 1
    if wrap:
        return cur % count + 1, (cur - 2) % count + 1
    return min(cur + 1, count), max(cur - 1, 1)


def enhance_for_detection(rgb):
    """Boost local contrast + brightness so MediaPipe can find hands in dim rooms."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    rgb = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    mean_l = float(l.mean())
    if mean_l < 80.0:
        gamma = 0.65 if mean_l > 40.0 else 0.55
        lut = ((np.arange(256) / 255.0) ** gamma * 255.0).astype("uint8")
        rgb = cv2.LUT(rgb, lut)
    return rgb


FINGER_JOINTS = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Hand:
    def __init__(self, label: str, pts: list, pc: dict):
        self.label = label
        self.pts = pts
        self.size = max(dist(pts[0], pts[9]), 1e-6)
        self.pinch_ratio = dist(pts[4], pts[8]) / self.size
        self.mid = ((pts[4][0] + pts[8][0]) / 2.0, (pts[4][1] + pts[8][1]) / 2.0)
        self.ext = {}
        for name, (tip, pip) in FINGER_JOINTS.items():
            ratio = dist(pts[0], pts[tip]) / max(dist(pts[0], pts[pip]), 1e-6)
            self.ext[name] = ratio > pc["finger_extend"]
        self.thumb = dist(pts[4], pts[5]) / self.size > pc["thumb_extend"]
        self.pinching = False
        self.curled4 = not any(self.ext.values())
        self.curl = dist(pts[5], pts[8]) / self.size
        self.fist = self.curled4 and self.curl < pc["fist_curl_max"]
        self.t_mid = dist(pts[4], pts[10]) / self.size
        self.three = (
            self.thumb
            and self.ext["index"]
            and self.ext["middle"]
            and not self.ext["ring"]
            and not self.ext["pinky"]
        )
        self.palm = (
            sum(pts[i][0] for i in (0, 5, 9, 13, 17)) / 5.0,
            sum(pts[i][1] for i in (0, 5, 9, 13, 17)) / 5.0,
        )
        self.tips = (
            sum(pts[i][0] for i in (8, 12, 16, 20)) / 4.0,
            sum(pts[i][1] for i in (8, 12, 16, 20)) / 4.0,
        )
        self.palm_open = sum(self.ext.values()) >= pc["palm_min_fingers"]
        self.thumbs_up = (
            self.curled4
            and self.thumb
            and (pts[5][1] - pts[4][1]) / self.size > pc["thumbs_up_min"]
        )
        if self.thumbs_up:
            self.fist = False

    def pose_name(self) -> str:
        if self.thumbs_up:
            return "thumbs"
        if self.pinching:
            return "pinch"
        if self.fist:
            return "fist"
        if self.three:
            return "three"
        if self.palm_open:
            return "palm"
        return "-"

    def describe(self) -> str:
        fingers = "".join(n[0] for n, v in self.ext.items() if v)
        return (
            f"{self.label[0].upper()} pinch={self.pinch_ratio:.2f} curl={self.curl:.2f} "
            f"tm={self.t_mid:.2f} thumb={'Y' if self.thumb else 'n'} "
            f"ext={fingers or '-'} pose={self.pose_name()}"
        )


class Hold:
    def __init__(self):
        self.start = None
        self.fired = False
        self.notified = False
        self.last_active = 0.0

    def update(
        self,
        active: bool,
        now: float,
        need: float,
        grace: float = 0.0,
        force: bool = False,
    ):
        if active:
            self.last_active = now
            if self.start is None:
                self.start = now
                self.fired = False
                return "started"
            if not self.fired and now - self.start >= need:
                self.fired = True
                return "fire"
            return None
        if self.start is None:
            return None
        if not force and now - self.last_active <= grace:
            return None
        ev = "cancelled" if not self.fired else None
        self.start = None
        self.fired = False
        return ev


class Scrub:
    def __init__(self):
        self.active = False
        self.dir = "right"
        self.sign = 1
        self.best = 0.0
        self.t_start = 0.0
        self.t_moved = 0.0
        self.t_step = 0.0

    def start(self, direction: str, x: float, now: float):
        self.active = True
        self.dir = direction
        self.sign = 1 if direction == "right" else -1
        self.best = x
        self.t_start = now
        self.t_moved = now
        self.t_step = now

    def update(self, x: float, now: float, sc: dict):
        if (x - self.best) * self.sign > sc["scrub_eps"]:
            self.best = x
            self.t_moved = now
        elif (self.best - x) * self.sign > sc["scrub_reverse"]:
            self.active = False
            return "reverse"
        if now - self.t_moved > sc["scrub_pause_s"]:
            self.active = False
            return "pause"
        if (
            now - self.t_start >= sc["scrub_delay_s"]
            and now - self.t_step >= sc["scrub_interval_s"]
        ):
            self.t_step = now
            return "step"
        return None


class PalmSwipe:
    def __init__(self):
        self.state = "idle"
        self.trace = deque()
        self.last_open = -1e9
        self.scrub = Scrub()
        self.via = "hand move"

    def reset(self):
        self.state = "idle"
        self.trace.clear()
        self.scrub.active = False

    def _still(self, now: float, need: float, box: float, tips: bool = False) -> bool:
        if not self.trace or now - self.trace[0][0] < need:
            return False
        recent = [s for s in self.trace if now - s[0] <= need]
        xs = [s[1] for s in recent]
        ys = [s[2] for s in recent]
        if (max(xs) - min(xs)) > box or (max(ys) - min(ys)) > box:
            return False
        if tips:
            txs = [s[3] for s in recent]
            tys = [s[4] for s in recent]
            return (max(txs) - min(txs)) <= 2 * box and (max(tys) - min(tys)) <= 2 * box
        return True

    def update(self, palm_open: bool, pos, tip, now: float, sc: dict):
        if palm_open:
            self.last_open = now
        elif now - self.last_open > sc["palm_grace_s"]:
            if self.state != "idle":
                log.debug("palm closed (was %s)", self.state)
            self.reset()
            return None

        sample = (now, pos[0], pos[1], tip[0], tip[1])
        self.trace.append(sample)
        horizon = max(sc["arm_s"], sc["rearm_s"], sc["window_s"]) + 0.1
        while self.trace and now - self.trace[0][0] > horizon:
            self.trace.popleft()

        if self.state == "scrub":
            x = pos[0] if self.via == "hand move" else tip[0]
            ev = self.scrub.update(x, now, sc)
            if ev == "step":
                log.info("scroll step %s", self.scrub.dir)
                return self.scrub.dir, True
            if ev in ("pause", "reverse"):
                self.state = "armed" if ev == "pause" else "fired"
                self.trace.clear()
                self.trace.append(sample)
                log.info(
                    "scroll %s",
                    "paused - ready for the next swipe"
                    if ev == "pause"
                    else "turned back - rest to re-arm",
                )
            return None

        if self.state in ("idle", "fired"):
            if self.state == "idle":
                still = self._still(now, sc["arm_s"], sc["still_max"])
            else:
                still = self._still(
                    now, sc["rearm_s"], sc["rearm_still_max"], tips=True
                )
            if still:
                self.state = "armed"
                self.trace.clear()
                self.trace.append(sample)
                log.info("palm armed - move or wave left/right to trigger")
            return None

        best = 0.0
        for t, px, py, tx, ty in self.trace:
            if now - t > sc["window_s"]:
                continue
            for via, dx, dy, need in (
                ("hand move", pos[0] - px, pos[1] - py, sc["min_distance"]),
                ("wave", tip[0] - tx, tip[1] - ty, sc["wave_distance"]),
            ):
                if abs(dx) > abs(best):
                    best = dx
                if abs(dx) >= need and abs(dx) >= sc["axis_ratio"] * abs(dy):
                    direction = "right" if dx > 0 else "left"
                    log.info("palm swipe %s via %s (dx=%+.3f)", direction, via, dx)
                    self.trace.clear()
                    self.trace.append(sample)
                    if sc["scrub"]:
                        self.state = "scrub"
                        self.via = via
                        self.scrub.start(
                            direction, pos[0] if via == "hand move" else tip[0], now
                        )
                    else:
                        self.state = "fired"
                    return direction, False
        if abs(best) > 0.03:
            log.debug(
                "sway travel dx=%+.3f (need hand >=%.2f or wave >=%.2f within %.1fs)",
                best,
                sc["min_distance"],
                sc["wave_distance"],
                sc["window_s"],
            )
        return None


class ThreeFingerSwipe:
    def __init__(self):
        self.state = "idle"
        self.trace = deque()
        self.last_seen = -1e9

    def reset(self):
        self.state = "idle"
        self.trace.clear()

    def _still(self, now: float, need: float, box: float) -> bool:
        if not self.trace or now - self.trace[0][0] < need:
            return False
        recent = [s for s in self.trace if now - s[0] <= need]
        xs = [s[1] for s in recent]
        ys = [s[2] for s in recent]
        return (max(xs) - min(xs)) <= box and (max(ys) - min(ys)) <= box

    def update(self, active: bool, pos, now: float, sc: dict):
        if active:
            self.last_seen = now
        elif now - self.last_seen > 0.2:
            if self.state != "idle":
                log.debug("three-finger pose lost (was %s)", self.state)
            self.reset()
            return None

        sample = (now, pos[0], pos[1])
        self.trace.append(sample)
        horizon = max(sc["arm_s"], sc["rearm_s"], sc["window_s"]) + 0.1
        while self.trace and now - self.trace[0][0] > horizon:
            self.trace.popleft()

        if self.state in ("idle", "fired"):
            if self.state == "idle":
                still = self._still(now, sc["arm_s"], sc["still_max"])
            else:
                still = self._still(now, sc["rearm_s"], sc["rearm_still_max"])
            if still:
                self.state = "armed"
                self.trace.clear()
                self.trace.append(sample)
                log.info("three-finger swipe armed - move left/right to trigger")
            return None

        best = 0.0
        for t, px, py in self.trace:
            if now - t > sc["window_s"]:
                continue
            dx = pos[0] - px
            dy = pos[1] - py
            if abs(dx) > abs(best):
                best = dx
            if abs(dx) >= sc["min_distance"] and abs(dx) >= sc["axis_ratio"] * abs(dy):
                direction = "right" if dx > 0 else "left"
                log.info("three-finger swipe %s (dx=%+.3f)", direction, dx)
                self.trace.clear()
                self.trace.append(sample)
                self.state = "fired"
                return direction
        if abs(best) > 0.03:
            log.debug(
                "three-finger travel dx=%+.3f (need >=%.2f within %.1fs)",
                best,
                sc["min_distance"],
                sc["window_s"],
            )
        return None


class PinchDrag:
    def __init__(self):
        self.scrub = Scrub()
        self.src = 0
        self.reset()

    def reset(self):
        self.since = None
        self.anchor = None
        self.fired = False
        self.scrub.active = False

    def update(self, pinching: bool, pos, now: float, dc: dict):
        if not pinching:
            self.reset()
            return None
        if self.since is None:
            self.since = now
            return None
        if self.anchor is None:
            if now - self.since >= dc["settle_s"]:
                self.anchor = pos
                log.info("pinch held - move or wave left/right to move the window")
            return None
        if self.scrub.active:
            ev = self.scrub.update(pos[self.src][0], now, dc)
            if ev == "step":
                log.info("scroll step %s", self.scrub.dir)
                return self.scrub.dir, True
            if ev == "pause":
                self.anchor = pos
                log.info("scroll paused - move again for another step")
            elif ev == "reverse":
                self.fired = True
                log.info("scroll turned back - release the pinch to re-arm")
            return None
        if self.fired:
            return None
        best = 0.0
        for src, via, need in (
            (0, "hand move", dc["min_distance"]),
            (1, "wave", dc["wave_distance"]),
        ):
            dx = pos[src][0] - self.anchor[src][0]
            dy = pos[src][1] - self.anchor[src][1]
            if abs(dx) > abs(best):
                best = dx
            if abs(dx) >= need and abs(dx) >= dc["axis_ratio"] * abs(dy):
                direction = "right" if dx > 0 else "left"
                log.info("pinch drag %s via %s (dx=%+.3f)", direction, via, dx)
                if dc["scrub"]:
                    self.src = src
                    self.scrub.start(direction, pos[src][0], now)
                else:
                    self.fired = True
                return direction, False
        if abs(best) > 0.03:
            log.debug(
                "pinch drag dx=%+.3f (need hand >=%.2f or wave >=%.2f)",
                best,
                dc["min_distance"],
                dc["wave_distance"],
            )
        return None


class ResizeTracker:
    def __init__(self):
        self.base = None
        self.last = 0.0

    def reset(self):
        if self.base is not None:
            log.info("resize ended")
        self.base = None

    def update(self, a_mid, b_mid, now: float, rc: dict):
        dx = abs(a_mid[0] - b_mid[0])
        dy = abs(a_mid[1] - b_mid[1])
        if self.base is None:
            self.base = [dx, dy]
            self.last = now
            log.info("resize started (both hands pinching)")
            return None
        ddx = dx - self.base[0]
        ddy = dy - self.base[1]
        axis = "h" if abs(ddx) >= abs(ddy) else "v"
        d = ddx if axis == "h" else ddy
        if abs(d) < rc["step_norm"] or now - self.last < rc["min_interval_s"]:
            return None
        sign = 1 if d > 0 else -1
        self.base[0 if axis == "h" else 1] += sign * rc["step_norm"]
        self.last = now
        return f"resize_{'spread' if sign > 0 else 'squeeze'}_{axis}"


class Engine:
    def __init__(
        self,
        cfg: dict,
        sway: Sway,
        notifier: Notifier,
        hand_notifier: Notifier | None = None,
        hud: Hud | None = None,
    ):
        self.cfg = cfg
        self.pc = cfg["pose"]
        self.sc = cfg["swipe"]
        self.tfsc = cfg["three_finger_swipe"]
        self.rc = cfg["resize"]
        self.wsc = cfg["workspaces"]
        self.cooldown = cfg["loop"]["cooldown_s"]
        self.grace = cfg["loop"]["lost_grace_s"]
        self.bindings = cfg["bindings"]
        self.sway = sway
        self.notifier = notifier
        lc = cfg["loop"]
        self.hand_notifier = hand_notifier if hand_notifier is not None else notifier
        self.notify_hand = bool(lc["notify_hand"])
        self.hand_min_absent = float(lc["hand_notify_min_absent_s"])
        self.hand_cooldown = float(lc["hand_notify_cooldown_s"])
        self.last_hand_notify = -1e9
        self.notify_pose = bool(lc["notify_pose"])
        self.pose_after = float(lc["pose_notify_after_s"])
        self.pose_cooldown = float(lc["pose_notify_cooldown_s"])
        self.cur_pose = None
        self.pose_since = 0.0
        self.pose_sent = False
        self.last_pose_notify = -1e9
        self.hud = hud if hud is not None else Hud(False, 0.12)
        self.hud_ws = None
        self.hud_ws_t = 0.0
        self.hud_pose = "No hand"
        self.hud_last = ""

        self.swipe = PalmSwipe()
        self.tfswipe = ThreeFingerSwipe()
        self.pdrag = PinchDrag()
        self.dc = cfg["pinch_drag"]
        self.label_votes = deque(maxlen=max(1, int(cfg["loop"]["label_votes"])))
        self.hold_grace = cfg["loop"]["hold_grace_s"]
        self.two_grace = cfg["loop"]["two_grace_s"]
        self.last_two = -1e9
        self.kill_armed = False
        self.need_open = bool(self.pc["fists_require_open_first"])
        self.holds = {g: Hold() for g in HOLD_GESTURES}
        self.resize = ResizeTracker()
        self.pinch_state: dict[str, bool] = {}
        self.mode = "none"
        self.last_seen = 0.0
        self.last_action = -1e9
        self.debug_lines: list[str] = []

    def process(self, hands: list, now: float) -> None:
        self._process(hands, now)
        self._hud_refresh(now)

    def _hud_refresh(self, now: float) -> None:
        if not self.hud.enabled:
            return
        if self.hud_ws is None or (
            not self.sway.dry_run and now - self.hud_ws_t >= 1.0
        ):
            self.hud_ws = self.sway.current_ws()
            self.hud_ws_t = now
        body = f"Workspace {self.hud_ws}\n{self.hud_pose}"
        if self.hud_last:
            body += f"\n{self.hud_last}"
        self.hud.show(body, now)

    def _process(self, hands: list, now: float) -> None:
        self.debug_lines = []
        hands = hands[:2]
        if not hands:
            if self.mode != "none" and now - self.last_seen > self.grace:
                log.info("hands lost -> resetting state")
                self.reset_all(now)
                self.mode = "none"
                self.hud_pose = "No hand"
            return
        prev_seen = self.last_seen
        self.last_seen = now
        mode = "one" if len(hands) == 1 else "two"
        if mode == "two":
            self.last_two = now
        elif self.mode == "two" and now - self.last_two <= self.two_grace:
            return
        if mode != self.mode:
            log.info("tracking mode: %s -> %s", self.mode, mode)
            was_none = self.mode == "none"
            self.reset_all(now)
            self.mode = mode
            if was_none:
                self._hand_appeared(mode, now, now - prev_seen)
        if mode == "one":
            self._one(hands[0], now)
        else:
            self._two(hands, now)

    def _hand_appeared(self, mode: str, now: float, absent: float) -> None:
        if self.hud.enabled or not self.notify_hand or absent < self.hand_min_absent:
            return
        if now - self.last_hand_notify < self.hand_cooldown:
            return
        self.last_hand_notify = now
        self.hand_notifier.send(
            "Two hands detected" if mode == "two" else "Hand detected",
            title="Gesture control",
            timeout_ms=1200,
            tag="gestured-hand",
            wait=False,
        )

    def _pose_toast(self, name: str | None, now: float) -> None:
        if name != self.cur_pose:
            self.cur_pose = name
            self.pose_since = now
            self.pose_sent = False
            return
        if (
            name is None
            or self.pose_sent
            or not self.notify_pose
            or self.hud.enabled
            or now - self.pose_since < self.pose_after
            or now - self.last_pose_notify < self.pose_cooldown
        ):
            return
        self.pose_sent = True
        self.last_pose_notify = now
        log.info("pose detected: %s", name)
        self.hand_notifier.send(
            name,
            title="Gesture control",
            timeout_ms=900,
            tag="gestured-pose",
            wait=False,
        )

    def reset_all(self, now: float) -> None:
        for g in self.holds:
            self._hold_update(g, "any", False, now, force=True)
        self.swipe.reset()
        self.tfswipe.reset()
        self.pdrag.reset()
        self.resize.reset()
        self.pinch_state.clear()
        self.label_votes.clear()
        self.kill_armed = False
        self.cur_pose = None

    def _pinch(self, key: str, h: Hand) -> bool:
        was = self.pinch_state.get(key, False)
        if h.fist:
            val = False
        else:
            val = h.pinch_ratio < (self.pc["pinch_off"] if was else self.pc["pinch_on"])
        self.pinch_state[key] = val
        return val

    def bindings_for(self, gesture: str, hand: str) -> list:
        return [
            b
            for b in self.bindings
            if b.gesture == gesture and (b.hand == "any" or b.hand == hand)
        ]

    def _debug(self, hand: Hand) -> None:
        line = hand.describe()
        self.debug_lines.append(line)
        log.debug(line)

    def _vote_label(self, label: str) -> str:
        self.label_votes.append(label)
        votes = list(self.label_votes)
        return max(set(votes), key=votes.count)

    def _one(self, h: Hand, now: float) -> None:
        label = self._vote_label(h.label)
        h.label = label
        h.pinching = self._pinch("one", h)
        self._debug(h)

        self._hold_update("fists_hold", "any", h.fist and not h.pinching, now)
        up = h.thumbs_up
        self._hold_update("thumbs_up_hold", label, up, now)
        pose3 = h.three and not h.pinching
        self._hold_update("three_finger_hold", label, pose3, now)

        if up:
            self._pose_toast("Thumbs up", now)
        elif h.pinching:
            self._pose_toast("Pinch", now)
        elif h.fist:
            self._pose_toast("Fist", now)
        else:
            self._pose_toast(None, now)

        palm = (
            h.palm_open
            and not h.pinching
            and h.pinch_ratio >= self.pc["palm_min_spread"]
        )
        res = self.swipe.update(palm, h.palm, h.tips, now, self.sc)
        if res:
            direction, step = res
            self.emit(f"palm_swipe_{direction}", label, now, cooldown=not step)

        res = self.tfswipe.update(pose3, h.palm, now, self.tfsc)
        if res:
            direction = res
            self.emit(f"three_finger_swipe_{direction}", label, now, cooldown=True)

        res = self.pdrag.update(h.pinching, (h.palm, h.mid), now, self.dc)
        if res:
            drag, step = res
            self.emit(f"pinch_drag_{drag}", label, now, cooldown=not step)

        if up:
            pose = "Thumbs up"
        elif h.pinching:
            pose = "Pinch"
            if self.pdrag.scrub.active:
                pose += " - dragging"
            elif self.pdrag.anchor is not None:
                pose += " - ready, move or wave"
        elif h.fist:
            pose = "Fist"
        elif h.three:
            pose = "Three fingers"
            if self.tfswipe.state == "armed":
                pose += " - ready, swipe left/right"
        elif palm:
            pose = "Open palm"
            if self.swipe.state == "scrub":
                pose += " - scrolling"
            elif self.swipe.state == "armed":
                pose += " - ready, move or wave"
        else:
            pose = "Hand"
        self.hud_pose = f"{label.capitalize()} hand: {pose}"

    def _two(self, hands: list, now: float) -> None:
        a, b = sorted(hands, key=lambda h: h.pts[0][0])
        a.pinching = self._pinch("slot0", a)
        b.pinching = self._pinch("slot1", b)
        self._debug(a)
        self._debug(b)

        resize_active = a.pinching and b.pinching

        self._hold_update("three_finger_hold", "any", False, now)
        self._hold_update("thumbs_up_hold", "any", False, now)
        self._hold_update("fists_hold", "any", False, now)

        if resize_active:
            self._pose_toast("Both hands pinching", now)
            self.hud_pose = "Both hands pinching - resize"
        else:
            self._pose_toast(None, now)
            self.hud_pose = "Two hands"

        if resize_active:
            g = self.resize.update(a.mid, b.mid, now, self.rc)
            if g:
                self.emit(g, "both", now, cooldown=False)
        else:
            self.resize.reset()

    def _hold_update(
        self, gesture: str, hand: str, active: bool, now: float, force: bool = False
    ) -> None:
        h = self.holds[gesture]
        if not active:
            ev = h.update(False, now, 0.0, self.hold_grace, force)
            if ev == "cancelled":
                log.info("hold cancelled: %s", gesture)
                if h.notified:
                    self.notifier.send("Cancelled")
                    h.notified = False
            return
        bs = self.bindings_for(gesture, hand)
        if not bs:
            return
        need = bs[0].hold_s
        ev = h.update(True, now, need)
        if ev == "started":
            log.info("hold started: %s (hand=%s, need %.2fs)", gesture, hand, need)
            if bs[0].notify:
                self.notifier.send(bs[0].notify, timeout_ms=int(need * 1000) + 600)
                h.notified = True
        elif ev == "fire":
            log.info("hold complete: %s", gesture)
            h.notified = False
            self.emit(gesture, hand, now)

    def emit(self, gesture: str, hand: str, now: float, cooldown: bool = True) -> None:
        if cooldown and now - self.last_action < self.cooldown:
            log.info("gesture %s ignored (cooldown)", gesture)
            return
        bs = self.bindings_for(gesture, hand)
        if not bs:
            log.info("gesture %s [%s] has no binding", gesture, hand)
            return
        log.info("GESTURE %s [%s]", gesture, hand)
        for b in bs:
            self._run(b)
        self.last_action = now

    def _run(self, b: Binding) -> None:
        cur = self.sway.current_ws()
        nxt, prv = ws_targets(cur, int(self.wsc["count"]), bool(self.wsc["wrap"]))
        try:
            cmd = b.command.format(cur=cur, next=nxt, prev=prv)
        except (KeyError, IndexError, ValueError) as e:
            log.error("[%s] bad command template %r: %s", b.name, b.command, e)
            return
        log.info(
            "[%s] workspace %d (next=%d prev=%d) -> %s", b.name, cur, nxt, prv, cmd
        )
        self.sway.run(cmd)
        target = (
            nxt if "{next}" in b.command else (prv if "{prev}" in b.command else None)
        )
        if target is not None and (
            cmd.startswith("workspace number") or "; workspace number" in cmd
        ):
            self.hud_ws = target
        self.hud_last = f"Last: {b.name}"


def pidfile_path() -> Path:
    return Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "gestured.pid"


def acquire_pidfile() -> bool:
    path = pidfile_path()
    if path.exists():
        try:
            old = int(path.read_text().strip())
            os.kill(old, 0)
        except (ValueError, OSError):
            log.info("removing stale pidfile")
        else:
            if old != os.getpid():
                log.error("already running (pid %d)", old)
                return False
    path.write_text(str(os.getpid()))
    return True


def release_pidfile() -> None:
    path = pidfile_path()
    try:
        if path.exists() and path.read_text().strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass


def open_camera(cc: dict):
    log.info("opening camera %s ...", cc["index"])
    cap = cv2.VideoCapture(cc["index"], cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    if cc.get("fourcc"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cc["fourcc"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cc["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cc["height"])
    cap.set(cv2.CAP_PROP_FPS, cc["device_fps"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    log.info(
        "camera opened: %dx%d @ %.0f fps (requested %dx%d)",
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        cap.get(cv2.CAP_PROP_FPS),
        cc["width"],
        cc["height"],
    )
    return cap


def apply_camera_controls(cc: dict) -> None:
    controls = cc.get("controls") or {}
    if not controls:
        return
    exe = shutil.which("v4l2-ctl")
    if not exe:
        log.warning(
            "v4l2-ctl not found; skipping camera controls (sudo dnf install v4l-utils)"
        )
        return
    idx = cc["index"]
    dev = str(idx) if isinstance(idx, str) else f"/dev/video{idx}"
    for name, value in controls.items():
        try:
            r = subprocess.run(
                [exe, "-d", dev, f"--set-ctrl={name}={value}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError) as e:
            log.warning("camera control %s=%s failed: %s", name, value, e)
            continue
        if r.returncode == 0:
            log.info("camera control %s=%s", name, value)
        else:
            log.warning(
                "camera control %s=%s not applied: %s",
                name,
                value,
                (r.stderr or r.stdout).strip() or f"exit {r.returncode}",
            )


def build_hands(res, aspect: float, cfg: dict) -> list:
    out = []
    if not res.multi_hand_landmarks:
        return out
    swap = cfg["detector"]["swap_handedness"]
    for lms, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
        label = hd.classification[0].label.lower()
        if swap:
            label = "left" if label == "right" else "right"
        pts = [(p.x, p.y * aspect) for p in lms.landmark]
        out.append(Hand(label, pts, cfg["pose"]))
    return out


def run(args) -> int:
    try:
        cfg = load_config(find_config(args.config))
    except (ConfigError, OSError, tomllib.TOMLDecodeError) as e:
        log.error("config problem: %s", e)
        return 2

    if not acquire_pidfile():
        return 1

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    notifier = Notifier(cfg["loop"]["notifications"] and not args.dry_run)
    sway = Sway(args.dry_run)
    hand_notifier = Notifier(
        cfg["loop"]["notifications"] and cfg["loop"]["notify_hand"]
    )
    hud = Hud(
        cfg["loop"]["notifications"] and cfg["loop"]["hud"],
        cfg["loop"]["hud_min_interval_s"],
        args.dry_run,
    )
    engine = Engine(cfg, sway, notifier, hand_notifier, hud)

    cap = open_camera(cfg["camera"])
    if cap is None:
        log.error(
            "could not open camera %s (check /dev/video*, 'video' group, other apps using it)",
            cfg["camera"]["index"],
        )
        Notifier(True).send("Could not open the camera", title="Gesture control")
        release_pidfile()
        return 1
    apply_camera_controls(cfg["camera"])

    dc = cfg["detector"]
    log.info(
        "loading MediaPipe Hands (model_complexity=%s) ...", dc["model_complexity"]
    )
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=dc["model_complexity"],
        min_detection_confidence=dc["min_detection_confidence"],
        min_tracking_confidence=dc["min_tracking_confidence"],
    )

    lc = cfg["loop"]
    flip = cfg["camera"]["flip"]
    log.info(
        "running: active %d fps, idle %d fps%s%s",
        lc["fps_active"],
        lc["fps_idle"],
        ", DRY-RUN" if args.dry_run else "",
        ", preview" if args.preview else "",
    )
    if not hud.enabled:
        Notifier(not args.dry_run and lc["notifications"]).send(
            "Gesture control started", title="Gesture control"
        )

    next_t = 0.0
    slack = 0.5 / max(cfg["camera"]["device_fps"], 1)
    last_hand_t = -1e9
    idle = False
    grab_failures = 0
    frames = 0
    hand_frames = 0
    stats_t = time.monotonic()
    exit_code = 0

    try:
        while not stop.is_set():
            if not cap.grab():
                if stop.is_set():
                    break
                grab_failures += 1
                if grab_failures == 1 or grab_failures % 10 == 0:
                    log.warning("camera grab failed (%d in a row)", grab_failures)
                if grab_failures >= 50:
                    log.error("camera seems gone, exiting")
                    exit_code = 1
                    break
                time.sleep(0.05)
                continue
            grab_failures = 0

            now = time.monotonic()
            if now + slack < next_t:
                continue

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            if flip:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = enhance_for_detection(rgb)
            rgb.flags.writeable = False
            res = detector.process(rgb)

            aspect = frame.shape[0] / frame.shape[1]
            hands = build_hands(res, aspect, cfg)
            if hands:
                last_hand_t = now
                hand_frames += 1
            engine.process(hands, now)
            frames += 1

            new_idle = (now - last_hand_t) > lc["idle_after_s"]
            if new_idle != idle:
                idle = new_idle
                log.info(
                    "%s mode (%d fps)",
                    "idle" if idle else "active",
                    lc["fps_idle"] if idle else lc["fps_active"],
                )
            next_t = now + 1.0 / (lc["fps_idle"] if idle else lc["fps_active"])

            if args.preview:
                if res.multi_hand_landmarks:
                    for lm in res.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                y = 20
                for line in [
                    f"mode={engine.mode} {'idle' if idle else 'active'}"
                ] + engine.debug_lines:
                    cv2.putText(
                        frame,
                        line,
                        (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                    y += 18
                cv2.imshow("gestured (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop.set()

            if now - stats_t >= lc["stats_every_s"]:
                span = now - stats_t
                log.info(
                    "alive: %.1f fps processed, hands in %d/%d frames, mode=%s, %s",
                    frames / span,
                    hand_frames,
                    frames,
                    engine.mode,
                    "idle" if idle else "active",
                )
                frames = 0
                hand_frames = 0
                stats_t = now
    except Exception:  # noqa: BLE001
        log.exception("fatal error in main loop")
        exit_code = 1
    finally:
        log.info("shutting down ...")
        try:
            detector.close()
        except Exception:  # noqa: BLE001
            pass
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()
        release_pidfile()
        if hud.enabled:
            hud.close("Gesture control stopped")
        else:
            Notifier(not args.dry_run and lc["notifications"]).send(
                "Gesture control stopped", title="Gesture control"
            )
        log.info("stopped")
    return exit_code


def main() -> int:
    ap = argparse.ArgumentParser(description="Hand-gesture control for Sway")
    ap.add_argument(
        "--config",
        help="path to config.toml "
        "(default: ~/.config/gestured/config.toml, then next to this script)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="detect gestures and log the Sway commands, but do not send them",
    )
    ap.add_argument(
        "--preview",
        action="store_true",
        help="show a debug window with landmarks and live measurements (needs XWayland)",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="debug logging: per-frame pinch ratios, finger states, poses",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    log.info("gestured starting (pid %d)", os.getpid())
    return run(args)


if __name__ == "__main__":
    sys.exit(main())