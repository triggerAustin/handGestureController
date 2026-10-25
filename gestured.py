#!/usr/bin/env python3
"""gestured - hand-gesture control for Sway."""

from __future__ import annotations
import os, sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
if "--preview" in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import argparse, json, logging, math, shutil, signal, subprocess, time
from collections import deque
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import cv2, numpy as np, i3ipc, mediapipe as mp

log = logging.getLogger("gestured")

DEFAULTS = {
    "camera": {
        "index": 0,
        "width": 640,
        "height": 480,
        "fourcc": "MJPG",
        "device_fps": 30,
        "flip": True,
        "controls": {},
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
        "palm_min_fingers": 3,
        "thumbs_up_min": 0.45,
    },
    "three_finger_swipe": {
        "arm_s": 0.05,
        "still_max": 0.06,
        "rearm_still_max": 0.04,
        "min_distance": 0.08,
        "window_s": 0.5,
        "axis_ratio": 1.5,
        "rearm_s": 0.25,
    },
    "pinch_drag": {
        "settle_s": 0.06,
        "min_distance": 0.10,
        "wave_distance": 0.08,
        "axis_ratio": 1.5,
        "scrub": True,
        "scrub_delay_s": 0.4,
        "scrub_interval_s": 0.3,
        "scrub_pause_s": 0.2,
        "scrub_eps": 0.004,
        "scrub_reverse": 0.06,
    },
    "resize": {"step_norm": 0.04, "min_interval_s": 0.12},
    "workspaces": {"count": 10, "wrap": True},
    "pointer": {
        "enabled": True,
        "sensitivity": 4.5,
        "deadzone": 0.004,
        "ema_alpha": 0.55,
        "accel": 0.5,
        "velocity_floor": 0.018,
    },
}

HOLD_GESTURES = {"fists_hold", "thumbs_up_hold", "pinky_hold"}
KNOWN_GESTURES = HOLD_GESTURES | {
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
        self.name, self.gesture, self.hand, self.command, self.hold_s, self.notify = (
            name,
            gesture,
            hand,
            command,
            hold_s,
            notify,
        )


def deep_merge(a, b):
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in a.items()}
    for k, v in b.items():
        out[k] = (
            deep_merge(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else v
        )
    return out


def load_config(path):
    log.info("loading config: %s", path)
    with open(path, "rb") as f:
        user = tomllib.load(f)
    raw = user.pop("binding", [])
    cfg = deep_merge(DEFAULTS, user)
    bs = []
    for i, b in enumerate(raw):
        g, h, c, n = (
            b.get("gesture"),
            b.get("hand", "any"),
            b.get("command"),
            b.get("name", f"b{i}"),
        )
        if g not in KNOWN_GESTURES:
            raise ConfigError(f"[{n}] unknown gesture {g!r}")
        if h not in VALID_HANDS:
            raise ConfigError(f"[{n}] bad hand {h!r}")
        if not c:
            raise ConfigError(f"[{n}] missing command")
        hold = float(b.get("hold_s", 0.7 if g in HOLD_GESTURES else 0.0))
        bs.append(Binding(n, g, h, c, hold, b.get("notify", "")))
    if not bs:
        raise ConfigError("no [[binding]] entries")
    cfg["bindings"] = bs
    return cfg


def find_config(explicit):
    if explicit:
        cands = [Path(explicit).expanduser()]
    else:
        xdg = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        cands = [
            xdg / "gestured" / "config.toml",
            Path(__file__).resolve().parent / "config.toml",
        ]
    for c in cands:
        if c.is_file():
            return c
    raise ConfigError("config not found")


def ws_targets(cur, count, wrap):
    if not 1 <= cur <= count:
        cur = 1
    return (
        (cur % count + 1, (cur - 2) % count + 1)
        if wrap
        else (min(cur + 1, count), max(cur - 1, 1))
    )


def enhance_for_detection(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(l)
    l = cv2.LUT(l, ((np.arange(256) / 255.0) ** 0.5 * 255.0).astype("uint8"))
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


FINGER_JOINTS = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Hand:
    def __init__(self, label, pts, pc):
        self.label, self.pts, self.pinching = label, pts, False
        self.size = max(dist(pts[0], pts[9]), 1e-6)
        self.pinch_ratio = dist(pts[4], pts[8]) / self.size
        self.mid = ((pts[4][0] + pts[8][0]) / 2, (pts[4][1] + pts[8][1]) / 2)
        self.ext = {
            n: dist(pts[0], pts[t]) / max(dist(pts[0], pts[p]), 1e-6)
            > pc["finger_extend"]
            for n, (t, p) in FINGER_JOINTS.items()
        }
        self.thumb = dist(pts[4], pts[5]) / self.size > pc["thumb_extend"]
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
        self.index_tip = pts[8]
        self.palm = (
            sum(pts[i][0] for i in (0, 5, 9, 13, 17)) / 5,
            sum(pts[i][1] for i in (0, 5, 9, 13, 17)) / 5,
        )
        self.palm_open = sum(self.ext.values()) >= pc["palm_min_fingers"]
        self.thumbs_up = (
            self.curled4
            and self.thumb
            and (pts[5][1] - pts[4][1]) / self.size > pc["thumbs_up_min"]
        )
        if self.thumbs_up:
            self.fist = False
        self.pointing = (
            self.ext["index"]
            and not self.ext["middle"]
            and not self.ext["ring"]
            and not self.ext["pinky"]
        )
        self.pinky_only = (
            self.ext["pinky"]
            and not self.ext["index"]
            and not self.ext["middle"]
            and not self.ext["ring"]
        )

    def pose_name(self):
        if self.thumbs_up:
            return "thumbs"
        if self.pinching:
            return "pinch"
        if self.pointing:
            return "point"
        if self.fist:
            return "fist"
        if self.three:
            return "three"
        if self.palm_open:
            return "palm"
        if self.pinky_only:
            return "pinky"
        return "-"

    def describe(self):
        f = "".join(n[0] for n, v in self.ext.items() if v)
        return f"{self.label[0].upper()} pinch={self.pinch_ratio:.2f} curl={self.curl:.2f} ext={f or '-'} pose={self.pose_name()}"


class Hold:
    def __init__(self):
        self.start, self.fired, self.notified, self.last_active = (
            None,
            False,
            False,
            0.0,
        )

    def update(self, active, now, need, grace=0.0, force=False):
        if active:
            self.last_active = now
            if self.start is None:
                self.start, self.fired = now, False
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
        self.start, self.fired = None, False
        return ev


class Scrub:
    def __init__(self):
        self.active, self.dir, self.sign, self.best = False, "right", 1, 0.0
        self.t_start = self.t_moved = self.t_step = 0.0

    def start(self, d, x, now):
        self.active, self.dir, self.sign = True, d, 1 if d == "right" else -1
        self.best, self.t_start, self.t_moved, self.t_step = x, now, now, now

    def update(self, x, now, sc):
        if (x - self.best) * self.sign > sc["scrub_eps"]:
            self.best, self.t_moved = x, now
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


class ThreeFingerSwipe:
    def __init__(self):
        self.state, self.trace, self.last_seen = "idle", deque(), -1e9

    def reset(self):
        self.state, self.trace = "idle", deque()

    def _still(self, now, need, box):
        if not self.trace or now - self.trace[0][0] < need:
            return False
        recent = [s for s in self.trace if now - s[0] <= need]
        return (max(s[1] for s in recent) - min(s[1] for s in recent)) <= box and (
            max(s[2] for s in recent) - min(s[2] for s in recent)
        ) <= box

    def update(self, active, pos, now, sc):
        if active:
            self.last_seen = now
        elif now - self.last_seen > 0.2:
            self.reset()
            return None
        sample = (now, pos[0], pos[1])
        self.trace.append(sample)
        while (
            self.trace
            and now - self.trace[0][0]
            > max(sc["arm_s"], sc["rearm_s"], sc["window_s"]) + 0.1
        ):
            self.trace.popleft()
        if self.state in ("idle", "fired"):
            still = (
                self._still(now, sc["arm_s"], sc["still_max"])
                if self.state == "idle"
                else self._still(now, sc["rearm_s"], sc["rearm_still_max"])
            )
            if still:
                self.state = "armed"
                self.trace.clear()
                self.trace.append(sample)
            return None
        best = 0.0
        for t, px, py in self.trace:
            if now - t > sc["window_s"]:
                continue
            dx, dy = pos[0] - px, pos[1] - py
            if abs(dx) > abs(best):
                best = dx
            if abs(dx) >= sc["min_distance"] and abs(dx) >= sc["axis_ratio"] * abs(dy):
                d = "right" if dx > 0 else "left"
                self.trace.clear()
                self.trace.append(sample)
                self.state = "fired"
                return d
        return None


class PinchDrag:
    def __init__(self):
        self.scrub = Scrub()
        self.src = 0
        self.reset()

    def reset(self):
        self.since, self.anchor, self.fired, self.scrub.active = (
            None,
            None,
            False,
            False,
        )

    def update(self, pinching, pos, now, dc):
        if not pinching:
            self.reset()
            return None
        if self.since is None:
            self.since = now
            return None
        if self.anchor is None:
            if now - self.since >= dc["settle_s"]:
                self.anchor = pos
            return None
        if self.scrub.active:
            ev = self.scrub.update(pos[self.src][0], now, dc)
            if ev == "step":
                return self.scrub.dir, True
            if ev == "pause":
                self.anchor = pos
            elif ev == "reverse":
                self.fired = True
            return None
        if self.fired:
            return None
        for src, need in ((0, dc["min_distance"]), (1, dc["wave_distance"])):
            dx = pos[src][0] - self.anchor[src][0]
            dy = pos[src][1] - self.anchor[src][1]
            if abs(dx) >= need and abs(dx) >= dc["axis_ratio"] * abs(dy):
                d = "right" if dx > 0 else "left"
                if dc["scrub"]:
                    self.src = src
                    self.scrub.start(d, pos[src][0], now)
                else:
                    self.fired = True
                return d, False
        return None


class ResizeTracker:
    def __init__(self):
        self.base, self.last = None, 0.0

    def reset(self):
        self.base = None

    def update(self, a, b, now, rc):
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        if self.base is None:
            self.base, self.last = [dx, dy], now
            return None
        ddx, ddy = dx - self.base[0], dy - self.base[1]
        axis = "h" if abs(ddx) >= abs(ddy) else "v"
        d = ddx if axis == "h" else ddy
        if abs(d) < rc["step_norm"] or now - self.last < rc["min_interval_s"]:
            return None
        sign = 1 if d > 0 else -1
        self.base[0 if axis == "h" else 1] += sign * rc["step_norm"]
        self.last = now
        return f"resize_{'spread' if sign > 0 else 'squeeze'}_{axis}"


class RealCursor:
    def __init__(self, cfg):
        self.enabled = cfg.get("enabled", True)
        self.sensitivity = cfg.get("sensitivity", 4.5)
        self.deadzone = cfg.get("deadzone", 0.004)
        self.ema_alpha = cfg.get("ema_alpha", 0.55)
        self.accel = cfg.get("accel", 0.5)
        self.velocity_floor = cfg.get("velocity_floor", 0.018)
        self.x, self.y = 0.5, 0.5
        self._prev_hand = None
        self._recent = deque(maxlen=6)
        self._dx_s = 0.0
        self._dy_s = 0.0
        self._last_sent = 0.0
        self._min_interval = 1.0 / 60.0
        self._w, self._h = 1920, 1080
        self._screen_t = 0.0

    def _screen(self):
        now = time.monotonic()
        if now - self._screen_t < 5.0:
            return self._w, self._h
        try:
            r = subprocess.run(
                ["swaymsg", "-t", "get_outputs", "-r"],
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            for o in json.loads(r.stdout):
                if o.get("focused"):
                    self._w = o["current_mode"]["width"]
                    self._h = o["current_mode"]["height"]
                    break
        except Exception:
            pass
        self._screen_t = now
        return self._w, self._h

    def _sway(self, *args):
        if not self.enabled:
            log.info("[cursor-dry] swaymsg %s", " ".join(args))
            return
        subprocess.run(
            ["swaymsg", *args],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def update(self, hx, hy, now):
        if not self.enabled:
            return
        if self._prev_hand is None:
            self._prev_hand = (hx, hy)
            return

        raw_dx = hx - self._prev_hand[0]
        raw_dy = hy - self._prev_hand[1]
        self._prev_hand = (hx, hy)

        if abs(raw_dx) < self.deadzone:
            raw_dx = 0.0
        if abs(raw_dy) < self.deadzone:
            raw_dy = 0.0

        self._recent.append((raw_dx, raw_dy))
        total_dx = sum(d[0] for d in self._recent)
        total_dy = sum(d[1] for d in self._recent)
        if abs(total_dx) < self.velocity_floor and abs(total_dy) < self.velocity_floor:
            self._dx_s *= 0.5
            self._dy_s *= 0.5
            return

        a = self.ema_alpha
        self._dx_s = self._dx_s * (1 - a) + raw_dx * a
        self._dy_s = self._dy_s * (1 - a) + raw_dy * a

        def _velocity_mult(d):
            mag = abs(d)
            if mag < 0.002:
                return 0.30
            elif mag < 0.006:
                t = (mag - 0.002) / 0.004
                return 0.30 + (1.0 - 0.30) * t
            elif mag < 0.015:
                t = (mag - 0.006) / 0.009
                return 1.0 + (2.5 - 1.0) * t
            elif mag < 0.030:
                t = (mag - 0.015) / 0.015
                return 2.5 + (4.5 - 2.5) * t
            else:
                return 6.0

        dx = _velocity_mult(self._dx_s) * self._dx_s * self.sensitivity
        dy = _velocity_mult(self._dy_s) * self._dy_s * self.sensitivity

        if abs(dx) < 0.0001 and abs(dy) < 0.0001:
            return

        self.x = min(max(self.x + dx, 0.0), 1.0)
        self.y = min(max(self.y + dy, 0.0), 1.0)

        if now - self._last_sent < self._min_interval:
            return
        self._last_sent = now
        w, h = self._screen()
        self._sway(
            "seat", "seat0", "cursor", "set", str(int(self.x * w)), str(int(self.y * h))
        )

    def reset_anchor(self):
        self._prev_hand = None
        self._recent.clear()
        self._dx_s = 0.0
        self._dy_s = 0.0

    def click(self, button="left"):
        code = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}[button]
        if not self.enabled:
            log.info("[cursor-dry] ydotool click %s", code)
            return
        subprocess.run(
            ["ydotool", "click", code],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class Engine:
    def __init__(self, cfg, sway, hud):
        self.cfg = cfg
        self.pc = cfg["pose"]
        self.tfsc = cfg["three_finger_swipe"]
        self.dc = cfg["pinch_drag"]
        self.rc = cfg["resize"]
        self.wsc = cfg["workspaces"]
        self.bindings = cfg["bindings"]
        self.sway, self.hud = sway, hud
        self.cooldown = cfg["loop"]["cooldown_s"]
        self.grace = cfg["loop"]["lost_grace_s"]
        self.hold_grace = cfg["loop"]["hold_grace_s"]
        self.two_grace = cfg["loop"]["two_grace_s"]
        self.tfswipe, self.pdrag, self.resize = (
            ThreeFingerSwipe(),
            PinchDrag(),
            ResizeTracker(),
        )
        self.label_votes = deque(maxlen=3)
        self.holds = {g: Hold() for g in HOLD_GESTURES}
        self.pinch_state = {}
        self.mode = "none"
        self.last_seen = 0.0
        self.last_two = -1e9
        self.last_action = -1e9
        self.debug_lines = []
        self.hud_pose = "No hand"
        self.cursor = RealCursor(cfg.get("pointer", {}))
        self._was_pointing = False
        self._thumb_clicked = False
        self._thumb_dist_ema = None
        self.cursor_armed = False

    def process(self, hands, now):
        self._process(hands, now)
        if self.hud.enabled:
            self.hud.show(f"Workspace {self.sway.current_ws()}\n{self.hud_pose}", now)

    def _process(self, hands, now):
        self.debug_lines = []
        hands = hands[:2]
        if not hands:
            if self.mode != "none" and now - self.last_seen > self.grace:
                self.reset_all(now)
                self.mode = "none"
                self.hud_pose = "No hand"
            return
        self.last_seen = now
        mode = "one" if len(hands) == 1 else "two"
        if mode == "two":
            self.last_two = now
        elif self.mode == "two" and now - self.last_two <= self.two_grace:
            return
        if mode != self.mode:
            self.reset_all(now)
            self.mode = mode
        if mode == "one":
            self._one(hands[0], now)
        else:
            self._two(hands, now)

    def reset_all(self, now):
        for g in self.holds:
            self._hold_update(g, "any", False, now, force=True)
        self.tfswipe.reset()
        self.pdrag.reset()
        self.resize.reset()
        self.pinch_state.clear()
        self.label_votes.clear()
        self._was_pointing = False
        self._thumb_dist_ema = None
        self._thumb_clicked = False
        self.cursor.reset_anchor()

    def _pinch(self, key, h):
        was = self.pinch_state.get(key, False)
        val = (
            False
            if h.fist
            else h.pinch_ratio < (self.pc["pinch_off"] if was else self.pc["pinch_on"])
        )
        self.pinch_state[key] = val
        return val

    def bindings_for(self, g, h):
        return [
            b
            for b in self.bindings
            if b.gesture == g and (b.hand == "any" or b.hand == h)
        ]

    def _one(self, h, now):
        label = (
            max(set(self.label_votes), key=list(self.label_votes).count)
            if self.label_votes
            else h.label
        )
        self.label_votes.append(h.label)
        h.label = label
        h.pinching = self._pinch("one", h)
        self.debug_lines.append(h.describe())
        log.debug(h.describe())

        # Pinky hold toggles cursor arm/disarm.
        hd = self.holds["pinky_hold"]
        if h.pinky_only:
            ev = hd.update(True, now, 0.25)
            if ev == "started":
                log.info("pinky held (arming in 0.25s)")
            elif ev == "fire":
                self.cursor_armed = not self.cursor_armed
                log.info("cursor mode -> %s", "ON" if self.cursor_armed else "OFF")
                self.cursor.reset_anchor()
        else:
            hd.update(False, now, 0.0, self.hold_grace)

        # Cursor: index-finger pointing, armed, not pinching.
        cursor_active = h.pointing and not h.pinching and self.cursor_armed

        if cursor_active:
            self.cursor.update(h.index_tip[0], h.index_tip[1], now)
            self._was_pointing = True
            raw_thumb_dist = dist(h.pts[4], h.pts[5]) / h.size
            if self._thumb_dist_ema is None:
                self._thumb_dist_ema = raw_thumb_dist
            else:
                self._thumb_dist_ema = self._thumb_dist_ema * 0.9 + raw_thumb_dist * 0.1
            if raw_thumb_dist < 0.85 * self._thumb_dist_ema and not self._thumb_clicked:
                self.cursor.click("left")
                self._thumb_clicked = True
            if raw_thumb_dist > 0.95 * self._thumb_dist_ema:
                self._thumb_clicked = False
        else:
            if self._was_pointing:
                self.cursor.reset_anchor()
                self._was_pointing = False
            self._thumb_dist_ema = None
            self._thumb_clicked = False

        self._hold_update("fists_hold", "any", h.fist and not h.pinching, now)
        self._hold_update("thumbs_up_hold", label, h.thumbs_up, now)

        res = self.tfswipe.update(h.three and not h.pinching, h.palm, now, self.tfsc)
        if res:
            self.emit(f"three_finger_swipe_{res}", label, now, cooldown=True)

        if h.pinching:
            res = self.pdrag.update(h.pinching, (h.palm, h.mid), now, self.dc)
            if res:
                self.emit(f"pinch_drag_{res[0]}", label, now, cooldown=not res[1])

        if h.thumbs_up:
            pose = "Thumbs up"
        elif h.pinky_only:
            pose = "Pinky (hold to toggle cursor)"
        elif cursor_active:
            pose = "Pointing (cursor)"
        elif h.pinching:
            pose = "Pinch"
        elif h.fist:
            pose = "Fist"
        elif h.three:
            pose = "Three fingers"
        elif h.palm_open:
            pose = "Open palm (cursor OFF)"
        else:
            pose = "Hand"
        if self._thumb_clicked and cursor_active:
            pose += " - CLICK!"
        if self.cursor_armed:
            pose += " [armed]"
        self.hud_pose = f"{label.capitalize()} hand: {pose}"

    def _two(self, hands, now):
        a, b = sorted(hands, key=lambda h: h.pts[0][0])
        a.pinching, b.pinching = self._pinch("s0", a), self._pinch("s1", b)
        self.debug_lines.append(a.describe())
        self.debug_lines.append(b.describe())
        active = a.pinching and b.pinching
        for g in ("thumbs_up_hold", "fists_hold", "pinky_hold"):
            self._hold_update(g, "any", False, now)
        if active:
            self.hud_pose = "Both pinching - resize"
            r = self.resize.update(a.mid, b.mid, now, self.rc)
            if r:
                self.emit(r, "both", now, cooldown=False)
        else:
            self.resize.reset()
            self.hud_pose = "Two hands"

    def _hold_update(self, g, hand, active, now, force=False):
        h = self.holds[g]
        if not active:
            ev = h.update(False, now, 0.0, self.hold_grace, force)
            if ev == "cancelled" and h.notified:
                h.notified = False
            return
        bs = self.bindings_for(g, hand)
        if not bs:
            return
        need = bs[0].hold_s
        ev = h.update(True, now, need)
        if ev == "started":
            if bs[0].notify:
                subprocess.run(
                    ["notify-send", "-t", str(int(need * 1000) + 600), bs[0].notify],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                h.notified = True
        elif ev == "fire":
            h.notified = False
            self.emit(g, hand, now)

    def emit(self, g, hand, now, cooldown=True):
        if cooldown and now - self.last_action < self.cooldown:
            return
        bs = self.bindings_for(g, hand)
        if not bs:
            return
        log.info("GESTURE %s [%s]", g, hand)
        for b in bs:
            self._run(b)
        self.last_action = now

    def _run(self, b):
        cur = self.sway.current_ws()
        nxt, prv = ws_targets(cur, int(self.wsc["count"]), bool(self.wsc["wrap"]))
        try:
            cmd = b.command.format(cur=cur, next=nxt, prev=prv)
        except Exception as e:
            log.error("[%s] %s", b.name, e)
            return
        for r in self.sway.command(cmd):
            if r.success:
                log.info("sway ok: %s", cmd)
            else:
                log.error("sway rejected: %s", r.error)


class Sway:
    def __init__(self, dry):
        self.dry, self.conn = dry, None

    def _c(self):
        if self.conn is None:
            self.conn = i3ipc.Connection()
        return self.conn

    def current_ws(self):
        try:
            for w in self._c().get_workspaces():
                if w.focused:
                    return w.num
        except Exception:
            self.conn = None
        return 1

    def command(self, cmd):
        if self.dry:
            log.info("[dry-run] sway: %s", cmd)
            return []
        try:
            return self._c().command(cmd)
        except Exception as e:
            log.warning("sway fail: %s", e)
            self.conn = None
            return []


class Hud:
    def __init__(self, enabled, interval):
        self.enabled, self.interval = enabled, interval

    def show(self, body, now):
        if not self.enabled:
            return
        try:
            subprocess.Popen(
                [
                    "notify-send",
                    "-a",
                    "gestured",
                    "-t",
                    "0",
                    "-h",
                    "string:x-canonical-private-synchronous:gestured-hud",
                    "-h",
                    "string:x-dunst-stack-tag:gestured-hud",
                    "Gesture control",
                    body,
                ],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


def open_camera(cc):
    cap = cv2.VideoCapture(cc["index"], cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    if cc.get("fourcc"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cc["fourcc"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cc["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cc["height"])
    cap.set(cv2.CAP_PROP_FPS, cc["device_fps"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def apply_controls(cc):
    controls = cc.get("controls") or {}
    if not controls:
        return
    exe = shutil.which("v4l2-ctl")
    if not exe:
        return
    dev = (
        f"/dev/video{cc['index']}" if isinstance(cc["index"], int) else str(cc["index"])
    )
    for n, v in controls.items():
        subprocess.run(
            [exe, "-d", dev, f"--set-ctrl={n}={v}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def build_hands(res, aspect, cfg):
    out = []
    if not res.multi_hand_landmarks:
        return out
    swap = cfg["detector"]["swap_handedness"]
    for lms, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
        lbl = hd.classification[0].label.lower()
        if swap:
            lbl = "left" if lbl == "right" else "right"
        out.append(Hand(lbl, [(p.x, p.y * aspect) for p in lms.landmark], cfg["pose"]))
    return out


def run(args):
    try:
        cfg = load_config(find_config(args.config))
    except (ConfigError, OSError, tomllib.TOMLDecodeError) as e:
        log.error("config: %s", e)
        return 2

    stop = __import__("threading").Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    hud = Hud(
        cfg["loop"]["notifications"] and cfg["loop"]["hud"],
        cfg["loop"]["hud_min_interval_s"],
    )
    engine = Engine(cfg, Sway(args.dry_run), hud)

    cap = open_camera(cfg["camera"])
    if not cap:
        log.error("camera failed")
        return 1
    apply_controls(cfg["camera"])

    mp_h = mp.solutions.hands
    mp_d = mp.solutions.drawing_utils
    det = mp_h.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=cfg["detector"]["model_complexity"],
        min_detection_confidence=cfg["detector"]["min_detection_confidence"],
        min_tracking_confidence=cfg["detector"]["min_tracking_confidence"],
    )

    lc = cfg["loop"]
    flip = cfg["camera"]["flip"]
    slack = 0.5 / max(cfg["camera"]["device_fps"], 1)
    next_t = 0.0
    last_hand_t = -1e9
    idle = False
    grab_failures = 0
    frames = 0
    stats_t = time.monotonic()

    try:
        while not stop.is_set():
            if not cap.grab():
                grab_failures += 1
                if grab_failures >= 50:
                    log.error("camera gone")
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
            rgb = enhance_for_detection(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            rgb.flags.writeable = False
            res = det.process(rgb)
            hands = build_hands(res, frame.shape[0] / frame.shape[1], cfg)
            if hands:
                last_hand_t = now
            engine.process(hands, now)
            frames += 1
            new_idle = (now - last_hand_t) > lc["idle_after_s"]
            if new_idle != idle:
                idle = new_idle
            next_t = now + 1.0 / (lc["fps_idle"] if idle else lc["fps_active"])
            if args.preview:
                if res.multi_hand_landmarks:
                    for lm in res.multi_hand_landmarks:
                        mp_d.draw_landmarks(frame, lm, mp_h.HAND_CONNECTIONS)
                y = 20
                for line in [
                    f"mode={engine.mode}",
                    f"armed={engine.cursor_armed}",
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
                cv2.imshow("gestured", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop.set()
            if now - stats_t >= lc["stats_every_s"]:
                log.info(
                    "alive: %.1f fps, mode=%s, armed=%s",
                    frames / (now - stats_t),
                    engine.mode,
                    engine.cursor_armed,
                )
                frames = 0
                stats_t = now
    finally:
        try:
            det.close()
        except Exception:
            pass
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()
        log.info("stopped")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
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

