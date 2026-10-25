# gestured - hand-gesture control for Sway

Webcam -> MediaPipe Hands -> gesture detectors -> Sway IPC + ydotool.
Toggle it on and off with `$mod+Shift+g`.

## Low-light note

`gestured.py` applies **CLAHE + a mild gamma lift** to every frame before handing it to
MediaPipe, and `config.toml` sets manual exposure + brightness + gamma on the camera at
startup (`auto_exposure=1`, `exposure_time_absolute=400`, `brightness=170`, `gamma=135`,
`backlight_compensation=0` by default). This makes detection usable in dim rooms.
If you still get nothing, add light — no software fix beats a lamp in front of you.

## Gestures (all remappable in `config.toml`)

| Gesture | Action | Notes |
|---|---|---|
| Open palm, held still, then sway right/left | workspace N+1 / N-1 (wraps 10 -> 1) | Unchanged |
| Three-finger pose (thumb+index+middle, ring+pinky curled), held still, then sway right/left | move focused window to N+1 / N-1 | Moves the window **and follows it** |
| Thumb+index pinch, then move/wave right/left | send focused window to N+1 / N-1 | **Works whether cursor is armed or not.** Cursor freezes while pinching |
| Single fist, held **3 s** | close focused window | Long hold prevents accidents; toast shown at ~3 s |
| Both hands pinching, spread/squeeze | resize width or height in 10 px steps | Unchanged |
| Thumbs up ~0.25 s, either hand | press Enter in the focused window | Unchanged |
| **Pinky, held ~0.25 s** | **arm / disarm cursor mode** | Logs `cursor mode -> ON/OFF`; HUD shows `[armed]` |
| **Index finger only, pointing** | **cursor follows fingertip** (only when armed) | Uses `swaymsg seat seat0 cursor set` |
| **Thumb flick inward** (while pointing, armed) | **left click** | `ydotool click 0xC0` |

Rules:
- Cursor is off by default. Hold the pinky to arm it, hold again to disarm.
- While armed: pointing moves the cursor; a thumb flick clicks.
- While armed: pinching still moves the window — the cursor freezes during the pinch.
- Hold poses still for ~0.3 s (log: `palm armed` / `three-finger swipe armed`) before swaying.
- The cursor won't move while pinching, so pinch and cursor never fight.

### The three "families"

- **Sway** — switch workspace (open palm) or carry a window with you (three fingers).
- **Pinch** — manipulate a window: send it away without following (pinch-drag), or resize it
  (both hands pinching). Pinch-drag still fires even while the cursor is armed; the cursor
  freezes during the pinch and resumes on release.
- **Hold** — committed actions: fist = close window (3 s), thumbs up = accept.
- **Pointer** — pinky arms, index points, thumb flicks to click.

### Cursor mode (the velocity curve)

The cursor response has three gears, blending smoothly:

| Hand motion (fraction of frame / frame) | Cursor speed | Use case |
|---|---|---|
| < 0.2% | 0.30× | Precision — land on small buttons |
| 0.2% – 0.6% | 0.30 → 1.0× | Fine control |
| 0.6% – 1.5% | 1.0 → 2.5× | Normal |
| 1.5% – 3.0% | 2.5 → 4.5× | Reaching |
| > 3.0% | 6.0× | Sweeping across the screen |

Tuning knobs are in `[pointer]` in `config.toml` (see below).

## Setup

### 1. System packages
- Arch: `sudo pacman -S libnotify mako v4l-utils uv ydotool` (plus a notification daemon)
- Fedora: `sudo dnf install libnotify dunst v4l-utils python3.12 ydotool`, then install `uv`
- Debian/Ubuntu: `sudo apt install libnotify-bin dunst v4l-utils python3.12 python3.12-venv ydotool`,
  then install `uv` (mediapipe has no wheels for 3.13+)

Camera access: your user must be in the `video` group (`groups`), and `ls /dev/video*` should
list a device. Log out/in after adding the group.

`ydotool` needs `/dev/uinput`:

    sudo usermod -aG input "$USER"
    # log out/in

Then create a user systemd unit for `ydotoold` (Fedora doesn't ship one):

    mkdir -p ~/.config/systemd/user
    cat > ~/.config/systemd/user/ydotoold.service << 'EOF'
    [Unit]
    Description=ydotool daemon
    [Service]
    ExecStart=/usr/bin/ydotoold --socket-path=%t/.ydotool_socket --socket-own=%U:%G
    Restart=on-failure
    [Install]
    WantedBy=default.target
    EOF
    systemctl --user daemon-reload
    systemctl --user enable --now ydotoold.service
    ydotool click 0xC0   # should produce a real left click

### 2. Python 3.12 virtualenv

    mkdir -p ~/.local/share/gestured
    cd ~/.local/share/gestured
    # put gestured.py, config.toml, toggle.sh, requirements.txt, README.md here
    chmod +x toggle.sh gestured.py
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt

### 3. Config

    mkdir -p ~/.config/gestured
    cp config.toml ~/.config/gestured/config.toml

### 4. First run: dry-run with preview

    cd ~/.local/share/gestured
    .venv/bin/python gestured.py --dry-run --preview -v

The preview needs XWayland. The window shows per-frame finger states, poses, and the
`armed=True/False` cursor state so you can confirm pinky detection works. Press `q` in
the window or Ctrl+C to quit. Lines tagged `[dry-run]` show the Sway commands that
would have run.

### 5. Sway bind

Add to `~/.config/sway/config`, then `swaymsg reload`:

    bindsym $mod+Shift+g exec ~/.local/share/gestured/toggle.sh

(`exec`, not `exec_always`, so a reload does not restart it.) Press the bind to start,
again to stop. Extra args work: `exec ~/.local/share/gestured/toggle.sh --dry-run`.

### 6. Logs

    tail -f ~/.local/state/gestured/gestured.log

Startup steps, camera info, mode changes, pinch/hold progress, cursor arm/disarm events,
every fired gesture with the exact Sway command, and a heartbeat every 10 s.
Previous session is kept as `gestured.log.prev`.

## Tuning (in `config.toml`)

### Camera and detection

| Symptom | Setting |
|---|---|
| Resize pinch not detected / too sticky | `[pose] pinch_on`, `pinch_off` |
| Palm never arms | `[swipe] still_max` up, `arm_s` down, `[pose] palm_min_fingers` 3 -> 2 |
| Three-finger swipe never arms | `[three_finger_swipe] still_max` up, `arm_s` down, `[pose] thumb_extend` down |
| Fingers misread as curled/extended | `[pose] finger_extend` (lower = easier) |
| Left and right hands swapped | `[detector] swap_handedness = true` |
| Hands flicker between poses | `[detector] model_complexity = 1` (Full model) |
| Too much CPU | lower `fps_active`, `fps_idle`; keep `model_complexity = 0` |

### Swipe / pinch / hold gestures

| Symptom | Setting |
|---|---|
| Swipes fire by accident | `[swipe] arm_s` up, `min_distance` up |
| Sway does not fire | `[swipe] min_distance` down, `window_s` up |
| Opposite swipe fires on return | `[swipe] rearm_s` up |
| Pinch-drag fires before pinch finishes | `[pinch_drag] settle_s` up |
| Pinch-drag too far to trigger | `[pinch_drag] min_distance` down |
| Pinch-drag fires on diagonal | `[pinch_drag] axis_ratio` up |
| Pinch-drag keeps stepping | `[pinch_drag] scrub = false` |
| Fist kill fires too easily | its `hold_s` up |
| Fist kill feels too slow | `hold_s = 2.0` |
| Pinky doesn't arm cursor | its `hold_s` down to `0.15` |
| Pinky arms by accident | its `hold_s` up to `0.4` |

### Cursor (`[pointer]`)

| Symptom | Setting |
|---|---|
| Cursor too fast / overshoots | `sensitivity` down (try `3.0`) |
| Cursor can't cross the screen | `sensitivity` up (try `6.0`), `accel` up |
| Cursor jittery when hand is still | `deadzone` up (try `0.006`), `velocity_floor` up |
| Cursor laggy / behind fingertip | `ema_alpha` up (try `0.65`), `velocity_floor` down |
| Can't do fine positioning | `velocity_floor` up (`0.025`), `sensitivity` down |
| Precision feels too coarse | lower the `< 0.002` threshold in `_velocity_mult` |
| Need faster sweeps | raise the `< 0.030` cap in `_velocity_mult` |

### Click (thumb flick)

The click uses distance from the thumb tip (landmark 4) to the index MCP (landmark 5).
The threshold is `raw_thumb_dist < 0.85 * _thumb_dist_ema` inside `_one()`. The reset
threshold is `> 0.95 * _thumb_dist_ema`.

- **Clicks too easily** (fires while pointing): change `0.85` to `0.78`.
- **Never clicks**: change `0.85` to `0.90`.
- **Multiple clicks per flick**: raise the reset from `0.95` to `1.0`.
- **Doesn't reset between clicks**: lower reset to `0.92`.

## Architecture

`gestured.py` is a single file, deliberately. The critical paths are:

- **`Hand`** — per-frame feature extraction from the 21 MediaPipe landmarks
  (pinch ratio, finger-extended flags, pointing/palm/fist/thumbs-up/three/pinky poses,
  thumb-flick distance).
- **`RealCursor`** — receives index fingertip position, runs the velocity curve,
  sends `swaymsg seat seat0 cursor set X Y` and `ydotool click 0xC0`.
- **`Engine._one` / `Engine._two`** — dispatch loop. One-hand gestures go through `_one`,
  two-hand through `_two`. Cursor gating is `h.pointing and not h.pinching and self.cursor_armed`.
- **`Engine._hold_update`** — timing state machine for `fists_hold`, `thumbs_up_hold`,
  `pinky_hold`. Fires the binding's `command` after `hold_s` seconds.
- **`Sway`** — i3ipc wrapper. `run(cmd)` executes; `dry_run` logs.
- **`Hud`** — one persistent `notify-send` notification showing workspace + pose + armed state.

There is no overlay window, no socket, no GTK dependency. The real system cursor *is*
the cursor.

## Troubleshooting

- **`could not open camera`**: wrong `[camera] index`, no `video` group membership,
  or another app holds the camera.
- **`ImportError` for mediapipe**: check `.venv/bin/python --version` is 3.9-3.12.
- **`RuntimeError: ValidatedGraphConfig Initialization failed`**: `protobuf` is out of
  range. MediaPipe 0.10.14 wants `protobuf>=4.25.3,<5`. Fix with
  `uv pip install --python .venv/bin/python 'protobuf>=4.25.3,<5'`.
- **Swipe direction feels inverted**: `[camera] flip = false` or camera is mounted
  unusually.
- **No notifications**: needs `notify-send` and a running daemon (mako/dunst).
- **Dark image / `hands in 0/N frames`**: raise `exposure_time_absolute` (but not past
  ~600 or hands blur) or `brightness`, or add light. Check the startup log for
  `camera control exposure_time_absolute=...`; if it says `not applied`, the camera
  rejected it.
- **Cursor doesn't move**: `systemctl --user status ydotoold.service` and `ydotool click 0xC0`
  from a terminal. If `ydotoold` isn't running, or you're not in `input`, `swaymsg seat
  seat0 cursor set` still works for movement but clicks won't fire.
- **Cursor moves but never clicks**: thumb flick threshold too tight. Change `0.85` to `0.90`.
- **Pinky won't arm**: check the preview HUD for `Pinky (hold to toggle cursor)` in the
  log. If it never shows `pinky`, `pinky_hold` is failing — raise `finger_extend` or
  the `pinky_only` threshold.
- **`NameError: name 'np' is not defined`**: missing `import numpy as np` near the top of
  `gestured.py`.