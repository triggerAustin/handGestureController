# gestured - hand-gesture control for Sway

Webcam -> MediaPipe Hands -> gesture detectors -> Sway IPC. No key injection.
Toggle it on and off with `$mod+Shift+g`.

## Low-light note

`gestured.py` applies **CLAHE + a mild gamma lift** to every frame before handing it to
MediaPipe, and `config.toml` sets manual exposure + brightness + gamma on the camera at
startup (`auto_exposure=1`, `exposure_time_absolute=400`, `brightness=170`, `gamma=135`,
`backlight_compensation=0` by default). This makes detection usable in dim rooms.
If you still get nothing, add light — no software fix beats a lamp in front of you.

## Gestures (all remappable in `config.toml`)

| Gesture | Action | Follows? |
|---|---|---|
| Open palm, held still, then sway right/left | workspace N+1 / N-1 (wraps 10 -> 1) | N/A |
| Three-finger pose (thumb+index+middle, ring+pinky curled), held still, then sway right/left | move focused window to N+1 / N-1 | **Yes** (you go too) |
| Thumb+index pinch, then move/wave right/left | send focused window to N+1 / N-1 | **No** (you stay) |
| Single fist, held ~0.5 s, either hand | close focused window (toast first; open hand to cancel) | N/A |
| Both hands pinching, spread/squeeze | resize width or height in 10 px steps | N/A |
| Thumbs up ~0.25 s, either hand | press Enter in the focused window | N/A |

Rules: hold the pose still for a moment (log: `palm armed` / `three-finger swipe armed`),
then sway; one swipe per arming, and the hand must rest ~0.6 s before the next.
Pinch-drag responds only to a true thumb+index pinch (a fist does nothing).
Any frame with two hands goes only to the two-hand handler (resize).
The old two-fist kill is not bound by default — a single fist now closes the window,
with a 0.5 s toast that you can cancel by opening your hand.

### The three "families"

- **Sway** — switch workspace (open palm) or *carry* a window with you (three fingers).
- **Pinch** — manipulate a window: *send it away without following* (pinch-drag) or
  *resize it* (both hands pinching).
- **Hold** — committed actions: fist = close, thumbs up = accept.

## Setup

### 1. System packages
- Arch: `sudo pacman -S libnotify mako v4l-utils uv` (plus a notification daemon)
- Fedora: `sudo dnf install libnotify dunst v4l-utils python3.12`, then install `uv`
- Debian/Ubuntu: `sudo apt install libnotify-bin dunst v4l-utils python3.12 python3.12-venv`,
  then install `uv` (mediapipe has no wheels for 3.13+)

Camera access: your user must be in the `video` group (`groups`), and `ls /dev/video*` should list a device.
Log out/in after adding the group.

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

### 4. First run: dry-run with preview (nothing is sent to Sway)

    cd ~/.local/share/gestured
    .venv/bin/python gestured.py --dry-run --preview -v

The preview needs XWayland. Overlay shows live pinch ratio, finger states, and pose so
you can tune thresholds. Press `q` in the window or Ctrl+C to quit.
Lines tagged `[dry-run]` show the Sway commands that would have run.

### 5. Sway bind

Add to `~/.config/sway/config`, then `swaymsg reload`:

    bindsym $mod+Shift+g exec ~/.local/share/gestured/toggle.sh

(`exec`, not `exec_always`, so a reload does not restart it.) Press the bind to start, again to stop.
Extra args work: `exec ~/.local/share/gestured/toggle.sh --dry-run`.

### 6. Logs

    tail -f ~/.local/state/gestured/gestured.log

Startup steps, camera info, mode changes (active/idle), pinch/hold progress, every fired
gesture with the exact Sway command, and a heartbeat every 10 s. Previous session is kept
as `gestured.log.prev`.

## Tuning (in `config.toml`)

| Symptom | Setting |
|---|---|
| Resize pinch not detected / too sticky | `[pose] pinch_on`, `pinch_off` |
| Palm never arms | `[swipe] still_max` up, `arm_s` down, `[pose] palm_min_fingers` 3 -> 2 |
| Sway does not fire | `[swipe] min_distance` down, `window_s` up |
| Swipes fire by accident | `[swipe] arm_s` up, `min_distance` up |
| Opposite swipe fires on the return | `[swipe] rearm_s` up |
| Three-finger swipe never arms | `[three_finger_swipe] still_max` up, `arm_s` down, `[pose] thumb_extend` down |
| Three-finger swipe too sensitive | `[three_finger_swipe] arm_s` up, `min_distance` up |
| Pinch-drag fires before you finish the pinch | `[pinch_drag] settle_s` up |
| Pinch-drag too far to trigger | `[pinch_drag] min_distance` down |
| Pinch-drag fires on diagonal | `[pinch_drag] axis_ratio` up |
| Pinch-drag keeps stepping | `[pinch_drag] scrub = false` |
| Single-fist kill too easy | its `hold_s` up (0.7 or 0.8) |
| Fingers misread as curled/extended | `[pose] finger_extend` (lower = easier to call extended) |
| Left and right hands swapped | `[detector] swap_handedness = true` |
| Too much CPU | lower `fps_active`, `fps_idle`; keep `model_complexity = 0` |

## Troubleshooting
- `could not open camera`: wrong `[camera] index`, no `video` group membership, or another app holds the camera.
- `ImportError` / install failure for mediapipe: check `.venv/bin/python --version` is 3.9-3.12.
- Swipe direction feels inverted: you probably have `flip = false` or the camera is mounted unusually.
- No notifications: needs `notify-send` and a running daemon (mako/dunst); gestures still work without it.
- Dark image / `hands in 0/N frames`: raise `exposure_time_absolute` (but not past ~600 or hands blur) or `brightness`,
  or add light. Also check `camera control exposure_time_absolute=...` appears in the startup log; if it says
  `not applied`, the camera rejected it.
- `NameError: name 'np' is not defined`: the file is missing `import numpy as np` near the top.