# gestured - hand-gesture control for Sway

Webcam -> MediaPipe Hands -> gesture detectors -> Sway IPC. No key injection.
Toggle it on and off with `$mod+Shift+g`.

## Gestures (all remappable in `config.toml`)

| Gesture | Action |
|---|---|
| Right-hand open palm, held still, then sway right/left | workspace N+1 / N-1 (wraps 10 -> 1) |
| Left-hand open palm, held still, then sway right/left | move window to N+1 / N-1 and follow it (wraps) |
| Both fists held ~0.6 s | kill window (notification first; open hands to cancel) |
| Both hands pinching, spread/squeeze | resize width or height in 10 px steps |
| Thumb + index + middle out, ring + pinky curled, ~0.7 s, either hand | floating toggle |

Rules: hold the open palm still for ~0.3 s (log: `palm armed`), then sway; one swipe per arming,
and the hand must rest ~0.6 s before the next (so the return stroke does not fire the opposite way).
Any frame with two hands goes only to the two-hand handler.

## Setup

### 1. System packages
- Arch: `sudo pacman -S libnotify uv` (plus a notification daemon, e.g. `mako`)
- Debian/Ubuntu: `sudo apt install libnotify-bin` and install `uv` (or use pyenv)

Camera access: your user must be in the `video` group (`groups`), and `ls /dev/video*` should list a device.

### 2. Python 3.12 virtualenv
MediaPipe 0.10.14 has wheels only up to Python 3.12. If your system Python is newer, use uv:

    mkdir -p ~/.local/share/gestured
    cp gestured.py config.toml toggle.sh requirements.txt README.md ~/.local/share/gestured/
    cd ~/.local/share/gestured
    chmod +x toggle.sh gestured.py
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt

(Without uv: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`.)

### 3. Config

    mkdir -p ~/.config/gestured
    cp config.toml ~/.config/gestured/config.toml

### 4. First run: dry-run with a preview window (nothing is sent to Sway)

    cd ~/.local/share/gestured
    .venv/bin/python gestured.py --dry-run --preview -v

The preview needs XWayland (if it fails: `QT_QPA_PLATFORM=xcb`). Overlay shows live pinch ratio, which fingers
count as extended, and the detected pose, so you can tune thresholds. Press `q` in the window or Ctrl+C to quit.
Lines tagged `[dry-run]` show the Sway commands that would have run.

### 5. Sway bind
Add to `~/.config/sway/config`, then `swaymsg reload`:

    bindsym $mod+Shift+g exec ~/.local/share/gestured/toggle.sh

(`exec`, not `exec_always`, so a reload does not restart it.) Press the bind to start, again to stop.
Extra args work: `exec ~/.local/share/gestured/toggle.sh --dry-run`.

### 6. Logs

    tail -f ~/.local/state/gestured/gestured.log

Startup steps, camera info, mode changes (active/idle), pinch/hold progress, every fired gesture with the exact
Sway command, and a heartbeat every 10 s. Previous session is kept as `gestured.log.prev`.

## Tuning (in `config.toml`)

| Symptom | Setting |
|---|---|
| Resize pinch not detected / too sticky | `[pose] pinch_on`, `pinch_off` |
| Palm never arms | `[swipe] still_max` up, `arm_s` down, `[pose] palm_min_fingers` 3 -> 2 |
| Sway does not fire | `[swipe] min_distance` down, `window_s` up |
| Swipes fire by accident | `[swipe] arm_s` up, `min_distance` up |
| Opposite swipe fires on the return | `[swipe] rearm_s` up |
| Three-finger pose flaky (thumb) | `[pose] thumb_extend` (lower = easier) |
| Fingers misread as curled/extended | `[pose] finger_extend` |
| Fists not recognized | `[pose] finger_extend` up a little, `fist_min_pinch` down |
| Diagonal motion fires | `[swipe] axis_ratio` up |
| Resize too jumpy / too slow | `[resize] step_norm`, `min_interval_s` |
| Left and right hands swapped | `[detector] swap_handedness = true` |
| Too much CPU | lower `fps_active`, `fps_idle`; keep `model_complexity = 0` |

## Troubleshooting
- `could not open camera`: wrong `[camera] index`, no `video` group membership, or another app holds the camera.
- `ImportError` / install failure for mediapipe: check `.venv/bin/python --version` is 3.9-3.12.
- Swipe direction feels inverted: you probably have `flip = false` or the camera is mounted unusually.
- No notifications: needs `notify-send` and a running daemon (mako/dunst); gestures still work without it.
