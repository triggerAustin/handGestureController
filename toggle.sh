#!/usr/bin/env bash
# Start gestured if it is not running, stop it if it is. Bind this to $mod+Shift+g.
# Extra arguments are passed to gestured.py on start (e.g. --dry-run).

DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PIDFILE="${XDG_RUNTIME_DIR:-/tmp}/gestured.pid"
OVERLAY_PIDFILE="${XDG_RUNTIME_DIR:-/tmp}/gestured-overlay.pid"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/gestured"
LOG="$STATE/gestured.log"
OVERLAY_LOG="$STATE/overlay.log"
PY="$DIR/.venv/bin/python"

mkdir -p "$STATE"

stop_all() {
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
        kill -TERM "$(cat "$PIDFILE")"
    fi
    if [[ -f "$OVERLAY_PIDFILE" ]]; then
        kill -TERM "$(cat "$OVERLAY_PIDFILE")" 2>/dev/null
        rm -f "$OVERLAY_PIDFILE"
    fi
    rm -f "$PIDFILE"
}

# Running? Stop and exit.
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    stop_all
    exit 0
fi

# Clean up leftovers from a previous failed run.
stop_all

[[ -x "$PY" ]] || PY="$(command -v python3)"

mv -f "$LOG" "$LOG.prev" 2>/dev/null
mv -f "$OVERLAY_LOG" "$OVERLAY_LOG.prev" 2>/dev/null

# Start backend. nohup keeps it alive; $! is the real python PID.
nohup "$PY" "$DIR/gestured.py" "$@" >"$LOG" 2>&1 </dev/null &
echo $! > "$PIDFILE"

# Wait for the socket to appear (max 3 s) before starting the overlay.
SOCK="${XDG_RUNTIME_DIR:-/tmp}/gestured-pointer.sock"
for _ in $(seq 1 30); do
    [[ -S "$SOCK" ]] && break
    sleep 0.1
done

# Start overlay under system python (GObject/PyGObject is a system package).
nohup python3 "$DIR/overlay.py" >"$OVERLAY_LOG" 2>&1 </dev/null &
echo $! > "$OVERLAY_PIDFILE"