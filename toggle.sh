#!/usr/bin/env bash
# Start gestured if it is not running, stop it if it is. Bind this to $mod+Shift+g.
# Extra arguments are passed to gestured.py on start (e.g. --dry-run).

DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PIDFILE="${XDG_RUNTIME_DIR:-/tmp}/gestured.pid"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/gestured"
LOG="$STATE/gestured.log"
PY="$DIR/.venv/bin/python"

mkdir -p "$STATE"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    kill -TERM "$(cat "$PIDFILE")"
    exit 0   # the daemon removes its pidfile and shows the "stopped" notification
fi

[[ -x "$PY" ]] || PY="$(command -v python3)"

mv -f "$LOG" "$LOG.prev" 2>/dev/null
setsid -f "$PY" "$DIR/gestured.py" "$@" >"$LOG" 2>&1 </dev/null
