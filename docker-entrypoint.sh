#!/bin/sh
set -eu

DISPLAY="${DISPLAY:-:99}"
export DISPLAY
SCREEN="${AISTUDIO_XVFB_SCREEN:-1280x900x24}"

find /app/data/accounts -path '*/profile/Singleton*' -type s -delete 2>/dev/null || true
find /app/data/accounts -path '*/profile/Singleton*' -type f -delete 2>/dev/null || true
find /app/data/accounts -path '*/profile/Singleton*' -type l -delete 2>/dev/null || true

rm -f "/tmp/.X${DISPLAY#:}-lock"
Xvfb "$DISPLAY" -screen 0 "$SCREEN" -ac +extension RANDR >/tmp/xvfb.log 2>&1 &

for _ in $(seq 1 50); do
    if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done

if [ -n "${AISTUDIO_VNC_PASSWORD:-}" ]; then
    mkdir -p /root/.vnc
    x11vnc -storepasswd "$AISTUDIO_VNC_PASSWORD" /root/.vnc/passwd >/dev/null
    x11vnc -display "$DISPLAY" -forever -shared -listen 0.0.0.0 -rfbport 5900 -rfbauth /root/.vnc/passwd >/tmp/x11vnc.log 2>&1 &
else
    x11vnc -display "$DISPLAY" -forever -shared -listen 0.0.0.0 -rfbport 5900 -nopw >/tmp/x11vnc.log 2>&1 &
fi

exec "$@"
