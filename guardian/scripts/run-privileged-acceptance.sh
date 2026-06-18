#!/usr/bin/env bash
# Privileged acceptance for Familiar's Linux body. Needs root: CAP_SYS_ADMIN for
# the fanotify helper, CAP_NET_ADMIN for the daemon. The network side runs in a
# throwaway netns so the host firewall is untouched.
#
#   sudo ./scripts/run-privileged-acceptance.sh
#
# Builds as the invoking user (cargo/target stay user-owned), then runs the
# daemon + helper + a scripted exfil in a netns and asserts an autonomous drop
# rule was installed and the audit recorded Detection + Actuation. This one run
# exercises the whole stack: fanotify (helper, CAP_SYS_ADMIN), NFQUEUE sensing,
# /proc attribution, read→outbound linking, and the reversible netlink block.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD

if [[ $EUID -ne 0 ]]; then
    echo "needs root; re-run: sudo $0" >&2
    exit 1
fi

RUN_USER=${SUDO_USER:-$USER}
echo "== building daemon + helper as $RUN_USER (keeps target/ user-owned) =="
sudo -u "$RUN_USER" bash -lc "cd '$REPO' && cargo build -p familiar-daemon -p familiar-fanotify-helper"

DAEMON="$REPO/target/debug/familiar-daemon"
HELPER="$REPO/target/debug/familiar-fanotify-helper"
[[ -x "$DAEMON" && -x "$HELPER" ]] || { echo "build did not produce the binaries" >&2; exit 1; }

# Absolute python3 (root's PATH may lack the user's); used for the test listener.
PY=$(command -v python3 || true)
[[ -x "$PY" ]] || PY=$(sudo -u "$RUN_USER" bash -lc 'command -v python3' || true)
[[ -x "$PY" ]] || { echo "need python3 for the test listener" >&2; exit 1; }

echo "== full-stack autonomous-exfil acceptance (netns) =="
WORK=$(mktemp -d /tmp/familiar-acc.XXXXXX)
STATE="$WORK/state"; SENS="$WORK/watched"; SOCK="$WORK/fileread.sock"
mkdir -p "$STATE" "$SENS"
echo secret > "$SENS/secret"

# Arm sensors + detector + block actuator (capability snapshot the daemon loads).
cat > "$STATE/capabilities.json" <<JSON
{ "states": {
  "SensorOutboundConn": true, "SensorSensitiveRead": true,
  "DetectorExfil": true, "ActuatorBlockConn": true, "ActuatorFreezeProcess": false
} }
JSON

cat > "$WORK/config.json" <<JSON
{ "sensitive_prefixes": ["$SENS"], "established_dsts": [], "link_window_ms": 5000,
  "permission_timeout_ms": 30000, "queue_num": 0, "tick_ms": 100,
  "state_dir": "$STATE", "cgroup_root": "/sys/fs/cgroup", "helper_socket": "$SOCK",
  "desktop_notify": false }
JSON

NS=familiar-acc-$$
ip netns add "$NS"
cleanup() { kill "${DPID:-0}" "${HPID:-0}" "${LPID:-0}" 2>/dev/null || true; ip netns del "$NS" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT
ip netns exec "$NS" ip link set lo up

# Daemon in the netns (CAP_NET_ADMIN via root); helper is global (fanotify is not
# netns-scoped) and writes FileRead events to the daemon's socket. Capture both
# logs so a failure is diagnosable.
ip netns exec "$NS" "$DAEMON" "$WORK/config.json" >"$WORK/daemon.log" 2>&1 &
DPID=$!
sleep 2
"$HELPER" "$SOCK" "$SENS" >"$WORK/helper.log" 2>&1 &
HPID=$!
sleep 1

# A listener so the exfil connection ESTABLISHES and stays open — a refused
# connection vanishes before /proc attribution can scan it (the documented race).
ip netns exec "$NS" "$PY" -c 'import socket,time
s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.bind(("127.0.0.1",9999)); s.listen()
c,_=s.accept(); time.sleep(6)' &
LPID=$!
sleep 1

# Scripted exfil by ONE process (so the read and the outbound share a pid, which
# is how the detector links them): this bash opens the secret via redirect
# (fanotify sees this bash pid), waits for the FileRead to reach the daemon, then
# the SAME pid connects out and holds the socket open for attribution.
#
# Wrapped in `timeout`: once the daemon links the read→outbound it installs the
# drop rule, which drops this connection's packets, so the connect/hold blocks
# with no RST. That is expected — the block fires from the queued SYN, not from
# the handshake completing — so we cap the target and proceed to the assertions.
timeout 8 ip netns exec "$NS" bash -c "
  read -r _ < '$SENS/secret'
  sleep 1
  exec 3<>/dev/tcp/127.0.0.1/9999 || true
  read -t 4 -u 3 _ || true
" 2>/dev/null || true
sleep 1

echo "--- ruleset in netns ---"
ip netns exec "$NS" nft list ruleset || true
RC=0
if ip netns exec "$NS" nft list ruleset 2>/dev/null | grep -q drop; then
    echo "PASS: autonomous drop rule installed"
else
    echo "FAIL: no drop rule"; RC=1
fi

echo "--- audit kinds ---"
grep -oE '"kind":"[A-Za-z]+"' "$STATE/audit.jsonl" 2>/dev/null | sort | uniq -c || echo "(no audit log)"
if grep -q '"kind":"Actuation"' "$STATE/audit.jsonl" 2>/dev/null && grep -q '"kind":"Detection"' "$STATE/audit.jsonl" 2>/dev/null; then
    echo "PASS: audit recorded Detection + Actuation"
else
    echo "FAIL: audit missing Detection/Actuation"; RC=1
fi

if [[ $RC -ne 0 ]]; then
    echo "--- daemon.log ---"; cat "$WORK/daemon.log" 2>/dev/null || echo "(none)"
    echo "--- helper.log ---"; cat "$WORK/helper.log" 2>/dev/null || echo "(none)"
    echo "--- capabilities.json (armed set the daemon loaded) ---"; cat "$STATE/capabilities.json"
    echo "--- /proc/net/tcp in ns (was the exfil socket attributable?) ---"
    ip netns exec "$NS" cat /proc/net/tcp 2>/dev/null | awk 'NR==1 || /270F|2710/{print}' || true
fi

[[ $RC -eq 0 ]] && echo "== privileged acceptance PASSED ==" || echo "== privileged acceptance FAILED =="
exit $RC
