#!/bin/bash
# Periodic Exp2 Stage-1 monitor. Prints a heartbeat every interval and emits an
# ALERT line when a run newly completes, a task fails, or coverage stops being
# exact. Intended to run in the background and be tailed or pattern-matched.
set -uo pipefail

cd /gpfs/data/bbj-lab/users/daniel/input-representation-benchmark

INTERVAL="${INTERVAL:-600}"
STATE_FILE="${STATE_FILE:-/tmp/exp2_monitor_state.txt}"
touch "${STATE_FILE}"

snapshot() {
  # One line per exp2 task: index|state  (COMPLETED / RUNNING / PENDING / FAILED)
  {
    squeue -r -h -u "$(whoami)" -o '%i|%j|%T' 2>/dev/null \
      | awk -F'|' '$2 ~ /exp2/ && $3 != "" { split($1,a,"_"); t=a[length(a)]; if (t ~ /^[0-9]+$/) print t"|"$3 }'
    sacct -X -n -P -S 2026-08-03T00:00:00 -u "$(whoami)" --format=JobID,JobName,State 2>/dev/null \
      | awk -F'|' '$2 ~ /ml4h-exp2/ && $3=="COMPLETED" { split($1,a,"_"); t=a[length(a)]; if (t ~ /^[0-9]+$/) print t"|COMPLETED" }'
  } | sort -u
}

echo "[exp2-monitor] armed at $(date '+%F %T %Z'); interval=${INTERVAL}s; state=${STATE_FILE}"

snapshot > "${STATE_FILE}.cur"
mv "${STATE_FILE}.cur" "${STATE_FILE}"

while true; do
  sleep "${INTERVAL}"
  snapshot > "${STATE_FILE}.cur"

  completed_now=$(awk -F'|' '$2=="COMPLETED"{print $1}' "${STATE_FILE}.cur" | sort -un | wc -l)
  running_now=$(awk -F'|' '$2=="RUNNING"{print $1}' "${STATE_FILE}.cur" | sort -u | wc -l)
  pending_now=$(awk -F'|' '$2=="PENDING"{print $1}' "${STATE_FILE}.cur" | sort -u | wc -l)

  # Newly completed since last tick.
  new_done=$(comm -13 \
    <(awk -F'|' '$2=="COMPLETED"{print $1}' "${STATE_FILE}" | sort -u) \
    <(awk -F'|' '$2=="COMPLETED"{print $1}' "${STATE_FILE}.cur" | sort -u))
  if [ -n "${new_done}" ]; then
    echo "[exp2-monitor] $(date '+%F %T') ALERT newly completed: $(echo "${new_done}" | tr '\n' ' ')"
  fi

  # Coverage check (its own tail already says OK / ACTION NEEDED).
  cov=$(python utilities/scripts/verify_exp2_coverage.py 2>&1)
  if ! echo "${cov}" | grep -q "active or completed exactly once"; then
    echo "[exp2-monitor] $(date '+%F %T') ALERT coverage not exact:"
    echo "${cov}" | grep -E 'missing|duplicated|covered configurations' | sed 's/^/[exp2-monitor]   /'
  fi

  echo "[exp2-monitor] $(date '+%F %T') heartbeat done=${completed_now}/72 running=${running_now} pending~${pending_now}"

  mv "${STATE_FILE}.cur" "${STATE_FILE}"
done
