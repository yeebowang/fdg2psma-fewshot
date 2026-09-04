#!/usr/bin/env bash
# Back-compat wrapper → run_fdg_eval_after_fc70_queue_bg.sh
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/ICLR2026/run/run_fdg_eval_after_fc70_queue_bg.sh" "$@"
