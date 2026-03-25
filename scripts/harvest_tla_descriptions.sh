#!/usr/bin/env bash
# Clone / update upstream TLA+ repos and write data/derived/tla_descriptions.json
#
# Run from anywhere:
#   bash /path/to/ChatTLA/scripts/harvest_tla_descriptions.sh
# Or from repo root:
#   bash scripts/harvest_tla_descriptions.sh
# Expect several minutes (SANY/static extract over all coarse modules).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
echo "Harvesting into $REPO_ROOT/data/derived/ (this takes a few minutes)..." >&2
python3 scripts/tla_description_sources/harvest_descriptions.py --update-repo -v
python3 scripts/tla_description_sources/audit_descriptions.py
