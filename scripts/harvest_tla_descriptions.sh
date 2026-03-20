#!/usr/bin/env bash
# Clone / update upstream TLA+ repos and write data/derived/tla_descriptions.json
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
python3 scripts/tla_description_sources/harvest_descriptions.py --update-repo -v
python3 scripts/tla_description_sources/audit_descriptions.py
