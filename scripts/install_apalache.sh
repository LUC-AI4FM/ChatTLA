#!/bin/bash
# Install Apalache symbolic model checker into src/shared/apalache/
# Used as a tie-breaker validator for diamond-tier specs (requires type
# annotations on VARIABLES — see src/validators/apalache_validator.py).

set -euo pipefail

VERSION="${APALACHE_VERSION:-0.56.1}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_ROOT/src/shared/apalache"

if [ -f "$DEST/bin/apalache-mc" ]; then
    echo "[install_apalache] Already installed at $DEST"
    exit 0
fi

mkdir -p "$DEST"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "[install_apalache] Downloading Apalache $VERSION..."
curl -sL -o "$TMPDIR/apalache.tgz" \
    "https://github.com/informalsystems/apalache/releases/download/v$VERSION/apalache-$VERSION.tgz"

echo "[install_apalache] Extracting to $DEST..."
tar xzf "$TMPDIR/apalache.tgz" -C "$TMPDIR"
cp -r "$TMPDIR/apalache-$VERSION"/* "$DEST/"

echo "[install_apalache] Done. Test with: $DEST/bin/apalache-mc --help"
