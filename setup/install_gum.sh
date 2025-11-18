#! /bin/bash
set -euo pipefail

# Pin a version so builds are reproducible
GUM_VERSION="${GUM_VERSION:-0.16.2}"

# Map OS/arch to Gum release names
case "$(uname -s)" in
  Linux)  OS=Linux ;;
  Darwin) OS=Darwin ;;
  *) echo "Unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

case "$(uname -m)" in
  x86_64|amd64) ARCH=x86_64 ;;
  arm64|aarch64) ARCH=arm64 ;;
  *) echo "Unsupported arch: $(uname -m)" >&2; exit 1 ;;
esac

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$ROOT/.vendor/gum/bin"
mkdir -p "$DEST"

BASE="https://github.com/charmbracelet/gum/releases/download/v${GUM_VERSION}"
TARBALL="gum_${GUM_VERSION}_${OS}_${ARCH}.tar.gz"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
curl -fsSL -o "$TMP/$TARBALL" "$BASE/$TARBALL"

# (Optional) verify checksums
if command -v sha256sum >/dev/null 2>&1; then
  curl -fsSL -o "$TMP/checksums.txt" "$BASE/checksums.txt"
  (cd "$TMP" && sha256sum -c --ignore-missing checksums.txt | grep "$TARBALL" || true)
elif command -v shasum >/dev/null 2>&1; then
  curl -fsSL -o "$TMP/checksums.txt" "$BASE/checksums.txt"
  (cd "$TMP" && grep "$TARBALL" checksums.txt | shasum -a 256 -c - || true)
fi
# (You can also verify with cosign if you want stronger guarantees.) :contentReference[oaicite:1]{index=1}

# Extract
tar -xzf "$TMP/$TARBALL" -C "$TMP"

# Locate the 'gum' binary no matter where it is
gum_candidate="$(find "$TMP" -type f -name gum -print -quit)"
if [ -z "$gum_candidate" ]; then
  echo "Could not find 'gum' in $TARBALL. Contents were:"
  tar -tzf "$TMP/$TARBALL"
  exit 1
fi

# Install (no sudo needed)
install -m 0755 "$gum_candidate" "$DEST/gum"

echo "gum installed to $DEST"
