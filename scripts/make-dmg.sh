#!/usr/bin/env bash
# Bundles the release binary into ipic.app and packs a macOS DMG.
# Usage: scripts/make-dmg.sh [path-to-release-binary]
#   (default: target/release/ipic; CI passes the --target variant path)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BINARY="${1:-target/release/ipic}"
ARCH="$(lipo -archs "$BINARY" | tr ' ' '-')"
# Tag is the source of truth in CI (GITHUB_REF_NAME=v0.1.1 -> 0.1.1);
# local builds fall back to the workspace version.
if [[ -n "${GITHUB_REF_NAME:-}" && "${GITHUB_REF_NAME}" == v* ]]; then
  VERSION="${GITHUB_REF_NAME#v}"
else
  VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -1)"
fi
DMG_NAME="ipic_${VERSION}_${ARCH}.dmg"

if [[ ! -x "$BINARY" ]]; then
  echo "release binary not found at $BINARY — run: cargo build --release -p ipic-app" >&2
  exit 1
fi

rm -rf dist
APP="dist/ipic.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BINARY" "$APP/Contents/MacOS/ipic"
cp crates/ipic-app/assets/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"
sed "s/@VERSION@/$VERSION/g" crates/ipic-app/assets/Info.plist > "$APP/Contents/Info.plist"

# Ad-hoc signature: arm64 binaries must carry one, and a stable post-copy
# signature keeps Gatekeeper's prompt deterministic. (No Developer ID:
# first launch needs right-click > Open.)
codesign --force --sign - "$APP" >/dev/null 2>&1

# DMG staging: the app plus a /Applications symlink for drag-install.
STAGE="dist/dmg-stage"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "ipic" -srcfolder "$STAGE" -ov -format UDZO "dist/$DMG_NAME" -quiet
rm -rf "$STAGE"

echo "dist/$DMG_NAME"
