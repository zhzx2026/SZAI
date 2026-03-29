#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

APP_NAME="SZ-AI-Mac"
DIST_DIR="$ROOT_DIR/dist"
BUILD_DIR="$ROOT_DIR/build/pyinstaller"
APP_PATH="$DIST_DIR/$APP_NAME.app"
ZIP_PATH="$DIST_DIR/$APP_NAME.zip"

python3 -m pip install -r requirements-macos-app.txt
rm -rf "$APP_PATH" "$ZIP_PATH" "$BUILD_DIR"

pyinstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name "$APP_NAME" \
  --distpath "$DIST_DIR" \
  --workpath "$BUILD_DIR" \
  --specpath "$BUILD_DIR" \
  --paths "$ROOT_DIR/src" \
  --hidden-import "sz_ai.data" \
  --hidden-import "sz_ai.model" \
  --collect-all "torch" \
  --collect-all "numpy" \
  --osx-bundle-identifier "com.zhzx.szai" \
  "$ROOT_DIR/app/sz_ai_mac_app.py"

ditto -c -k --sequesterRsrc --keepParent "$APP_PATH" "$ZIP_PATH"

echo "Built app: $APP_PATH"
echo "Built zip: $ZIP_PATH"
