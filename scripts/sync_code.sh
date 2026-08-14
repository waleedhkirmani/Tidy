#!/usr/bin/env bash
# Push local code to Google Drive so the Colab notebook can run it.
# No git commit needed. Run `rclone config` once to auth Drive.
set -euo pipefail

rclone copy . drive:Tidy/code \
  --exclude ".venv/**" \
  --exclude "models/**" \
  --exclude ".git/**" \
  --exclude "__pycache__/**"
echo "Code synced to drive:Tidy/code"
