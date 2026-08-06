#!/usr/bin/env bash
# Pull checkpoints from Google Drive (written by the Colab training session)
# into ./models for local eval. Run `rclone config` once to auth Drive.
set -euo pipefail

rclone sync drive:Tidy/models models/
echo "Checkpoints synced to models/"
