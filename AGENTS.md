# AGENTS.md

Only do what I explicitly ask. Do not take unprompted actions: no commits/pushes/amends, no config changes, no dependency installs, no starting or killing background processes, no file edits beyond the literal request, no "cleanup" or "fixes along the way". If a request could imply extra steps, do the minimum and ask before anything beyond it. Never treat a general "go ahead" as permission for work I didn't name.

If you think of anything beyond the request, just give a brief suggestion instead of doing it.

## "Send everything to the cloud"

Training runs on Colab via Google Drive. Push code + checkpoints with rclone:

1. From the repo root: `./scripts/sync_code.sh` (pushes code to `drive:Tidy/code`; excludes `.venv`, `models`, `.git`, `__pycache__`).
2. `rclone copy models/ drive:Tidy/models` (all checkpoints).
3. Optionally verify: `rclone lsd drive:Tidy/` → should list `code` and `models`.

The shared-client_id `NOTICE` is harmless noise. Note: rclone's shared Google Drive client_id is being retired during 2026; set a custom one via `rclone config` before it breaks.
