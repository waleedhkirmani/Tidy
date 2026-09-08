# Tidy

A robotic arm that learns to pick up a cube and put it in a tray.

Tidy is a from-scratch reinforcement learning project: a Franka Panda arm in PyBullet, trained with a hand-written SAC implementation (no RL libraries), to grasp a cube and place it into a tray. It's the first of a planned pair of manipulation projects, built as a portfolio piece for Embodied AI / Robotics.

The long-term design philosophy is hybrid, not end-to-end RL: RL handles low-level motor control (reaching, grasping, placing), while later phases hand sequencing and object-to-container assignment to a classical planner. Tidy is currently at the Phase 1 → 2 boundary — single cube, single tray, fixed-spawn task solved at 100%, spawn randomization now being trained on.

## Demo

https://github.com/user-attachments/assets/0c78e6d9-59c4-4c92-b109-71acce245f8d

## Why this project exists

This isn't a tutorial clone. Every piece — the environment, the reward function, and the SAC implementation — was built and debugged from scratch, including from-scratch PyTorch implementations of PPO, DQN, and SAC prior to this project. The goal was to actually understand the failure modes of RL-for-manipulation, not just get a policy that works. Most of what's interesting about this repo lives in how many ways the reward function got exploited before it didn't — see [Development Story](#development-story) below.

Trained entirely on CPU (no local GPU), with cloud offload to Colab for the actual training runs.

## Results

- **100% success rate on fixed spawns** as of episode 4798 (`b_best_4798_fixed_spawn_100pct_working.pt`) — cube and tray at fixed positions, arm reliably reaches, grasps, lifts, carries, and lowers the cube into the tray. The model is present in releases.
- **Random-spawn generalization is the active frontier** — spawn randomization within a configurable spread was just added; this is the current training target.

## How it works

**Task loop:** reach the cube → grasp it → lift it → carry it above the tray → descend and release.

**Observation (9-D):** end-effector position (3) + cube position (3) + tray center (3). Pose-only — no vision.

**Action (4-D):** Δxyz step (±0.05 per axis) + binary gripper command. The policy outputs end-effector deltas; PyBullet's IK solver converts these into joint commands, so the network never has to learn joint-space control directly.

**Episode:** 400 steps max, each step is a short burst of 60 physics substeps toward the commanded target (not a wait-for-convergence step) — so the policy is effectively giving directions at a coarser rate than the 240 Hz physics runs at.

**Grasp detection** requires simultaneous contact on both fingers *and* the gripper commanded closed *and* the contact normals actually opposing each other (i.e. a real clamp) — early versions fired on any finger-cube touch, which turned out to be a major source of false "success."

## Development story

The project log (`projectLog.txt`) is a warts-and-all record of the debugging process. A condensed version:

1. **Getting the arm to move at all.** Started with raw joint control and PyBullet's IK, moving the arm to single hard-coded positions, then chaining several. Large joint jumps made the arm collide with itself; fixing this meant discovering joint limits and adding a slider for manual control before trusting the arm to move on its own.
2. **Scripted pick-and-place.** Got a working (but unreliable) script to pick up a cube and drop it in a tray. The core bug: once the arm computed IK for a target, it would commit to that joint trajectory even if the target had since become stale — occasionally driving straight through/into the cube. Fixed by routing through an intermediate waypoint and recomputing IK from there, which took the script from flaky to 100% reliable.
3. **Refactor into classes**, then into a Gymnasium-style custom environment (duck-typed `reset()`/`step()`, not a formal `gymnasium.Env` subclass) with action/observation spaces and a (still slightly rough) termination condition.
4. **Reward design**, broken into subtasks — reach, grasp, lift, approach tray, lower, place — each with its own reward term. This is where most of the real debugging happened:
   - **Reward hacking, repeatedly.** The policy learned to hover just above the cube to farm a proximity reward instead of grasping; learned to sit on the floor once floor contact was accidentally net-positive; learned to stay at max height once "lift" and "lower" reward terms were simultaneously active and cancelling each other out; learned to oscillate near the tray rim to farm a centering reward. Each of these was diagnosed by inspecting per-component reward logs, not just total reward.
   - **A gripping-detection bug** (any finger-cube contact counted as a grasp) inflated apparent progress for a long stretch of training before being caught and replaced with the opposed-contact-normal check described above.
   - **An environment bug, not an RL bug** — the tray's center-point calculator had an offset error, so the arm was correctly learning to place the cube beside the tray. "The RL was perfect here, the environment design got me."
   - **Fixes that stuck:** a monotone-min descent tracker (reward only for *new* lowest height, so the policy can't farm reward by bobbing up and down), one-shot bonuses instead of repeatable ones for state transitions (entering the grasp window, executing a valid close), gating gripper-close commands so the arm won't move and close at the same time (which mechanically prevents a clean clamp), and halving the "approach tray" reward while the cube is still below rim height so the policy can't get paid for carrying it there in an unsafe orientation.
5. **Training loop tuning** — reward magnitudes were the recurring lever: too small and the policy made no bold moves; too large and it overfit to whichever exploit paid best in a given regime. Gamma and the learning rate for the actor were both adjusted mid-project to stop the policy from treating descent as too risky relative to its reward. The entropy term was pulled out of the critic loss once it was traced to the arm being paid a constant bonus for existing near the cube regardless of actual progress.

That loop — hypothesize exploit, inspect per-component logs, patch the specific reward term or detection bug, retrain — is the throughline of the whole project and the main thing this repo demonstrates.

## Repo layout

```
envs/            tidy_env.py (core environment, ~500 lines), robot.py, small_cube.py, tray.py
rl/              sac.py, networks.py, replay_buffer.py, checkpoint.py
train.py         SAC training loop (resumable, time-boxed for Colab sessions)
eval.py          GUI evaluation with camera controls
runner.py        random-action sanity baseline
working.py       scripted (non-learned) pick-and-place demo
scripts/         move_robot.py, test_motion.py, test_pybullet.py, sync_code.sh, pull_models.sh
colab_train.ipynb  cloud training notebook
models/          SAC checkpoints
typings/pybullet/  hand-written type stubs (pybullet ships without types)
```

## Environment details

- **Physics:** single PyBullet connection enforced per process (multiple connections corrupt PyBullet's internal state). Gravity -9.81.
- **Robot:** Franka Panda, fixed base, self-collision enabled. IK targets the finger link, orientation locked pointing straight down. Home pose is found by iterating IK 12× to escape local minima, with the forearm roll joint seeded mid-range to avoid it jamming against its limits. Joint forces are set high (200 N·m, 1000 N·m on the shoulder joint) with a low position gain (0.3) and capped velocity (4.0) for controlled, non-violent motion — early versions with weaker gripper force and uncapped motion sent the arm "flying" when it tried to move a grasped cube.
- **Spawning:** cube and tray have default fixed positions; `randomize_cube_spawn=True` samples the cube within a configurable spread, rejecting spawns that overlap the tray footprint.
- **Success condition:** cube's XY position within the tray's half-extents, cube height under the tray lip plus a small margin, and the fingers below the tray rim by 2cm — worth +50 reward and ends the episode.

## Reward function

Per-step reward is the sum of the components below, minus a small time penalty and a penalty for the end-effector getting too close to the floor:

| Component | What it rewards |
|---|---|
| **reach** | Progress toward the cube (distance-based gradient), plus a one-time bonus for entering the grasp window, plus a per-step bonus for staying in the graspable band |
| **grasp** | One-time bonus for a genuine clamp (opposed contact normals), smaller one-time bonus for executing a valid close inside the grasp window |
| **lift** | Height gained while gripping, capped once the cube reaches a safe carry height |
| **hold** | Small per-step bonus for maintaining grip, so holding is never net-negative |
| **approach** | Progress toward the tray while carrying the cube, paid at half rate while still below safe carry height |
| **lower** | Centering over the tray plus a monotone-min descent bonus (only pays for new lowest height, to prevent oscillation farming) |
| **success** | Large one-time bonus, ends episode |

The anti-hacking measures (monotone-min descent, one-shot bonuses, gated gripper-close, halved low-carry reward) exist specifically because earlier, simpler versions of this reward function were reliably exploited — see [Development Story](#development-story).

## RL implementation

Soft Actor-Critic, implemented from scratch in PyTorch (no RL library):

- **Actor:** MLP, 9 → 64 → 64 → 8, tanh activations, outputs a mean and log-std per action dimension (log-std clamped to [-20, 2]), tanh-squashed Gaussian sampling with the corresponding log-probability correction. The first 3 output dimensions are scaled to the ±0.05 position-delta range; the 4th is mapped to the gripper's [0, 1] range.
- **Critic:** twin critics, MLP 13 → 64 → 64 → 1 (state + action concatenated), tanh activations, with target networks updated via soft (Polyak) updates (τ = 0.005).
- **Hyperparameters:** γ = 0.99, α = 0.2 (fixed, not learned), learning rate 3e-4 (Adam) for both actor and critics.
- **Replay buffer:** fixed-size deque (10,000 transitions), uniform sampling, batch size 64.
- **Checkpointing:** full training state (both critic pairs, both optimizers, episode count) saved every 100 episodes and at exit, plus a rolling "best" checkpoint by 10-episode average return. A `TIDY_MODELS_DIR` env var redirects checkpoint output, used for the Colab workflow below.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch pybullet gymnasium numpy
```

Requires `torch>=2.0`, `pybullet>=3.2.7`, `gymnasium>=1.3.0`, `numpy>=1.24`.

## Usage

**Train locally (time- or episode-boxed):**
```bash
python train.py --episodes 5000
python train.py --minutes 600
```

**Evaluate a trained policy** (10 greedy episodes, prints success rate, average reward, and per-component reward breakdown; interactive camera via arrow keys / WASD / Q / E):
```bash
python eval.py
```

**Sanity-check the environment** with a uniform-random policy:
```bash
python runner.py
```

**Watch the scripted (non-learned) baseline:**
```bash
python working.py
```

**Self-check the robot's motion primitives** (home-pose feasibility, clean descent, scripted grasp success):
```bash
python scripts/test_motion.py
```

### Cloud training (Colab)

Training is physics-bound and CPU-friendly, so it's offloaded to Colab for long runs:

1. Push local code to Drive: `scripts/sync_code.sh`
2. In `colab_train.ipynb`: mount Drive, copy code from `drive:Tidy/code`, install deps, run `python train.py --minutes 600` (CPU runtime recommended)
3. Pull resulting checkpoints back: `scripts/pull_models.sh`

## Current limitations / open work

- Random-spawn generalization is not yet solved — the 100% success number is on **fixed** spawns only.
- Termination condition in the environment is known to be slightly flawed.
- No config files — hyperparameters are set in code.
- No experiment tracking beyond stdout logs, no multi-seed runs.
- PPO baseline (mentioned in the original scoping) not yet implemented.
- Classical planner (Phase 3 — object-to-container assignment/sequencing) not yet started; current scope is single cube, single tray.
- `assets/` and `tools/` directories are scaffolded but empty.

## License

MIT — Waleed Hassan
