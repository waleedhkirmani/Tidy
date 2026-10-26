# %%

import argparse
import os
import sys
from statistics import mean
import numpy as np
import pybullet as p
import torch
from rl.checkpoint import DEFAULT_PATH, MODELS_DIR, load_checkpoint
from rl.sac import SAC
from stable_baselines3 import SAC as SB3SAC
from envs.tidy_env import TidyEnv, ramp_curriculum_prob, ramp_spread

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["latest", "best"],
    default="latest",
    help="Which checkpoint to load: latest (b_model_latest.pt) or best (b_best.pt)",
)
parser.add_argument(
    "--sb3",
    default=None,
    help="Evaluate a Stable-Baselines3 SAC .zip instead of a hand-rolled checkpoint",
)
parser.add_argument(
    "--curriculum",
    type=float,
    default=None,
    help="Fraction of probe episodes; overrides the value derived from the checkpoint episode",
)
parser.add_argument(
    "--probe",
    choices=["grasp", "place"],
    default="grasp",
    help="Curriculum probe type: grasp (start above the cube) or place (start "
    "above the tray, cube in hand; fires at HALF the grasp probability)",
)
parser.add_argument(
    "--randomize-spawn",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Randomize the cube spawn (default). Use --no-randomize-spawn for a model trained on fixed spawns",
)
args = parser.parse_args()

env = TidyEnv(
    gui=True,
    randomize_cube_spawn=args.randomize_spawn,
    randomize_tray_spawn=args.randomize_spawn,
    curriculum_prob=args.curriculum,
    probe=args.probe,
)

cam = {"dist": 1.7, "yaw": 90.0, "pitch": -35.0, "target": np.array([0.5, 0.0, 0.2])}
p.resetDebugVisualizerCamera(cam["dist"], cam["yaw"], cam["pitch"], cam["target"])


def move_camera(keys):
    if p.B3G_LEFT_ARROW in keys:
        cam["yaw"] += 4
    if p.B3G_RIGHT_ARROW in keys:
        cam["yaw"] -= 4
    if p.B3G_UP_ARROW in keys:
        cam["pitch"] += 4
    if p.B3G_DOWN_ARROW in keys:
        cam["pitch"] -= 4
    if ord("q") in keys:
        cam["dist"] = max(0.2, cam["dist"] - 0.1)
    if ord("e") in keys:
        cam["dist"] += 0.1
    for key, axis, sgn in (
        (ord("w"), 1, 0.05),
        (ord("s"), 1, -0.05),
        (ord("a"), 0, -0.05),
        (ord("d"), 0, 0.05),
    ):
        if key in keys:
            cam["target"][axis] += sgn
    p.resetDebugVisualizerCamera(cam["dist"], cam["yaw"], cam["pitch"], cam["target"])


def poll_keys():
    keys = p.getKeyboardEvents()
    if p.getConnectionInfo()["isConnected"] == 0:
        sys.exit(0)  # window closed -> don't hang
    return keys


# start = load_checkpoint(sac, path="models/b_best_4798_fixed_spawn_100pct_working.pt")
# start = load_checkpoint(sac, path="models/b_best.pt") or load_checkpoint(sac)
if args.sb3:
    model = SB3SAC.load(args.sb3, device="cpu")
    print(
        f"Loaded {args.sb3} (ent_coef {float(model.log_ent_coef.exp().detach()):.4f})"
    )

    def act(state):
        action, _ = model.predict(state, deterministic=True)
        return action

else:
    sac = SAC()
    model_path = os.path.join(MODELS_DIR, "b_best.pt") if args.model == "best" else DEFAULT_PATH
    start = load_checkpoint(sac, path=model_path)
    if start == 0:
        print("WARNING: no checkpoint found, evaluating an untrained policy")
    else:
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        print(
            f"Loaded {os.path.basename(model_path)} (trained through episode {start - 1}"
            + f", alpha={ck.get('alpha', sac.alpha):g})"
        )
        spread = ramp_spread(start - 1)
        env.set_spawn_spread(spread)
        env.curriculum_prob = ramp_curriculum_prob(spread)
        if args.curriculum is not None:
            env.curriculum_prob = args.curriculum
        print(
            f"  ramp spread={spread:.3f} (episode knob), "
            f"curriculum_prob={env.curriculum_prob:.3f}"
            + ("" if args.curriculum is None else "  [--curriculum override]")
        )

    def act(state):
        return sac.select_best_action(state).cpu().numpy()[0]

# %%

KEYS = ["reach", "grasp", "lift", "hold", "approach", "lower", "success"]
n_episodes = 10
rows = []

for _ in range(n_episodes):
    state, info = env.reset()
    done = False
    ep_reward = 0
    comps = dict.fromkeys(KEYS, 0.0)
    terminated = False
    truncated = False
    while not done:
        keys = poll_keys()
        move_camera(keys)
        if keys.get(p.B3G_SPACE, 0) & p.KEY_WAS_TRIGGERED:
            print("(space: ending episode early)")
            terminated = False
            truncated = True
            break
        next_state, reward, terminated, truncated, info = env.step(act(state))
        done = terminated or truncated
        ep_reward += reward
        for k in KEYS:
            comps[k] += info["rewards"][k]
        state = next_state
    rows.append((ep_reward, comps, terminated))
    print(
        f"Episode reward: {ep_reward:8.2f}"
        + "".join(f" {k}={comps[k]:.2f}" for k in KEYS)
    )

rewards, comps, successes = zip(*rows)
print(
    f"\nSuccess rate: {sum(successes)}/{len(rows)} ({100 * sum(successes) / len(rows):.0f}%)"
)
print(f"avg_reward={mean(rewards):.2f}")
for k in KEYS:
    print(f"  avg_{k}={mean(c[k] for c in comps):.2f}")
