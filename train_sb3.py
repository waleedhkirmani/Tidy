import argparse
import time
from collections import deque
from statistics import mean

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from envs.tidy_env import TidyEnv

KEYS = ["reach", "grasp", "lift", "hold", "approach", "lower", "success"]


class Breakdown(BaseCallback):
    """Prints the eval.py-style per-component breakdown every 10 episodes."""

    def __init__(self, window=10):
        super().__init__()
        self.window = window
        self.reset_run()

    def reset_run(self):
        self.returns = deque(maxlen=self.window)
        self.succ_recent = deque(maxlen=self.window)
        self.comps = dict.fromkeys(KEYS, 0.0)
        self.n = 0

    def _on_step(self):
        comps = self.locals["infos"][0]["rewards"]
        for k in KEYS:
            self.comps[k] += comps[k]
        if "episode" in self.locals["infos"][0]:
            self.n += 1
            self.returns.append(self.locals["infos"][0]["episode"]["r"])
            self.succ_recent.append(comps["success"] > 0)
            if self.n % self.window == 0:
                print(
                    f"Episode {self.n:4d} | Return({self.window}): {mean(self.returns):8.2f}"
                    f" | succ: {sum(self.succ_recent)}/{len(self.succ_recent)}"
                    f" | steps: {self.num_timesteps}"
                    + "".join(f" | {k}={self.comps[k]:.2f}" for k in KEYS),
                    flush=True,
                )
                self.comps = dict.fromkeys(KEYS, 0.0)
        return True


class TimeBox(BaseCallback):
    """Stops the rollout loop cleanly after `minutes` of wall clock."""

    def __init__(self, minutes):
        super().__init__()
        self.deadline = time.time() + minutes * 60

    def _on_step(self):
        return time.time() < self.deadline


parser = argparse.ArgumentParser()
parser.add_argument("--minutes", type=float, default=10, help="Wall-clock budget")
parser.add_argument(
    "--curriculum",
    type=float,
    default=0.8,
    help="Fraction of episodes that start hovering above the cube (grasp probe). "
    "0 = same config as train.py.",
)
parser.add_argument(
    "--probe",
    choices=["grasp", "place"],
    default="grasp",
    help="Curriculum probe type: grasp (start above the cube) or place (start "
    "above the tray, cube in hand; fires at HALF the grasp probability)",
)
parser.add_argument("--max-steps", type=int, default=400, help="Episode truncation")
parser.add_argument(
    "--out", default="models/sb3_sanity.zip", help="Where to save the trained model"
)
args = parser.parse_args()

# tiny nets: 4 intra-op threads cost more in launch overhead than they save
torch.set_num_threads(1)

env = TidyEnv(
    gui=False,
    randomize_cube_spawn=False,
    curriculum_prob=args.curriculum,
    probe=args.probe,
)
env.max_episode_steps = args.max_steps

# textbook SAC, so a failure here indicts the env/reward, not the hyperparameters
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef="auto",
    target_entropy="auto",
    device="cpu",
    seed=0,
    verbose=1,
)

print(
    f"SB3 SAC | curriculum={args.curriculum} | max_steps={args.max_steps}"
    f" | obs {env.observation_space.shape} | act {env.action_space.shape}",
    flush=True,
)
bd = Breakdown()
model.learn(
    total_timesteps=10_000_000,  # bounded by TimeBox instead
    callback=[bd, TimeBox(args.minutes)],
    log_interval=10,
)
model.save(args.out)

succ = bd.succ_recent
print(
    f"\nDone: {bd.n} episodes, success rate"
    f" {100 * mean(succ):.0f}% (last {len(succ)}) -> {args.out}"
)
