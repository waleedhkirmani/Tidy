import os
import torch

MODELS_DIR = os.environ.get("TIDY_MODELS_DIR", "models")
DEFAULT_PATH = os.path.join(MODELS_DIR, "b_model_latest.pt")


def save_checkpoint(agent, episode, path=DEFAULT_PATH):
    checkpoint = {
        "episode": episode,
        "actor": agent.actor.state_dict(),
        "critic1": agent.critic1.state_dict(),
        "critic2": agent.critic2.state_dict(),
        "target_critic1": agent.target_critic1.state_dict(),
        "target_critic2": agent.target_critic2.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "log_alpha": agent.log_alpha.detach(),
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(agent, path=DEFAULT_PATH):
    if not os.path.exists(path):
        return 0

    checkpoint = torch.load(path, map_location="cpu")
    try:
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic1.load_state_dict(checkpoint["critic1"])
        agent.critic2.load_state_dict(checkpoint["critic2"])

        agent.target_critic1.load_state_dict(checkpoint["target_critic1"])
        agent.target_critic2.load_state_dict(checkpoint["target_critic2"])

        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    except (RuntimeError, KeyError) as e:
        # ponytail: architecture change (obs 9->18) invalidates old checkpoints;
        # don't crash the resume loop, start fresh instead.
        print(
            f"WARNING: checkpoint {path} incompatible with current architecture "
            f"({e}); starting fresh"
        )
        return 0

    # Checkpoints saved before the temperature was learnable have no log_alpha.
    # Presence-guarded rather than checkpoint["log_alpha"]: a missing key inside
    # the block above would return 0 and silently restart training from scratch
    # with a trained policy discarded. Here it just keeps the constructor alpha.
    if "log_alpha" in checkpoint:
        agent.log_alpha.data.copy_(checkpoint["log_alpha"])

    return checkpoint["episode"] + 1
