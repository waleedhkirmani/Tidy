import os
import torch

MODELS_DIR = os.environ.get("TIDY_MODELS_DIR", "models")
DEFAULT_PATH = os.path.join(MODELS_DIR, "b_model_latest.pt")


def save_checkpoint(agent, episode, path=DEFAULT_PATH, buffer=None):
    checkpoint = {
        "episode": episode,
        "actor": agent.actor.state_dict(),
        "critic1": agent.critic1.state_dict(),
        "critic2": agent.critic2.state_dict(),
        "target_critic1": agent.target_critic1.state_dict(),
        "target_critic2": agent.target_critic2.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        # Recorded for reproducibility, but deliberately NOT restored on load:
        # alpha is a hyperparameter now, and old checkpoints carry the collapsed
        # auto-tuned value (~0.002) that this change exists to escape. The
        # constructor value / train.py --alpha wins.
        "alpha": agent.alpha,
        # Replay buffer rides inside b_model_latest.pt (the resume point) so a
        # fresh run -- deleting that file -- flushes model AND buffer in one step.
        "buffer": buffer.save_state() if buffer is not None else None,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(agent, path=DEFAULT_PATH, restore_buffer=None):
    if not os.path.exists(path):
        return 0

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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

    # No alpha restore: see the note in save_checkpoint. Old checkpoints' stale
    # log_alpha key is intentionally ignored rather than exp()'d back into alpha.
    ck_buffer = checkpoint.get("buffer")
    if restore_buffer is not None and ck_buffer is not None:
        restore_buffer.restore_state(ck_buffer)
    return checkpoint["episode"] + 1
