import os
import torch

DEFAULT_PATH = os.path.join("models", "b_model_latest.pt")


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
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(agent, path=DEFAULT_PATH):
    if not os.path.exists(path):
        return 0

    checkpoint = torch.load(path, map_location="cpu")

    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic1.load_state_dict(checkpoint["critic1"])
    agent.critic2.load_state_dict(checkpoint["critic2"])

    agent.target_critic1.load_state_dict(checkpoint["target_critic1"])
    agent.target_critic2.load_state_dict(checkpoint["target_critic2"])

    agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    return checkpoint["episode"] + 1
