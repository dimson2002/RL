#!/usr/bin/env python3
from gymnasium.wrappers import RecordVideo
"""
The Cross-Entropy Method
1. Play N number of episodes using our current model and environment.
2. Calculate the total reward for every episode and decide on a reward
boundary. Usually, we use some percentile of all rewards, such as 50th
or 70th.
3. Throw away all episodes with a reward below the boundary.
4. Train on the remaining "elite" episodes using observations as the input and
issued actions as the desired output.
5. Repeat from step 1 until we become satisfied with the result.
"""

import gymnasium as gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

EXPERIMENT_NAME = "-cartpole_hidden_state_64(2)"  # or -cartpole_net_architecture_v1
HIDDEN_SIZE = 64
BATCH_SIZE = 16
PERCENTILE = 70

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size / 2),
            nn.ReLU(),
            nn.Linear(hidden_size / 2, n_actions)
            )

    def forward(self, x):
        return self.net(x)


def episodes_generator(env, net, batch_size):
    """Generate batch of episodes.
    Every episode in next batch expects higher reward,
    because Net learning best a|s probability distribution.
    """
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, info = env.reset()
    while True:
        # Collect EpisodeSteps into Episode
        obs_v = torch.FloatTensor(np.array(obs)).unsqueeze(0)
        act_probs_v = net(obs_v).softmax(dim=1)
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _, _ = env.step(action)
        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done:
            # Append Episode to batch and clear env state
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    """Filter episodes in the batch.
    Ignore episodes with reward < percentile.
    Returns best state observations with corresponding actions. """
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(np.array(train_obs))
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    env = RecordVideo(env, 'video')
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment=EXPERIMENT_NAME)
    best_episodes = episodes_generator(env, net, BATCH_SIZE)
    env.start_video_recorder()

    for iter_no, batch in enumerate(best_episodes):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = loss(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break
    env.close_video_recorder()
    writer.close()


