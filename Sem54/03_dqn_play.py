#!/usr/bin/env python3
import gymnasium as gym
import time
import argparse
import numpy as np

import torch
from gymnasium.wrappers import RecordVideo

from libr import wrappers
from libr import dqn_model

import collections

DEFAULT_ENV_NAME = "ALE/Pong-v5"
MODEL_PATH = "Pong-v5-best_14.dat"
FPS = 15


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=MODEL_PATH,
                        help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" +
                             DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", default='video', help="Directory for video")
    args = parser.parse_args()

    env = wrappers.make_env(args.env, render_mode='rgb_array')
    env = RecordVideo(env, 'video')
    env.metadata['render_fps'] = FPS

    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n)
    model_state = torch.load(args.model, map_location=lambda stg, _: stg)
    net.load_state_dict(model_state)

    state, _ = env.reset()
    env.start_video_recorder()

    total_reward = 0.0
    c = collections.Counter()

    while True:
        env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)

    # Don't forget to close the video recorder before the env!
    env.close_video_recorder()
    env.close()
