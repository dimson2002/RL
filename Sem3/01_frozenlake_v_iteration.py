#!/usr/bin/env python3
"""
The value iteration method.

r(s, a) - reward of action from given state,
V(s) - value (or total expected reward) of the state,
Q(s, a) - value of the action, total reward we can get by executing action 'a' in state 's'.

1. Initialize the values of all states, V_i, to some initial value (usually zero).
2. For every state, s, in the MDP, perform the Bellman update V_s.
3. Repeat step 2 for some large number of steps or until changes become too small.
"""

import gymnasium as gym
import collections
from tensorboardX import SummaryWriter

# ENV_NAME = "FrozenLake-v1"
ENV_NAME = "FrozenLake8x8-v1"      # uncomment for larger version
GAMMA = 0.9
TEST_EPISODES = 20
EXPERIMENT_NAME = "-v-iteration"  # or -v-gamma-0.8


class Agent:
    """
    Agent contain functions that using in the training loop and keep tables:
    self.rewards - Reward table: A dictionary with the composite key "source state" + "action" + "target state". \
The value is obtained from the immediate reward.
    self.transits - Transitions table: A dictionary keeping counters of the experienced transitions. \
The key is the composite "state" + "action", and the value is another dictionary (counter) \
that maps the target state into a count of times that we have seen it.
    self.values - Value table: A dictionary that maps a state into the calculated value of this state.
    """
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state, _ = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        """ Gather random experience from the environment and update the reward and transition tables."""
        for i in range(count):
            action = self.env.action_space.sample()
            new_state, reward, terminated, truncated, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            if terminated or truncated:
                self.state, _ = self.env.reset()
            else:
                self.state = new_state

    def calc_action_value(self, state, action) -> float:
        """ Calculates the value of the action from the state using transition, reward, and values tables. """
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            val = reward + GAMMA * self.values[tgt_state]
            action_value += (count / total) * val
        return action_value

    def select_action(self, state) -> int:
        """ Make a decision about the best action to take from the given state."""
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or action_value > best_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        """Take the best action and plays one full episode using the provided environment."""
        total_reward = 0.0
        state, info = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if terminated or truncated:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        """
        1. Loop over all states in the environment.
        2. For every state calculate the values for the states reachable from it,
         obtaining candidates for the value of the state.
        3. Update the value of current state with the maximum value of the action available from the state.
         """
        for state in range(self.env.observation_space.n):
            state_values = [
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment=EXPERIMENT_NAME)

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (
                best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
