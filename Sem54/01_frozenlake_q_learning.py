"""
Tabular Q-learning - modification of the value iteration method:
1. Start with an empty table, mapping states to values of actions.
2. By interacting with the environment, sample the tuple (s, a, r, s').
3. Update the Q(s, a) value using the Bellman approximation:
Q(s, a) ← r + γ max Q(s′ , a′)
4. Repeat from step 2.
"""


import collections
import gymnasium as gym
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
ALPHA = 0.9
TEST_EPISODES = 20
EXPERIMENT_NAME = "-q-learning-determined"

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state, _ = self.env.reset()
        self.Q_values = collections.defaultdict(float)

    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        if terminated or truncated:
            self.state, _ = self.env.reset()
        else:
            self.state = new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        """ Select the action with the largest value from the Q-table. """
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.Q_values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, next_s):
        """ Update Q-values table using one step from the environment.
        Averaging between old and new values of Q using learning rate αlpha. """
        best_v, _ = self.best_value_and_action(next_s)
        new_v = r + GAMMA * best_v
        old_v = self.Q_values[(s, a)]
        self.Q_values[(s, a)] = old_v * (1 - ALPHA) + new_v * ALPHA

    def play_episode(self, env):
        """ Evaluate current policy to check the progress of learning.
        TODO: random action if Q_value == 0 """
        total_reward = 0.0
        state, _ = env.reset()
        while True:
            q_value, action = self.best_value_and_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
            state = new_state
        return total_reward


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment=EXPERIMENT_NAME)

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)

        mean_reward = 0.0
        for _ in range(TEST_EPISODES):
            mean_reward += agent.play_episode(test_env)
        mean_reward /= TEST_EPISODES
        writer.add_scalar("reward", mean_reward, iter_no)
        if mean_reward > best_reward:
            print(f"Best reward updated {best_reward:.3f} -> {mean_reward:.3f}")
            best_reward = mean_reward
        if mean_reward > 0.80:
            print(f"Solved in {iter_no:d} iterations!")
            break
    writer.close()
