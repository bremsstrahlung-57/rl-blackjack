from collections import defaultdict
from typing import cast

import gymnasium as gym
import numpy as np

from rl_env import BlackJackRLEnviroment

env = BlackJackRLEnviroment()
action_space = cast(gym.spaces.Discrete, env.action_space)

Q = defaultdict(lambda: np.zeros(action_space.n))


def epsilon_greedy(state, epsilon):
    if np.random.random() < epsilon:
        return action_space.sample()
    return int(np.argmax(Q[state]))


def train(
    num_episodes=500_000,
    alpha=0.05,
    gamma=0.1,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.99999,
):
    epsilon = epsilon_start

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = epsilon_greedy(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next = 0.0 if terminated else np.max(Q[next_state])
            td_target = float(reward) + gamma * best_next
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q
