import json

import gym
import numpy as np
from torch import softmax
from mellowmax_softmax.function import Boltzmax, Mellowmax
from mellowmax_softmax.algo import SARSA
from tqdm import tqdm, trange

RUNS = 25
STEP = 300000


def main() -> np.floating:
    """experiment in `taxi-v2`

    Returns:
        mean_reward(np.floating): mean reward of all runs
    """
    returns = []

    for _ in trange(RUNS):
        remain_steps = STEP
        env = gym.make('taxi-v2')

        with tqdm(total=STEP) as step_pbar:
            while remain_steps > 0:

                # TODO: add softmax policy
                agent = SARSA(env=env, softmax=None, max_iter=remain_steps)

                steps, reward, _ = agent.start()
                returns.append(reward)

                remain_steps -= steps
                step_pbar.update(steps)

    return np.average(returns)
