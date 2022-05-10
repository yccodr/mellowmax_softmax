import json

import gym
import numpy as np
from mellowmax_softmax.algo import SARSA
from mellowmax_softmax.function import Boltzmax, Mellowmax
from mellowmax_softmax.function.boltzmax import BoltzmannPolicy
from mellowmax_softmax.function.mellowmax import MellowmaxPolicy
from mellowmax_softmax.utils import sigterm_decorator
from torch import softmax
from tqdm import tqdm, trange

RUNS = 5
STEP = 300000


@sigterm_decorator()
def main() -> np.floating:
    """experiment in `taxi-v2`

    Returns:
        mean_reward(np.floating): mean reward of all runs
    """
    returns = []

    for _ in trange(RUNS):
        remain_steps = STEP
        env = gym.make('Taxi-v3')

        with tqdm(total=STEP) as step_pbar:
            while remain_steps > 0:

                agent = SARSA(
                    env=env,
                    #   policy=BoltzmannPolicy(16.55),
                    policy=MellowmaxPolicy(16.55),
                    max_iter=remain_steps)

                steps, reward, _ = agent.start()
                returns.append(reward)

                remain_steps -= steps
                step_pbar.update(steps)

    return np.average(returns)


if __name__ == '__main__':
    print(main())
