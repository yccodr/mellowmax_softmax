from typing import Optional, Union

import gym
import numpy as np
from gym import logger, spaces


def _random_matrix(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    # Initialize the transition probabilities by sampling uniformly from [0, .01].
    matrix = rng.uniform(0, 0.01, shape)
    # Added to each entry, with probability 0.5, Gaussian noise with mean 1 and variance 0.1.
    matrix += rng.normal(1, np.sqrt(0.1), shape) * rng.binomial(1, 0.5, shape)
    # Added to each entry, with probability 0.1, Gaussian noise with mean 100 and variance 1.
    matrix += rng.normal(100, np.sqrt(1), shape) * rng.binomial(1, 0.1, shape)
    return matrix


class RandomMDP(gym.Env):
    """
    ### Description

    This environment is a implementation of the random MDP described by Kavosh Asadi and Michael L. Littman in
    ["An Alternative Softmax Operator for Reinforcement Learning"](https://arxiv.org/pdf/1612.05628.pdf)

    ### Action Space

    The action is a `ndarray` with shape `(1,)`, |A| is unifomly sampled from {2, 3, 4, 5} which can
    take values `{0, 1, ..., |A| - 1}`.

    ### Observation Space

    The observation is a `ndarray` with shape `(1,)`, |S| is unifomly sampled from {2, 3, ..., 10}
    which can take values `{0, 1, ..., |S| - 1}`.

    ### Rewards

    Randomly generate rewards for each environment instance. The max reward is 0.5.

    ### Starting State

    Each episode will starts from the state with index 0.

    ### Episode Termination

    The episode terminates if it reachs the state with index |S| - 1 or reaches 1000 iteratinos.

    """

    def __init__(self):
        action_space_size = self.np_random.integers(2, 5, endpoint=True)
        state_space_size = self.np_random.integers(2, 10, endpoint=True)

        self.action_space = spaces.Discrete(action_space_size)
        self.observation_space = spaces.Discrete(state_space_size)

        self.state = None

        # S x A x S' -> R
        self.transition = _random_matrix(
            (state_space_size, action_space_size, state_space_size),
            self.np_random)

        self.transition = (self.transition - np.min(self.transition)) / (
            np.max(self.transition) - np.min(self.transition))
        self.transition /= np.sum(self.transition, axis=-1, keepdims=True)

        # S x A -> R
        self.reward = _random_matrix((state_space_size, action_space_size),
                                     self.np_random)

        # Limit the max reward to be 0.5.
        self.reward /= np.max(self.reward)
        self.reward *= 0.5

        self.iterations = 0
        self.steps_beyond_done = None

    def step(self, action: Union[int, np.ndarray]):
        assert self.action_space.contains(action)
        assert self.state is not None, 'Call reset before using step method.'

        self.iterations += 1

        reward = self.reward[self.state, action]

        self.state = self.np_random.choice(self.observation_space.n,
                                           p=self.transition[self.state,
                                                             action, :])

        done = self.state == 1 or self.iterations > 1000

        if done:
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                if self.steps_beyond_done == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned done = True. You "
                        "should always call 'reset()' once you receive 'done = "
                        "True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1
                reward = 0.0

        return np.array(self.state, dtype=np.int32), reward, done, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = 0
        self.iterations = 1
        self.steps_beyond_done = None

        if not return_info:
            return np.array(self.state, dtype=np.int32)

        return np.array(self.state, dtype=np.int32), {}

    def render(self, mode: str = 'human'):
        logger.warn("This environment doesn't support rendering.")
