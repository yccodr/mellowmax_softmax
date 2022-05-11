import gym
import pytest

from mellowmax_softmax import algo


class FakeEnv(gym.Env):

    def step(self, action):
        pass

    def reset(self, *, seed=None, return_info=False, options=None):
        pass

    def render(self, mode="human"):
        pass


def test_gvi_init_and_setter():
    env = FakeEnv()

    with pytest.raises(TypeError):
        algo.GVI(env, None)
    with pytest.raises(ValueError):
        algo.GVI(env, lambda x: x, delta=-1)
    with pytest.raises(ValueError):
        algo.GVI(env, lambda x: x, max_iter=-1)

    gvi = algo.GVI(env, lambda x: x)

    with pytest.raises(TypeError):
        gvi.set_softmax(None)
    with pytest.raises(ValueError):
        gvi.set_delta(-1)
    with pytest.raises(ValueError):
        gvi.set_max_iter(-1)
