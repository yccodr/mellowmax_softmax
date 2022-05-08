import gym

from mellowmax_softmax.envs.random_mdp import RandomMDP
from mellowmax_softmax.envs.simple_mdp import SimpleMDP


def test_envs_registered():
    assert isinstance(gym.make('SimpleMDP-v0').unwrapped, SimpleMDP)
    assert isinstance(gym.make('RandomMDP-v0').unwrapped, RandomMDP)
