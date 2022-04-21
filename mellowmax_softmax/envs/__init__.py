import gym

from .custom_mdp import CustomMDP as _CustomMDP
from .random_mdp import RandomMDP as _RandomMDP


def get_env(name: str) -> gym.Env:
    """
    Get the environment by name.
    Avaliable environments:
        - `Taxi` : Taxi-v3
        - `LunarLander` : LunarLander-v2
        - `RandomMDP` : RandomMDP
        - `CustomMDP` (default) : CustomMDP
    """

    if name == 'Taxi':
        env = gym.make('Taxi-v3')
    elif name == 'LunarLander':
        env = gym.make('LunarLander-v2')
    elif name == 'RandomMDP':
        env = _RandomMDP()
    else:
        env = _CustomMDP()

    return env
