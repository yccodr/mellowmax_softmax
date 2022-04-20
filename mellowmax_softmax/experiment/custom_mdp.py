import gym
from mellowmax_softmax.experiment import Experiment


class CustumMDP(Experiment, gym.Env):
    """
    ### Description

    This environment is a implementation of the simple MDP described by Kavosh Asadi and Michael L. Littman in
    ["An Alternative Softmax Operator for Reinforcement Learning"](https://arxiv.org/pdf/1612.05628.pdf)

    ### Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the
    action `a` and `b`.

    | Num | Action |
    |-----|--------|
    | 0   | a      |
    | 1   | b      |

    ### Observation Space

    The observation is a `ndarray` with shape `(1,)` which indicates the current state:

    | Num | Observation |
    |-----|-------------|
    | 0   | S1          |
    | 1   | S2          |

    ### Rewards

    Accoring to the description in the paper, the reward is `0.122` if the action is `a` and `0.033`
    if the action is `b`.

    | state | action | reward |
    |-------|--------|--------|
    | S1    | a      | 0.122  |
    | S1    | b      | 0.033  |

    ### Starting State

    Each episode will starts from S1.

    ### Episode Termination

    The episode terminates if it reachs S2.

    """

    pass
