from xmlrpc.client import boolean
import numpy as np
from ..function.mellowmax import mellowmax
from ..function.boltzmax import boltzmax


class GVI:
    """
    Generalized Value Iteration (GVI)
    """

    def __init__(self,
                 env=None,
                 softmax=None,
                 delta=1e-3,
                 max_iter=1000,
                 gamma=0.98) -> None:
        """ initialize GVI algorithm
        Arguments:
            env: gym environment
            softmax: softmax function
            delta: termination condition, default: 1e-3
            max_iter: maximum iteration before termination, default: 1000
            gamma: discount factor, default: 0.98
        """
        self.env = env
        self.softmax = softmax
        self.delta = delta
        self.max_iter = max_iter
        self.gamma = gamma

    def start(self) -> boolean:
        """ start value iteration
        Returns:
            bool: terminated before reaching max_iteration
        """
        # TODO: raise error if env == None
        self.get_rewards_and_transitions_from_env(self.env)
        self.num_iter, done = self.policy_iteration()
        return done

    def get_rewards_and_transitions_from_env(self, env):
        # Get state and action space sizes
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n

        # Intiailize matrices
        self.R = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))

        # Get rewards and transition probabilitites for all transitions from an OpenAI gym environment
        for s in range(self.num_states):
            for a in range(self.num_actions):
                for transition in env.P[s][a]:
                    prob, s_, r, done = transition
                    self.R[s, a, s_] = r
                    self.P[s, a, s_] = prob

    def policy_iteration(self):
        # initialize action value function Q
        Q = np.zeros((self.num_states, self.num_actions))
        num_iteration = 0
        done = 0
        while True:
            num_iteration += 1
            diff = 0
            Q_ = Q
            # update value function
            Q = np.array([[
                np.sum([
                    self.R[s, a, s_] +
                    self.gamma * self.P[s, a, s_] * self.softmax(Q[s_])
                    for s_ in range(self.num_states)
                ])
                for a in range(self.num_actions)
            ]
                          for s in range(self.num_states)])
            # maximum difference
            diff = np.max([
                abs(Q[s, a] - Q_[s, a])
                for a in range(self.num_actions)
                for s in range(self.num_states)
            ])
            # termination conditions
            if self.delta and diff < self.delta:
                done = 1
                break
            if self.max_iter and num_iteration >= self.max_iter:
                break
        return num_iteration, done

    def set_env(self, env) -> None:
        self.env = env

    def set_softmax(self, softmax) -> None:
        self.softmax = softmax

    def set_delta(self, delta) -> None:
        self.delta = delta

    def set_max_iter(self, max_iter) -> None:
        self.max_iter = max_iter

    def set_gamma(self, gamma) -> None:
        self.gamma = gamma

    def get_reult(self) -> int:
        """
        return:
            int: number of iteration executed before termination
        """
        return self.num_iter

    def reset(self) -> None:
        self.env = None
        self.softmax = None
        self.delta = 1e-3
        self.max_iter = 1000
        self.num_iter = None