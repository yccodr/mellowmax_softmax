import gym
import numpy as np


class GVI:
    """Generalized Value Iteration (GVI)
    """

    def __init__(
        self,
        env: gym.Env,
        softmax,
        delta: float = 1e-3,
        max_iter: int = 1000,
        gamma: float = 0.98,
        rng: np.random.RandomState = np.random.RandomState(10)
    ) -> None:
        """Initialize GVI algorithm.

        Args:
            env: Gym environment.
            softmax: Softmax function.
            delta: Termination condition, default 1e-3.
            max_iter: Maximum iteration before termination, default 1000.
            gamma: Discount factor, default 0.98.
            rng: Numpy random number generator, default np.random.RandomState(10).
        """

        if not callable(softmax):
            raise TypeError('softmax must be callable.')
        if delta < 0:
            raise ValueError('delta must be greater than 0.')
        if max_iter < 0:
            raise ValueError('max_iter must be greater than 0.')

        self.env = env
        self.softmax = softmax
        self.delta = delta
        self.max_iter = max_iter
        self.gamma = gamma
        self.rng = rng

    def start(self) -> bool:
        """Start value iteration.

        Returns:
            done (bool): If it terminated before reaching max_iteration.
        """

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
        # Q = np.zeros((self.num_states, self.num_actions))
        Q = self.rng.normal(1, 1.0, (self.num_states, self.num_actions))

        num_iteration = 0
        done = False
        while True:
            num_iteration += 1
            diff = 0
            Q_ = Q.copy()
            # Q = np.zeros((self.num_states, self.num_actions))

            # update value function
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    q = 0
                    for s_ in range(self.num_states):
                        r = self.gamma * self.P[s, a, s_] * self.softmax(Q_[s_])

                        q += np.sum(r) + self.R[s, a, s_]
                    Q[s, a] = self.gamma * q

            # maximum difference
            diff = np.max(np.abs(Q - Q_))

            # termination conditions
            if self.delta and diff < self.delta:
                done = True
                break
            if self.max_iter and num_iteration >= self.max_iter:
                break

        self.Q = Q

        return num_iteration, done

    def set_env(self, env: gym.Env) -> None:
        self.env = env

    def set_softmax(self, softmax) -> None:
        if not callable(softmax):
            raise TypeError('softmax must be callable.')
        self.softmax = softmax

    def set_delta(self, delta: float) -> None:
        if delta < 0:
            raise ValueError('delta must be greater than 0.')
        self.delta = delta

    def set_max_iter(self, max_iter: int) -> None:
        if max_iter < 0:
            raise ValueError('max_iter must be greater than 0.')
        self.max_iter = max_iter

    def set_gamma(self, gamma: float) -> None:
        self.gamma = gamma

    def get_result(self) -> int:
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
