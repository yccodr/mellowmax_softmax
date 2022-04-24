import numpy as np
from ..function.mellowmax import mellowmax
from ..function.boltzmax import boltzmax


class GVI:

    def start(self,
              env,
              softmax: str,
              gamma,
              delta=None,
              max_iteration=None,
              beta=None,
              epsilon=None):
        """ start value iteration
        Args:
            env: environment
            softmax: 'mellow' or 'bolz'
            gamma: discount factor
            delta or max_iteration: termination condition
            beta: argument of bolzmax
            epsilon: argument of mellowmax

        Returns:
            int: number of iterations 
            bool: terminated before reaching max_iteration
        """
        if softmax == 'bolz':
            if not beta:
                raise ValueError("Missing argument: beta")
            self.softmax = boltzmax
            self.beta = beta
        elif softmax == 'mellow':
            if not epsilon:
                raise ValueError("Missing argument: omega")
            self.softmax = mellowmax
            self.epsilon = epsilon
        else:
            raise ValueError("Softmax function not found")
        if not delta or max_iteration:
            raise ValueError("Missing argument: delta or max_iteration")
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.delta = delta
        self.get_rewards_and_transitions_from_env(env)
        num_iteration, done = self.policy_iteration()
        self.reset()
        return num_iteration, done

    def get_rewards_and_transitions_from_env(self, env):
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.R = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                for s_ in range(self.num_states):
                    self.R[s, a, s_] = env.reward[s, a]
                    self.P[s, a, s_] = env.transition[s, a, s_]

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
            if self.max_iteration and num_iteration >= self.max_iteration:
                break
        return num_iteration, done

    def reset(self):
        self.beta = None
        self.epsilon = None
        self.gamma = None
        self.max_iteration = None