import numpy as np


class SARSA:
    """
    State-action-reward-state-action (SARSA)
    """
    def __init__(self,
                 env=None,
                 policy=None,
                 alpha=0.1,
                 max_iter=1000,
                 gamma=0.98) -> None:
        """ initialize SARSA algorithm
        Arguments:
            env: gym environment
            policy: policy to choose action. ['eps_greedy', 'boltzmann', 'mellowmax']
            alpha: learning rate, default: 0.1
            max_iter: maximum iteration before termination, default: 1000
            gamma: discount factor, default: 0.98
        """

        self.env = env
        self.policy = policy
        self.alpha = alpha
        self.max_iter = max_iter
        self.gamma = gamma
        self.softmax = None

    def start(self) -> bool:
        """ start value iteration
        Returns:
            bool: terminated before reaching max_iteration
        """
        if self.env == None:
            raise ValueError("self.env is None")

        if self.policy == None:
            raise ValueError("self.policy is None")

        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n

        self.num_iter, self.reward, done = self.run()
        return done

    def run(self):
        # initialize acion value function Q
        Q = np.zeros((self.num_states, self.num_actions))
        # Q = np.random.normal(1, 1.0, (self.num_states, self.num_actions))

        num_iteration = 0
        reward = 0
        done = 0
        while True:
            next_state, reward, done, info = self.env.step(action)
            # choose action
            next_action = np.random.choice(range(self.num_actions), p=self.policy(Q[next_state]))
            self.update_value(state, action, reward, next_state, next_action,
                              done, Q)

            reward += reward
            state, action = next_state, next_action

            if done:
                done = 1
                break
            if num_iteration > self.max_iter:
                break

        self.Q = Q

        return num_iteration, reward, done

    def update_value(self, state, action, reward, next_state, next_action, done,
                     Q):
        if done:
            Q[state][action] += self.alpha * (reward - Q[state][action])
        else:
            Q[state][action] += self.alpha * (
                reward + self.gamma * Q[next_state][next_action] -
                Q[state][action])

    def set_env(self, env) -> None:
        self.env = env

    def set_policy(self, policy) -> None:
        self.policy = policy

    def set_alpha(self, alpha) -> None:
        self.alpha = alpha

    def set_max_iter(self, max_iter) -> None:
        self.max_iter = max_iter

    def set_gamma(self, gamma) -> None:
        self.gamma = gamma

    def get_result(self) -> int:
        """
        return:
            int: number of iteration executed before termination
        """
        return self.num_iter

    def reset(self) -> None:
        self.env = None
        self.policy = None
        self.alpha = 0.1
        self.max_iter = 1000
        self.gamma = 0.98
