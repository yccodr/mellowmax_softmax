from itertools import count
import numpy as np

import torch
from torch.optim import Adam
from torch.distributions import Categorical
from tqdm import tqdm


class reinforce():

    def __init__(self,
                 env,
                 policy,
                 softmax,
                 gamma=0.99,
                 lr=0.005,
                 maxEpisodeNum=40000) -> None:
        self.env = env
        self.policy = policy
        self.softmax = softmax
        self.gamma = gamma
        self.lr = lr
        self.maxEpisodeNum = maxEpisodeNum
        self.savedAction = []
        self.rewards = []
        self.epRewardHistory = []

        self.optim = Adam(self.policy.parameters(), lr=self.lr)

    def selectAction(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        probs = self.softmax(probs)
        m = Categorical(probs)
        action = m.sample()

        self.savedAction.append(m.log_prob(action)[0])

        return action.item()

    def calculateLoss(self):
        savedAction = torch.stack(self.savedAction)

        returns = []
        reversedRewards = np.flip(self.rewards, 0)
        g_t = 0
        for r in reversedRewards:
            g_t = r + self.gamma * g_t
            returns.insert(0, g_t)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / returns.std()

        loss = -torch.inner(returns.detach(), savedAction)

        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.savedAction[:]

    def update(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def train(self):
        self.epRewardHistory = []
        self.policy.train()

        for i_episode in tqdm(range(self.maxEpisodeNum)):
            t = 0
            state = torch.Tensor(self.env.reset())

            while t < 9999:
                t += 1
                action = self.selectAction(state)
                state, reward, done, _ = self.env.step(action)
                self.rewards.append(reward)
                if done:
                    break

            ep_reward = sum(self.rewards)
            self.epRewardHistory.append(ep_reward)
            loss = self.calculateLoss()
            self.update(loss)
            self.clear_memory()
            
        return self.epRewardHistory

    def getRewardHistory(self):
        return self.epRewardHistory
