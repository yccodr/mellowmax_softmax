import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from mellowmax_softmax import algo
from mellowmax_softmax import function as F


def paint(history):
    plt.figure(figsize=(10, 10))
    plt.plot(
        np.convolve(history[:, 0], np.full(10, 1.0 / 10), mode='valid'),
        label='Q(s1, a)',
    )
    plt.plot(
        np.convolve(history[:, 1], np.full(10, 1.0 / 10), mode='valid'),
        label='Q(s1, b)',
    )
    plt.legend()
    plt.savefig('./q.png')


env = gym.make('SimpleMDP-v0')

sarsa = algo.SARSA(
    env=env,
    policy=F.BoltzmannPolicy(beta=16.55),
    alpha=0.1,
    max_iter=1000,
)

q_history = []

for _ in tqdm(range(2000)):
    num_iter, reward, *_, Q = sarsa.start()
    q_history.append(Q.copy())

q_history = np.array(q_history)

paint(q_history[:, 0, :])

data = pd.DataFrame({
    'Q(s1, a)': q_history[:, 0, 0],
    'Q(s1, b)': q_history[:, 0, 1],
})

data.to_csv('./simple_mdp_sarsa_log.csv')
