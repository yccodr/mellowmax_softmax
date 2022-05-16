import gym
import matplotlib.pyplot as plt
import numpy as np

from mellowmax_softmax import algo
from mellowmax_softmax import function as F

# def paint_q(history):
#     print(history.shape)
#     Q = np.split(history, indices_or_sections=2, axis=-1)
#     # print(Q[0].shape)
#     # print(Q[1].shape)
#     # print(Q)

#     plt.figure(figsize=(10, 10))
#     plt.plot(np.convolve(Q[0][:, 0], np.full(10, 1.0 / 10), mode='valid'))
#     plt.plot(np.convolve(Q[1][:, 0], np.full(10, 1.0 / 10), mode='valid'))
#     # plt.colorbar()
#     plt.savefig('./q.png')

env = gym.make('SimpleMDP-v0')

sarsa = algo.SARSA(
    env=env,
    policy=F.BoltzmannPolicy(beta=16.55),
    alpha=0.1,
    max_iter=1000,
)

q_history = []

for _ in range(10):
    num_iter, *_, Q = sarsa.start()
    print(num_iter, Q[0, :])
    q_history.append(Q.copy())

# print(q_history)
# q_history = np.array(q_history)

# # s1
# paint_q(q_history[:, 0, :])
