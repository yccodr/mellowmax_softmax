import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from mellowmax_softmax import algo
from mellowmax_softmax import function as F


def paint(max_y, boltz_y, mm_y):
    plt.figure(figsize=(15, 15))
    plt.plot(max_y, label='max')
    plt.plot(boltz_y, label='boltz')
    plt.plot(mm_y, label='mm')
    plt.legend()
    plt.savefig('./simple_mdp_gvi.png')


env = gym.make('SimpleMDP-v0')

gvi = algo.GVI(
    env=env,
    delta=1e-15,
    max_iter=3000,
)

x = np.arange(start=1.0, stop=25.0, step=0.01, dtype=np.float32)

gvi.set_softmax(max)
gvi.start()
max_y = [gvi.get_result()] * len(x)

boltz_y = []

for beta in tqdm(x):
    gvi.set_softmax(F.Boltzmax(beta=beta))
    gvi.start()
    boltz_y.append(gvi.get_result())

print(f'boltz_max: {np.max(boltz_y)} at {np.argmax(boltz_y)/100}')

mm_y = []

for omega in tqdm(x):
    gvi.set_softmax(F.Mellowmax(omega=omega))
    gvi.start()
    mm_y.append(gvi.get_result())

paint(max_y, boltz_y, mm_y)

data = pd.DataFrame({
    'max': max_y,
    'boltz': boltz_y,
    'mm': mm_y,
})

data.to_csv('./simple_mdp_gvi_log.csv')
