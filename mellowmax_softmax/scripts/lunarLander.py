import torch
import gym
from mellowmax_softmax.algo.reinforce import reinforce
from mellowmax_softmax.model.lunarLanderPolicy import policy
from mellowmax_softmax.function.boltzmax import boltzmax
from mellowmax_softmax.function.mellowmax import mellowmax

random_seed = 20
torch.manual_seed(random_seed)

env = gym.make('LunarLander-v2')
env.seed(random_seed)

discrete = isinstance(env.action_space, gym.spaces.Discrete)
observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.n if discrete else env.action_space.shape[0]
hidden_size = 128

policyNet = policy(action_dim, observation_dim, hidden_size)

PG = reinforce(env, policyNet, boltzmax, maxEpisodeNum=10)
PG.train()