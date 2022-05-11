import torch
import gym
from mellowmax_softmax.algo.reinforce import reinforce
from mellowmax_softmax.model.lunarLanderPolicy import policy
from mellowmax_softmax.function.boltzmax import BoltzmannPolicy
from mellowmax_softmax.function.mellowmax import MellowmaxPolicy
from mellowmax_softmax.function.eps_greedy import EpsGreedy

#####################
SEED = 7777
MAX_EP = 40000
#####################

torch.manual_seed(SEED)

env = gym.make('LunarLander-v2')
env.reset(seed=SEED)

discrete = isinstance(env.action_space, gym.spaces.Discrete)
observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.n if discrete else env.action_space.shape[0]
hidden_size = 128

policyNet = policy(observation_dim, action_dim, hidden_size)

PG = reinforce(env, policyNet, BoltzmannPolicy(16.55), maxEpisodeNum=MAX_EP)

# reward: list of episode rewards
rewards = PG.train()