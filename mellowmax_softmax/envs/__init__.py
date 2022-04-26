from gym.envs.registration import register

register(
    id='SimpleMDP-v0',
    entry_point='mellowmax_softmax.envs.custom_mdp:CustomMDP',
    max_episode_steps=1000,
)

register(
    id='RandomMDP-v0',
    entry_point='mellowmax_softmax.envs.random_mdp:RandomMDP',
    max_episode_steps=1000,
)
