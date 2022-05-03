import gym
from mellowmax_softmax.algo.GVI import GVI
from mellowmax_softmax.function import boltzmax, mellowmax

env = gym.make('SimpleMDP-v0')

gvi = GVI(env=env, softmax=mellowmax.mellowmax)


# boltzman
def boltzman():
    for beta in range(1500, 2500):
        gvi.reset()
        gvi.set_env(env)
        gvi.set_softmax(boltzmax.boltzmax)
        gvi.set_delta(1e-10)
        gvi.set_max_iter(10000)

        gvi.beta = beta / 100

        finished = gvi.start()
        result = gvi.get_result()

        print(f'beta: {gvi.beta}, finished: {finished}, result: {result}')
        # print(f'Q: {gvi.Q}')


def mello():
    for beta in range(10, 250):
        gvi.reset()
        gvi.set_env(env)
        gvi.set_softmax(mellowmax.mellowmax)
        gvi.set_delta(1e-3)
        gvi.set_max_iter(1000)

        gvi.beta = beta / 10

        finished = gvi.start()
        result = gvi.get_result()

        print(f'beta: {gvi.beta}, finished: {finished}, result: {result}')
        # print(f'Q: {gvi.Q}')


boltzman()