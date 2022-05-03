import gym
from numpy import random
from mellowmax_softmax.algo.GVI import GVI
from mellowmax_softmax.function import boltzmax, mellowmax
from tqdm import tqdm
import json

random.seed = 7414
ITERATION = 100
EXP_NAME = 'random_mdp_boltz'

result = {
    'num': ITERATION,
    'avg_iter': 0,
    'multi_fixed_point': 0,
    'no_term': 0,
}


def boltz_test_termination():
    for round in range(200):
        env = gym.make('RandomMDP-v0')

        gvi = GVI()
        gvi.beta = 16.55

        avg_iter = 0
        no_term = 0
        multi_fixed_point_cnt = 0

        for _ in tqdm(range(ITERATION)):
            gvi.reset()
            gvi.set_env(env)
            gvi.set_softmax(boltzmax.boltzmax)

            # NOTE: delta not specified
            gvi.set_delta(1e-3)
            gvi.set_max_iter(1000)

            if not gvi.start():
                no_term += 1
            else:
                avg_iter += gvi.get_result()

        result['avg_iter'] += avg_iter

        avg_iter /= (ITERATION - no_term)

        print(
            f'round: {round}, no terminate: {no_term}, avg. iteration: {avg_iter}'
        )

        if no_term > 0:
            result['no_term'] += 1

        if multi_fixed_point_cnt > 0:
            result['multi_fixed_point'] += 1

        result['avg_iter'] += avg_iter

    result['avg_iter'] /= (ITERATION * 200 - result['no_term'])


def mellow_test_termination():
    for round in range(200):
        env = gym.make('RandomMDP-v0')
        gvi = GVI()
        gvi.beta = 16.55

        avg_iter = 0
        no_term = 0
        multi_fixed_point_cnt = 0

        for _ in tqdm(range(ITERATION)):
            gvi.reset()
            gvi.set_env(env)
            gvi.set_softmax(mellowmax.mellowmax)

            # NOTE: delta not specified
            gvi.set_delta(1e-3)
            gvi.set_max_iter(1000)

            if not gvi.start():
                no_term += 1
            else:
                avg_iter += gvi.get_result()

        avg_iter /= (ITERATION - no_term)

        print(
            f'round: {round}, no terminate: {no_term}, avg. iteration: {avg_iter}'
        )


exp = {
    'random_mdp_boltz': boltz_test_termination,
    'random_mdp_mellow': mellow_test_termination
}

exp['random_mdp_boltz']()

# store result
with open(f'./exp_results/{EXP_NAME}_log.json', 'w') as f:
    json.dump(result, f)
