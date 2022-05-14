import json

import gym
import numpy as np
from mellowmax_softmax.algo.GVI import GVI
from mellowmax_softmax.function import boltzmax, mellowmax
from numpy import fix, random
from tqdm import tqdm


class GVIExp:
    ENV_NAME = ''
    EXP_NAME = 'random_mdp_boltz'
    FUNCTION_PARAM = 16.55
    SEED = 7414
    ITERATION = 100

    def __init__(self):
        random.seed = self.SEED

    def make_env(self):
        self.env = gym.make(self.ENV_NAME)

    def dump_result(self):
        # store result
        with open(f'./exp_results/{self.EXP_NAME}_log.json', 'w') as f:
            json.dump(result, f)


random.seed = 7414
ITERATION = 100
EXP_NAME = 'random_mdp_boltz'
BETA = 16.55

result = {
    'num': ITERATION,
    'avg_iter': 0,
    'multi_fixed_point': 0,
    'no_term': 0,
}


def test_termination(softmax):
    for round in range(200):
        env = gym.make('RandomMDP-v0')
        gvi = GVI()

        avg_iter = 0
        no_term = 0
        multi_fixed_point_cnt = 0
        fixed_points = []

        for seed in tqdm(range(ITERATION)):
            gvi.reset()
            gvi.set_env(env)
            gvi.set_softmax(softmax)

            # NOTE: delta not specified
            gvi.set_delta(1e-3)
            gvi.set_max_iter(1000)
            gvi.rng.seed(seed)

            q_max = []

            if not gvi.start():
                no_term += 1
            else:
                avg_iter += gvi.get_result()

            for s in range(gvi.num_states):
                idx = np.argmax(gvi.Q[s])
                q_max.append(np.unravel_index(idx, gvi.Q[s].shape))

        avg_iter /= (ITERATION - no_term)

        if q_max not in fixed_points or len(fixed_points) == 0:
            fixed_points.append(q_max)

        print(
            f'round: {round}, no terminate: {no_term}, fixed_points: {len(fixed_points)}, avg. iteration: {avg_iter}'
        )


exp = {
    'random_mdp_boltz': test_termination(boltzmax.Boltzmax(16.55)),
    'random_mdp_mellow': test_termination(mellowmax.Mellowmax(16.55))
}

exp['random_mdp_boltz']()

# store result
with open(f'./exp_results/{EXP_NAME}_log.json', 'w') as f:
    json.dump(result, f)
