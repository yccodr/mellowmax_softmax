import json

import gym
import numpy as np
import pandas as pd
from mellowmax_softmax.algo.GVI import GVI
from mellowmax_softmax.function import boltzmax, mellowmax
from numpy import fix, random
from tqdm import tqdm, trange


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
    'round': [],
    'avg_iter': [],
    'multi_fixed_point': [],
    'no_term': [],
}


def test_termination(softmax):
    df = pd.DataFrame(result)

    for round in trange(200):
        env = gym.make('RandomMDP-v0')
        env.reset(seed=round)
        gvi = GVI()

        avg_iter = 0
        no_term = 0
        fixed_point = None
        multi_fixed_point = 0

        for seed in tqdm(range(ITERATION)):
            gvi.reset()
            gvi.set_rng(np.random.default_rng(seed + 200))
            gvi.set_env(env)
            gvi.set_softmax(softmax)

            # NOTE: delta not specified
            gvi.set_delta(1e-3)
            gvi.set_max_iter(1000)

            q_max = []

            if not gvi.start():
                no_term += 1
            else:
                avg_iter += gvi.get_result()

            for s in range(gvi.num_states):
                idx = np.argmax(gvi.Q[s])
                q_max.append(np.unravel_index(idx, gvi.Q[s].shape))

        valid_iter = ITERATION - no_term
        avg_iter = avg_iter / valid_iter if valid_iter > 0 else 0

        if fixed_point is None:
            fixed_point = q_max.copy()
        elif (fixed_point != q_max).all():
            multi_fixed_point = 1

        print(
            f'round: {round}, no terminate: {no_term}, fixed_points: {multi_fixed_point}, avg. iteration: {avg_iter}'
        )

        res = pd.Series([round, avg_iter, multi_fixed_point, no_term],
                        index=df.columns)

        df = pd.concat([df, res.to_frame().T], ignore_index=True)

        df.to_csv(f'exp_result/{EXP_NAME}_log.csv', encoding='utf-8')
        print(df)


exp = {
    'random_mdp_boltz': lambda: test_termination(boltzmax.Boltzmax(16.55)),
    'random_mdp_mellow': lambda: test_termination(mellowmax.Mellowmax(16.55))
}

if __name__ == '__main__':
    print(f'exp {EXP_NAME}')

    exp[EXP_NAME]()
