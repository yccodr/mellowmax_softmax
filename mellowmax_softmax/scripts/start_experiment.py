import argparse

import gym

from mellowmax_softmax.algo.gvi import GVI
from mellowmax_softmax.function import boltzmax, mellowmax

arg_exp = {
    "simple_mdp": 'SimpleMDP-v0',
    "lunar_lander": 'LunarLander-v2',
    "random_mdp": 'RandomMDP-v0',
    "taxi": 'Taxi-v3',
}


def run_app():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e",
                        "--env",
                        "--environment",
                        type=str,
                        default="simple_mdp",
                        help="Environment name")

    args = parser.parse_args()

    if args.env not in arg_exp:
        raise ValueError("Environment name not found")

    env = gym.make(arg_exp[args.env])

    gvi = GVI(env=env, softmax=mellowmax.mellowmax)

    finished = gvi.start()
    result = gvi.get_result()

    print(f'finished: {finished}, result: {result}')