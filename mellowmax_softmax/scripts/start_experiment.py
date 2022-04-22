import argparse

from mellowmax_softmax.envs import get_env

arg_exp = {
    "custom_mdp": 'CustomMDP',
    "lunar_lander": 'LunarLander',
    "random_mdp": 'RandomMDP',
    "taxi": 'Taxi',
}


def run_app():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e",
                        "--env",
                        "--environment",
                        type=str,
                        default="custom_mdp",
                        help="Environment name")

    args = parser.parse_args()

    if args.environment not in arg_exp:
        raise ValueError("Environment name not found")

    env = get_env(arg_exp[args.experiment])
