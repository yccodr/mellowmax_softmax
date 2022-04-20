import argparse
from enum import Enum

from mellowmax_softmax.experiment.custom_mdp import CustumMDP
from mellowmax_softmax.experiment.lunar_lander import LunarLander
from mellowmax_softmax.experiment.random_mdp import RandomMDP
from mellowmax_softmax.experiment.taxi import Taxi


class Experiments(Enum):
    CUSTOM_MDP = CustumMDP
    RANDOM_MDP = RandomMDP
    TAXI = Taxi
    LUNAR_LANDER = LunarLander


arg_exp = {
    "custom_mdp": Experiments.CUSTOM_MDP,
    "random_mdp": Experiments.RANDOM_MDP,
    "taxi": Experiments.TAXI,
    "lunar_lander": Experiments.LUNAR_LANDER
}


def run_app():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e",
                        "--experiment",
                        type=str,
                        default="custom_mdp",
                        help="Experiment name")

    args = parser.parse_args()

    if args.experiment not in arg_exp:
        raise ValueError("Experiment name not found")

    # instantiate Experiment class
    exp = arg_exp[args.experiment].value()

    exp.start()
