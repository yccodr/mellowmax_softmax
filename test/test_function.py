import numpy as np
import pytest
import torch

import mellowmax_softmax.function as F


def test_boltzmax_np():
    beta = 5
    boltzmax = F.Boltzmax(beta=beta)

    arr = np.array([0.1, 0.2, 0.3])
    arr_exp = np.exp(beta * (arr - arr.max()))
    expected = np.inner(arr, arr_exp) / np.sum(arr_exp)

    assert pytest.approx(boltzmax(arr)) == expected


def test_boltzmax_torch():
    beta = 5
    boltzmax = F.Boltzmax(beta=beta)

    arr = torch.Tensor([0.1, 0.2, 0.3])
    arr_exp = torch.exp(beta * (arr - arr.max()))
    expected = torch.inner(arr, arr_exp) / torch.sum(arr_exp)

    assert pytest.approx(boltzmax(arr)) == expected


def test_boltzmax_exception():
    boltzmax = F.Boltzmax()

    with pytest.raises(TypeError):
        boltzmax(None)


def test_eps_greedy_np():
    rng = np.random.default_rng(10)
    # First rng.random() is 0.9560017096289753
    # Second rng.random() is 0.20768181007914688

    eps_greedy = F.EpsGreedy(eps=0.5, rng=rng)

    arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    expected1 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    assert pytest.approx(eps_greedy(arr)) == expected1

    expected2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    assert pytest.approx(eps_greedy(arr)) == expected2


def test_eps_greedy_torch():
    rng = np.random.default_rng(10)
    # First rng.random() is 0.9560017096289753
    # Second rng.random() is 0.20768181007914688

    eps_greedy = F.EpsGreedy(eps=0.5, rng=rng)

    arr = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    expected1 = torch.Tensor([0.0, 0.0, 0.0, 0.0, 1.0])
    assert pytest.approx(eps_greedy(arr)) == expected1

    expected2 = torch.Tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    assert pytest.approx(eps_greedy(arr)) == expected2


def test_eps_greedy_exception():
    eps_greedy = F.EpsGreedy()

    with pytest.raises(TypeError):
        eps_greedy(None)
