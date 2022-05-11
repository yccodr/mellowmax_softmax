import numpy as np
import pytest
import torch

import mellowmax_softmax.function as F


class TestBoltzmax:

    def test_boltzmax_np(self):
        beta = 5
        boltzmax = F.Boltzmax(beta=beta)

        arr = np.array([0.1, 0.2, 0.3])
        arr_exp = np.exp(beta * (arr - arr.max()))
        expected = np.inner(arr, arr_exp) / np.sum(arr_exp)

        assert pytest.approx(boltzmax(arr)) == expected

    def test_boltzmax_torch(self):
        beta = 5
        boltzmax = F.Boltzmax(beta=beta)

        arr = torch.Tensor([0.1, 0.2, 0.3])
        arr_exp = torch.exp(beta * (arr - arr.max()))
        expected = torch.inner(arr, arr_exp) / torch.sum(arr_exp)

        assert pytest.approx(boltzmax(arr)) == expected

    def test_boltzmax_exception(self):
        boltzmax = F.Boltzmax()

        with pytest.raises(TypeError):
            boltzmax(None)


class TestBoltzmannPolicy:

    def test_boltzmann_policy_np(self):
        beta = 5
        boltzmann_policy = F.BoltzmannPolicy(beta=beta)

        arr = np.array([0.1, 0.2, 0.3])
        arr_exp = np.exp(beta * (arr - arr.max()))
        expected = arr_exp / np.sum(arr_exp)

        assert pytest.approx(boltzmann_policy(arr)) == expected

    def test_boltzmann_policy_torch(self):
        beta = 5
        boltzmann_policy = F.BoltzmannPolicy(beta=beta)

        arr = torch.Tensor([0.1, 0.2, 0.3])
        arr_exp = torch.exp(beta * (arr - arr.max()))
        expected = arr_exp / torch.sum(arr_exp)

        assert pytest.approx(boltzmann_policy(arr)) == expected

    def test_boltzmann_policy_exception(self):
        boltzmann_policy = F.BoltzmannPolicy()

        with pytest.raises(TypeError):
            boltzmann_policy(None)


class TestEpsGreedy:

    def test_eps_greedy_np(self):
        rng = np.random.default_rng(10)
        # First rng.random() is 0.9560017096289753
        # Second rng.random() is 0.20768181007914688

        eps_greedy = F.EpsGreedy(eps=0.5, rng=rng)

        arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        expected1 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        assert pytest.approx(eps_greedy(arr)) == expected1

        expected2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        assert pytest.approx(eps_greedy(arr)) == expected2

    def test_eps_greedy_torch(self):
        rng = np.random.default_rng(10)
        # First rng.random() is 0.9560017096289753
        # Second rng.random() is 0.20768181007914688

        eps_greedy = F.EpsGreedy(eps=0.5, rng=rng)

        arr = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5])

        expected1 = torch.Tensor([0.0, 0.0, 0.0, 0.0, 1.0])
        assert pytest.approx(eps_greedy(arr)) == expected1

        expected2 = torch.Tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        assert pytest.approx(eps_greedy(arr)) == expected2

    def test_eps_greedy_exception(self):
        eps_greedy = F.EpsGreedy()

        with pytest.raises(TypeError):
            eps_greedy(None)
