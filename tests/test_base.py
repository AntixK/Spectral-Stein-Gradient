import torch
import unittest
from score_estimator import BaseScoreEstimator


class Test_gram(unittest.TestCase):

    def setUp(self) -> None:
        self.score_est = BaseScoreEstimator()


    def test_gram(self):
        x1 = torch.rand((5, 10), requires_grad=True)
        x2 = torch.rand((5, 10), requires_grad=True)
        Kxx = self.score_est.gram_matrix(x1, x2, 3)
        assert Kxx.size(0) == x1.size(0), "Invalid size of gram matrix"
        assert Kxx.size(1) == x2.size(0), "Invalid size of gram matrix"

        assert Kxx.allclose(Kxx.t()), "Kernel matrix not symmetric"

    def test_gram_grad(self):
        x1 = torch.rand((5, 10), requires_grad=True)
        x2 = torch.rand((4, 10), requires_grad=True)

        Kxx, dKxx_dx1, dKxx_dx2 = self.score_est.grad_gram(x1, x2, 3)

        assert dKxx_dx1.size() == (5,4,10), "In correct x1_grad computation"
        assert dKxx_dx2.size() == (5,4,10), "In correct x2_grad computation"

    def test_heusristic_kw(self):

        x1 = torch.rand((5, 10), requires_grad=True)
        x2 = torch.rand((4, 10), requires_grad=True)

        self.score_est.heuristic_sigma(x1, x2)


if __name__ == '__main__':
    unittest.main()
