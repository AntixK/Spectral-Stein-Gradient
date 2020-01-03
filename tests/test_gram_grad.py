import torch
import unittest
from score_estimator import BaseScoreEstimator


class Test_gram_grad(unittest.TestCase):

    def setUp(self) -> None:
        self.score_est = BaseScoreEstimator()


    def test_gram(self):
        x1 = torch.rand((5, 10), requires_grad=True)
        x2 = torch.rand((5, 10), requires_grad=True)
        Kxx = self.score_est.gram_matrix(x1, x2, 3)
        assert Kxx.size(0) == x1.size(0), "Invalid size of gram matrix"
        assert Kxx.size(1) == x2.size(0), "Invalid size of gram matrix"

        assert Kxx.allclose(Kxx.t()), "Kernel matrix not symmetric"

    def test_gram_gram(self):
        x1 = torch.rand((5, 10), requires_grad=True)
        x2 = torch.rand((4, 10), requires_grad=True)

        Kxx, dKxx_dx1, dKxx_dx2 = self.score_est.grad_gram(x1, x2, 3)

        assert dKxx_dx1.size() == (5,4,10), "In correct x1_grad computation"
        assert dKxx_dx2.size() == (5,4,10), "In correct x2_grad computation"


if __name__ == '__main__':
    unittest.main()
