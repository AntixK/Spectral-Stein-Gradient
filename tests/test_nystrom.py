import torch
import unittest
from score_estimator import SpectralSteinEstimator

class TestNystrom(unittest.TestCase):

    def setUp(self) -> None:
        self.score_est = SpectralSteinEstimator()

    def test_nystrom(self):
        x1 = torch.rand((5, 100), requires_grad=True)
        x2 = torch.rand((5, 100), requires_grad=True)
        x = torch.rand((3, 100), requires_grad=True)

        Kxx, x1_autograd, x2_autograd = self.score_est.grad_gram(x1, x2, 3)

        eigvals, eigvecs = torch.eig(Kxx, eigenvectors=True)

        print(self.score_est.nystrom_method(x, x2, eigvecs, eigvals, 3))


if __name__ == '__main__':
    unittest.main()