import torch
import unittest
from score_estimator import SpectralSteinEstimator

class TestSSGE(unittest.TestCase):

    def setUp(self) -> None:
        self.score_est = SpectralSteinEstimator()
        torch.manual_seed(6534)

    def test_nystrom(self):
        x1 = torch.rand((5, 100), requires_grad=True)
        x2 = torch.rand((5, 100), requires_grad=True)
        x = torch.rand((3, 100), requires_grad=True)

        Kxx, x1_autograd, x2_autograd = self.score_est.grad_gram(x1, x2, 3)

        eigvals, eigvecs = torch.eig(Kxx, eigenvectors=True)

        phix = self.score_est.nystrom_method(x, x2, eigvecs, eigvals, 3)
        assert phix.size() == (3, 5), "Incorrect eigenfunction shape"

    def test_grads(self):
        x1 = torch.rand((16, 3, 3, 3), requires_grad=True)
        x2 = torch.rand((16, 3, 3, 3), requires_grad=True)


        x1 = x1.view(16, -1)
        x2 = x2.view(16, -1)
        grads = self.score_est.compute_score_gradients(x1, x2)
        print(grads.size())
        # assert grads.size() == (5, 3), "Incorrect grad shape"

if __name__ == '__main__':
    unittest.main()
