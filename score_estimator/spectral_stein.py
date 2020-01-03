import torch
from all_types import *
from .base import BaseScoreEstimator


class SpectralSteinEstimator(BaseScoreEstimator):
    def __init__(self, eta: float = None) -> None:
        self.eta = eta

    def nystrom_method(self,
                       x: Tensor,
                       eval_points: Tensor,
                       eigen_vecs: Tensor,
                       eigen_vals: Tensor,
                       kernel_sigma: float) -> Tensor:
        """
        Implements the Nystrom method for approximating the
        eigenfunction (generalized eigenvectors) for the kernel
        at x using the M eval_points (x_m). It is given
        by -

         .. math::
            phi_j(x) = \frac{M}{\lambda_j} \sum_{m=1}^M u_{jm} k(x, x_m)

        :param x: (Tensor) Point at which the eigenfunction is evaluated [N x D]
        :param eval_points: (Tensor) Sample points from the data of ize M [M x D]
        :param eigen_vecs: (Tensor) Eigenvectors of the gram matrix [M x M]
        :param eigen_vals: (Tensor) Eigenvalues of the gram matrix [M x 2]
        :param kernel_sigma: (Float) Kernel width
        :return: Eigenfunction at x [N x M]
        """
        M = torch.tensor(eval_points.size(-2), dtype=torch.float)
        Kxxm = self.gram_matrix(x, eval_points, kernel_sigma)
        phi_x =  torch.sqrt(M) * Kxxm @ eigen_vecs

        phi_x *= 1. / eigen_vals[:,0] # Take only the real part of the eigenvals
                                      # as the Im is 0 (Symmetric matrix)
        return phi_x


    def compute_score_gradients(self,
                                x: Tensor,
                                xm: Tensor = None) -> Tensor:
        """
        Computes the Spectral Stein Gradient Estimate (SSGE) for the
        score function.
        :param x: (Tensor)
        :param xm: (Tensor
        :return:
        """
        if xm is None:
            xm = x

        M = torch.tensor(torch.size(x)[-2], dtype=torch.float)

        Kxx, dKxx_dx1, dKxx_dx2 = self.grad_gram(x, x, sigma)

        if self.eta is not None:
            Kxx += self.eta * torch.eye(M)

        eigen_vals, eigen_vecs = torch.eig(Kxx, eigenvectors=True)

        phi_x = self.nystrom_method(x, xm, eigen_vecs, eigen_vals, sigma)

        # Monte-Carlo estimate of the gradient expectation
        dKxx_dx1_avg = dKxx_dx1.mean(dim = -3)



