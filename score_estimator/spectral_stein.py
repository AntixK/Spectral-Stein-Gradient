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
        score function. The SSGE is given by

        .. math::
            \nabla_{xi} phi_j(x) = \frac{1}{\mu_j M} \sum_{m=1}^M \nabla_{xi}k(x,x^m) \phi_j(x^m)

            \beta_{ij} = -\frac{1}{M} \sum_{m=1}^M \nabla_{xi} phi_j (x^m)

            \g_i(x) = \sum_{j=1}^J \beta_{ij} \phi_j(x)

        :param x: (Tensor) [N x D]
        :param xm: (Tensor) [M x D]
        :return: gradient estimate [N x D]
        """
        if xm is None:
            xm = x
        sigma = 3
        M = torch.tensor(xm.size(-2), dtype=torch.float)

        Kxx, dKxx_dx, _ = self.grad_gram(xm, xm, sigma)

        if self.eta is not None:
            Kxx += self.eta * torch.eye(M)

        eigen_vals, eigen_vecs = torch.eig(Kxx, eigenvectors=True)

        phi_x = self.nystrom_method(x, xm, eigen_vecs, eigen_vals, sigma)

        # Compute the Monte Carlo estimate of the gradient of
        # the eigenfunction at x
        dKxx_dx_avg = dKxx_dx.mean(dim=-3)
        mu = eigen_vals[:, 0].unsqueeze(-1) / torch.sqrt(M)
        #
        # beta1 = - torch.sqrt(M) * eigen_vecs.t() @ dKxx_dx_avg
        # beta1 *= (1. / eigen_vals[:, 0].unsqueeze(-1))
        #

        beta = - eigen_vecs.t() @ dKxx_dx_avg
        beta *= (1. / mu)

        # assert beta.allclose(beta1), f"incorrect computation {beta - beta1}"
        g = phi_x @ beta
        return g
