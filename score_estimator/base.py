import torch
from all_types import *
from abc import abstractmethod



class BaseScoreEstimator:

    @staticmethod
    def rbf_kernel(x1: Tensor,
                   x2: Tensor,
                   sigma: float) -> Tensor:
        return torch.exp(-((x1 - x2).pow(2).sum(-1))/(2 * sigma ** 2))

    def gram_matrix(self,
             x1: Tensor,
             x2: Tensor,
             sigma: float) -> Tensor:
        x1 = x1.unsqueeze(-2) # Make it into a column tensor
        x2 = x2.unsqueeze(-3) # Make it into a row tensor
        return self.rbf_kernel(x1, x2, sigma)

    def grad_gram(self,
                  x1: Tensor,
                  x2: Tensor,
                  sigma: float) -> Tensor:
        """
        Computes the gradients of the RBF gram matrix with respect
        to the inputs x1 an x2. It is given by
        .. math::
            \nabla_x1 k(x1, x2) = k(x1, x2) \frac{x1- x2}{\sigma^2}

            \nabla_x2 k(x1, x2) = k(x1, x2) -\frac{x1- x2}{\sigma^2}

        :param x1: (Tensor) [N x D]
        :param x2: (Tensor) [M x D]
        :param sigma: (Float) Width of the RBF kernel
        :return: Gram matrix [N x M],
                 gradients with respect to x1 [N x M x D],
                 gradients with respect to x2 [N x M x D]

        """
        with torch.no_grad():
            Kxx = self.gram_matrix(x1, x2, sigma)

            x1 = x1.unsqueeze(-2)  # Make it into a column tensor
            x2 = x2.unsqueeze(-3)  # Make it into a row tensor
            diff = (x1 - x2) / (sigma ** 2)

            dKxx_dx1 = Kxx.unsqueeze(-1) * (-diff)
            dKxx_dx2 = Kxx.unsqueeze(-1) * diff
            return Kxx, dKxx_dx1, dKxx_dx2

    @abstractmethod
    def compute_score_gradients(self, x: Tensor, xm: Tensor):
        raise NotImplementedError
