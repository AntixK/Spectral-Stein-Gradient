import torch
from all_types import *
from score_estimator import SpectralSteinEstimator

class EntropyGradient:

    def __init__(self,
                 eta: float,
                 num_eigs: int = None) -> None:

        self.score_estimator = SpectralSteinEstimator(eta = eta,
                                                      num_eigs=num_eigs)

    def compute_gradients(self,
                          x: Tensor,
                          x_grad: Tensor,
                          samples: Tensor = None,
                          return_score = False) -> Tensor:

        """
        Computes the gradient of the entropy with respect to the
        model that produced the samples x that mimics the samples
        from some modelling distribution q, from a prior
        distribution p(z). The gradient of the entropy is given by
        .. math::
            x = g_\theta(z), z \sim p(z)
            \nabla_\theta H(q) = - \nabla E_z [\nabla_x \log q(x) \nabla_theta g_\theta(z)]

        :param x: (Tensor) Data samples from the q distribution
                  or a one that mimics its samples. [N x D]
        :param x_grad: (Tensor) Gradient of those samples with
                       respect to its transformation parameters
        :param samples: (Tensor) Samples from the distribution q
                        This is optional. [N x D]
        :return: (Tensor) Gradient of the entropy
        """
        N  = x.size(0)

        # Flatten the vectors
        x = x.view(N, -1) # [N x CHW]
        x_grad = x_grad.view(N, -1) #[N x CHW]

        with torch.no_grad():
            score = self.score_estimator(x, samples) #[N x CHW]

        grad_entropy = -(score * x_grad).mean() # Element-wise multiplication

        if return_score:
            return [grad_entropy, score]
        else:
            return grad_entropy

    def __call__(self,
                 x: Tensor,
                 x_grad: Tensor,
                 samples: Tensor = None,
                 return_score = False):

        return self.compute_gradients(x, x_grad, samples, return_score)

