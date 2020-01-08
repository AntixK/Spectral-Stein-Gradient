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
                          samples: Tensor = None) -> Tensor:
        """

        :param x: (Tensor) [N x C x H x W]
        :param x_grad: (Tensor) [N x C x H x W]
        :param samples: (Tensor) [M x C x H x W]
        :return: [N x C x H x W]
        """
        N  = x.size(0)

        if samples is not None:
            M = samples.size(0)
        else:
            samples = x
            M = N

        # Flatten the vectors
        x = x.view(N, -1) # [N x CHW]
        x_grad = x_grad.view(N, -1) #[N x CHW]
        samples = samples.view(M, -1) # [M x CHW]

        score = self.score_estimator(x, samples) #[N x CHW]

        grad_entropy = -(score * x_grad).mean(dim=-2) # Element-wise multiplication
        return grad_entropy #.view(N, C, H, W)

    def __call__(self,
                 x: Tensor,
                 x_grad: Tensor,
                 samples: Tensor):

        return self.compute_gradients(x, x_grad, samples)

