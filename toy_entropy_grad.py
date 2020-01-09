import torch
import torch.distributions as dist
from gradient_estimators import EntropyGradient
from score_estimator import SpectralSteinEstimator
from all_types import *
import matplotlib.pyplot as plt
plt.style.use("seaborn")


class ToyEntropyGrad:
    def __init__(self,
                 q: dist,
                 num_samples: int,
                 num_eigs: int = None,
                 eta:float = None) -> None:

        self.q = q
        self.M = num_samples
        self.grad_estimator =  EntropyGradient(eta=eta, num_eigs=num_eigs)

    def run(self,
            x: Tensor,
            x_grad: Tensor) -> List[Tensor]:
        """

        :param x: (Tensor) Data samples from the q distribution
                  or a one that mimics its samples
        :param x_grad: (Tensor) Gradient of those samples with
                       respect to its transformation parameters
        :return:
        """

        dH, score = self.grad_estimator(x, x_grad, return_score = True)
        return dH


if __name__ == '__main__':
    # torch.manual_seed(1234)
    M = 100
    N = 100

    z = torch.linspace(-5,5, 150).view(-1, 1)

    mu = torch.tensor([1.0])
    sigma = torch.tensor([1.75])

    x = z * mu + sigma
    l = x.sum()
    x_grad = z

    eta = 0.0095
    q = dist.Normal(torch.tensor([1.0]), torch.tensor([0.75]))

    exp = ToyEntropyGrad(q, M, None, eta=eta)
    dH = exp.run(x, x_grad)
    print("Estimated Entropy Gradient:", dH.item())