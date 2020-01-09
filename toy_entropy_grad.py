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
        log_prob = q.log_prob(x)
        l = log_prob.sum()
        true_score = torch.autograd.grad(l, x)[0]

        true_dH = true_score * x_grad
        true_dH = -true_dH.mean()

        dH = self.grad_estimator(x, x_grad)
        print("True Entropy Gradient:", true_dH)
        print("Estimated Entropy Gradient:", dH)


if __name__ == '__main__':
    # torch.manual_seed(1234)
    M = 100
    N = 1000

    LB = -5
    UB = 5
    z = (LB - UB) * torch.rand(N, 1) + UB
    z.requires_grad = True

    theta = torch.tensor([1.0])

    x = z * theta
    l = x.sum()
    x_grad = torch.autograd.grad(l, z)[0]


    eta = 0.95
    q = dist.StudentT(torch.tensor([5.0]))

    exp = ToyEntropyGrad(q, M, None, eta=eta)
    exp.run(x, x_grad)

