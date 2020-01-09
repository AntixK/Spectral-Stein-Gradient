import torch
import torch.distributions as dist
from score_estimator import SpectralSteinEstimator
from all_types import *
import matplotlib.pyplot as plt
plt.style.use("seaborn")


class ToyScoreEstimator:

    def __init__(self,
                 q: dist,
                 num_samples: int,
                 num_eigs: int = None,
                 eta:float = None) -> None:

        self.q = q
        self.M = num_samples

        self.score_estimator = SpectralSteinEstimator(eta=eta, num_eigs=num_eigs)

    def run(self,
            x: Tensor) -> List[Tensor]:

        # Compute True Score
        x.requires_grad = True
        log_q_x = self.q.log_prob(x)
        l = log_q_x.sum()
        l.backward()
        true_dlog_q_dx = x.grad.detach()


        samples = self.q.sample((self.M,))

        dlog_q_dx = self.score_estimator(x, samples)

        return [log_q_x.detach(), true_dlog_q_dx, dlog_q_dx.detach()]

if __name__ == '__main__':
    # torch.manual_seed(1234)

    LB = -5
    UB = 5
    # x = (LB - UB) * torch.rand(150, 1) + UB
    x = torch.linspace(LB,UB, 150).view(-1, 1)

    M = 100
    eta = 0.95
    q = dist.Normal(torch.tensor([0.0]), torch.tensor([1.]))
    # q = dist.Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
    # q = dist.Exponential(torch.tensor([1.0]))
    # q = dist.LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
    # q = dist.Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
    # q = dist.StudentT(torch.tensor([5.0]))
    exp = ToyScoreEstimator(q, M, None, eta = eta)
    lik_func, score, est_score = exp.run(x)

    plt.figure()
    plt.plot(x.detach(), score, lw = 2, label = r"$\nabla_x \log(x)$")
    plt.plot(x.detach(), est_score, lw = 2, label = r"$\hat{\nabla}_x \log(x)$")
    plt.plot(x.detach(), lik_func, lw = 2, label = r"$\log(x)$")
    plt.title(f"Gaussian Distribution with {M} samples with $\eta$ = {eta}", fontsize = 15)
    plt.legend(fontsize =15)
    plt.savefig("assets/Gaussian.png", dpi = 300)
    plt.show()