import torch
import torch.distributions as dist
from score_estimator import SpectralSteinEstimator

import matplotlib.pyplot as plt
plt.style.use("seaborn")


class ToyExperiment:

    def __init__(self, q, num_samples, num_eigs = None, eta = None):
        self.q = q
        self.M = num_samples
        self.J = num_eigs
        self.eta = eta

    def run(self, x):
        x.requires_grad = True

        x.requires_grad = True
        log_q_x = self.q.log_prob(x)
        l = log_q_x.sum()
        l.backward()
        true_dlog_q_dx = x.grad.detach()

        samples = self.q.sample((self.M,))
        score_estimator = SpectralSteinEstimator(eta=self.eta, num_eigs=self.J)
        dlog_q_dx = score_estimator(x, samples)

        return log_q_x.detach(), true_dlog_q_dx, dlog_q_dx.detach()


torch.manual_seed(1234)
mean = torch.tensor([0.0])
std = torch.tensor([1.0])

x = torch.linspace(-5,5, 150).view(-1, 1)

M = 100
eta = 0.095
q = dist.Normal(mean, std)
# q = dist.Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
# q = dist.Exponential(torch.tensor([1.0]))
# q = dist.LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
# q = dist.Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
# q = dist.StudentT(torch.tensor([5.0]))
exp = ToyExperiment(q, M, None, eta = eta)
lik_func, score, est_score = exp.run(x)

plt.figure()
plt.plot(x.detach(), score, lw = 2, label = r"$\nabla_x \log(x)$")
plt.plot(x.detach(), est_score, lw = 2, label = r"$\hat{\nabla}_x \log(x)$")
plt.plot(x.detach(), lik_func, lw = 2, label = r"$\log(x)$")
plt.title(f"Gaussian Distribution with {M} samples with $\eta$ = {eta}", fontsize = 15)
plt.legend(fontsize =15)
plt.savefig("assets/Gaussian.png", dpi = 300)
plt.show()