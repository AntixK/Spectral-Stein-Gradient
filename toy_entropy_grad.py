import torch
import torch.distributions as dist
from gradient_estimators import EntropyGradient
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
        self.grad_estimator = EntropyGradient(eta, num_eigs)

    def run(self, x: Tensor) -> List[Tensor]:

        x.requires_grad = True
        samples = self.q.sample((self.M, ))

        log_prob = torch.exp(self.q.log_prob(x)).sum()
        log_prob.backward()
        x_grad = x.grad

        grad_entropy = self.grad_estimator(x, x_grad, samples)

        entropy = self.q.entropy()
        entropy.backward()
        true_grad_entropy = self.q.probs.grad
        print(true_grad_entropy)
        # print(grad_entropy)


if __name__ == '__main__':
    torch.manual_seed(1234)
    x = torch.randint(0,2, (10, 1)).view(-1, 1).float()

    M = 10
    eta = 0.0095
    l = torch.tensor([.12])
    l.requires_grad = True
    q = dist.Bernoulli(l)

    # e = q.entropy()
    # e.backward()
    # print(l.grad)
    #
    # print(1./l)

    exp = ToyEntropyGrad(q, M, eta=eta)
    exp.run(x)



