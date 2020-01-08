import pyro
import pyro.infer
import pyro.optim
from pyro import poutine
import torch
from torch.distributions import constraints
import pyro.distributions as dist
from tqdm import tqdm


pyro.set_rng_seed(1234)


class Experiment:

    def __init__(self, prior_params):
        self.prior_params = prior_params

    def _prior_sample(self):
        """
        P(\theta)
        :return:
        """
        return pyro.sample('theta', dist.Beta(**self.prior_params))

    def _likelihood_dist(self):
        """
        P(x | \theta)
        :param theta:
        :return:
        """
        return dist.Bernoulli(self.theta, 0.75)

    def _posterior_sample(self, obs, obs_ind):
        """
        P(\theta | x)
        :param likelihood:
        :param obs:
        :param obs_ind: Index of the observed data (for naming)
        :return:
        """
        return pyro.sample('obs_{}'.format(obs_ind), self._likelihood_dist(), obs=obs)

    def _model(self, observations):

        self.theta = self._prior_sample()
        for i in range(len(observations)):
            self._posterior_sample(observations[i], i)


    def _variational_sample(self, observations):
        alpha_q = pyro.param('alpha_q',torch.tensor(15.0),
                         constraint=constraints.positive)
        beta_q= pyro.param('beta_q', torch.tensor(15.0),
                         constraint = constraints.positive)
        return pyro.sample('theta', dist.Beta(alpha_q, beta_q))

    def _loss_func(self):
        return pyro.infer.Trace_ELBO()

    def _loss(self, model, guide, *args, **kwargs):
        guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)

        model_trace = poutine.trace(
            poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
        # construct the elbo loss function
        energy_term = model_trace.log_prob_sum()
        entropy_term = guide_trace.log_prob_sum()

        return -1 * (energy_term - entropy_term)


    def run(self,
            obervations,
            num_iter: int):

        # pyro.enable_validation(True)
        pyro.clear_param_store()

        optimizer = pyro.optim.Adam({"lr": 0.0005, "betas": (0.90, 0.999)})

        svi = pyro.infer.SVI(model = self._model,
                             guide = self._variational_sample,
                             optim = optimizer,
                             loss = self._loss_func())

        for t in tqdm(range(num_iter)):
            svi.step(obervations)

        return pyro.param('alpha_q').item(), pyro.param('beta_q').item()


exp = Experiment({'concentration1':10., 'concentration0': 10.},)
observations = []
for _ in range(60):
    observations.append(torch.tensor(1.0))
for _ in range(1):
    observations.append(torch.tensor(0.0))

a, b = exp.run(observations, 2000)
print(a/(a+b))
