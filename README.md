# Spectral Stein Gradient Estimator
Pytorch implementation of the paper *A Spectral Approach to Gradient Estimation for Implicit Distributions* 
(https://arxiv.org/abs/1806.02925) by Shi et. al.

### Toy Experiments

<img src="https://github.com/AntixK/Spectral-Stein-Gradient/blob/master/assets/Gaussian.png" align="left" height="380" width="410" >

<img src="https://github.com/AntixK/Spectral-Stein-Gradient/blob/master/assets/Cauchy.png" align="left" height="380" width="410" >

<img src="https://github.com/AntixK/Spectral-Stein-Gradient/blob/master/assets/Student T.png" align="left" height="380" width="410" >

<img src="https://github.com/AntixK/Spectral-Stein-Gradient/blob/master/assets/LogNormal.png" align="left" height="380" width="410" >


### Usage
```python
latent_z = torch.randn((100, 1))
# Generative model with parameters theta
x = f(latent_z, theta)

# Gradient of the model with respect to its parameters
x_grad = df_dtheta

# Complex modelling distribution which can be sampled
model_dist = torch.distributions.Normal(torch.tensor([1.0]), torch.tensor([0.75]))
samples = model_dist.sample((100, ))

# Get the estimate of the score 
score_estimator = SpectralSteinEstimator(eta=0.0095)
score = score_estimator(x, samples)

# Compute the gradient of the entropy with 
# respect to the model parameters
grad_estimator = EntropyGradient(eta=0.0095)
entropy_grad = grad_estimator(x, x_grad, samples)
```

For getting started, see the list of toy examples.

### References

[1] Original Implementation in TensorFlow (https://github.com/thjashin/spectral-stein-grad)

