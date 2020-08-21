# Statistics

To avoid unecessary technicalities, we will assume that we are working in a finite-dimensional setting and that all probability distributions are defined over $\mathbb{R}^n$.

We describe a probability distribution over $\mathbb{R}$ using a \emph{probability density function} $\pi$. The probability that a \emph{continuos random variable}, $\xi$, drawn from this distribution lies in the interval $(a,b)$ is given by

$$
P(a < \xi < b) = \int_a^b \pi(x)\mathrm{d}x.
$$

The mean is given by

$$
E(\xi) = \int_{-\infty}^{+\infty} x\, \pi(x)\mathrm{d}x,
$$

and the variance is given by

$$
V(\xi) =E(\xi^2) - E(\xi)^2.
$$

In the multivariate case, we have a multivariate PDF, $\pi: \mathbb{R}^n \rightarrow \mathbb{R}$ and we can define the mean in a similar fashion as before

$$
E(\xi) = \int_{\mathbb{R}^n} x\, \pi(x)\mathrm{d}^nx,
$$

and define the \emph{covariance matrix} as

$$
\Sigma = \int_{\mathbb{R}^n} (\xi - \mu)\transpose{(\xi - \mu)}\, \pi(x)\mathrm{d}^nx,
$$

where $\mu = E(\xi)$.

Central to our discussion here is *Bayes Theorem*

$$
P(A | B) = \frac{P(B | A) P(A)}{P(B)}.
$$
