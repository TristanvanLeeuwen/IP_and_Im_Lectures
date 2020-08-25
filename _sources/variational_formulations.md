---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.9'
    jupytext_version: 1.5.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Variational formulations for inverse problems

## Analysis

## Linear PDEs

## Non-linear PDEs

+++

## Exercises

+++

### Well-posedness and optimality conditions

The following functionals are given (for $\alpha > 0$ and $A \in \mathbb{R}^{2 \times 2}$ an invertible matrix):

* $J_1: \mathbb{R} \rightarrow \mathbb{R}, u \mapsto \frac{1}2(u-f)^2 + \alpha|u|$
* $J_2: \mathbb{R} \rightarrow \mathbb{R}, u \mapsto |u-f| + \alpha u^2$
* $J_3: \mathbb{R}^2 \rightarrow \mathbb{R}, u \mapsto \frac{1}2\Vert A u - f\Vert_{\ell^2}^2  + \alpha \Vert u \Vert_{\ell^2}$

For the optimisation problems $J_i(u) \rightarrow \min_u$ perform the following analysis:

__Proof__, that a minimum exists (use the fundamental theorem of optimisation) and __proof__ that it is unique.

__Compute__ the optimality conditions and thereof (using cases) a solution formula dependent on $f$. It holds for $p\in\partial\left\|u\right\|_{\ell^2}$ that
$$p = \frac{u}{\left\|u\right\|_{\ell^2}} \text{ for } \; u\neq 0 \text{, and }$$
$$p \in \mathrm{B}_1(0) \quad \text{ for } \; u = 0\, \qquad$$
where $\mathrm{B}_1(0)$ denotes the unit ball around $0$.

Hint: Remark, that for $J_3$ no explicit solution formula can be given. Hence, use the following substitution
$c:=\frac{\alpha}{\left\|u\right\|_{\ell^2}}$ and provide a solution formula dependent on $c$ and $f$.

+++

### Well-posedness of the ROF model

* For existence of a solution, verify why the TV functional is lower semi-continuous in the corresponding topology.

* Does the Rudin-Osher-Fatemi model have a unique minimizer? Why or why not?

+++

### Deconvolution using the $L_1$ norm

For given data $f$ and a convolution kernel $k$ we study the following regularized variational method:

$$ 
\left\| k \ast u - f \right\|_{L^2(\Omega)}^2 \:+\: \alpha \: \int_\Omega | (\mathcal{F}u)(w) | \: dw \: \rightarrow \: \min_{u} 
$$

where $(\mathcal{F}u)(w)$ denotes the Fourier transform of $u$ at wave number $w$. Similar to the lecture, find an explicit representation of the solution of the problem using the [convolution theorem](https://en.wikipedia.org/wiki/Convolution_theorem) and the [Plancherel theorem](https://en.wikipedia.org/wiki/Plancherel_theorem). For simplicity you can assume that everything is real valued.

Hint: The derivative of the absolute value function is multivalued (you need cases).

+++

### Fréchet derivatives

Let $\Omega \subset \mathbb{R}^2$ and $\Sigma \subset \mathbb{R}^2$. Compute the Fréchet derivatives of the following functionals:

* $J(u) = \frac{1}{2} \left\| \nabla u \right\|_{L^2(\Omega)}^2$ where $u \in W^{1,2}(\Omega)$.

```{admonition} Answer
:class: tip, dropdown

We have $J(u + h) = \textstyle{\frac{1}{2}}\|\nabla u + \nabla h\|^2 = \textstyle{\frac{1}{2}}\|\nabla u\|^2 +  \int_{\Omega}\nabla u(x) \cdot \nabla h(x) \mathrm{d}x + \textstyle{\frac{1}{2}}\|\nabla h\|^2$. This suggests that $DJ(u) : U \rightarrow \mathbb{R}$ can be defined as $DJ(u)v = \int_{\Omega}\nabla u(x) \cdot \nabla v(x) \mathrm{d}x$. Indeed, we can verify that

$$
\lim_{\|h\|\rightarrow 0} \frac{\left| \int_{\Omega}\nabla h (x) \cdot \nabla h(x)\mathrm{d}x \right|}{\sqrt{\int_{\Omega} |h(x)|^2 + |\nabla h(x) \cdot \nabla h(x) | \mathrm{d}x}} = 0.
$$
```

* $J(u) = \frac{1}{2} \left\| Ku-f \right\|_{L^2(\Sigma)}^2$ where $K: L^2(\Omega) \rightarrow L^2(\Sigma)$ is a compact linear operator, $u : \Omega \rightarrow \mathbb{R}$ and $f : \Sigma \rightarrow \mathbb{R}$.

```{admonition} Answer
:class: tip, dropdown

We have $J(u+h) = J(u) + \langle Ku - f, Kv \rangle + \textstyle{\frac{1}{2}}\|Kv\|^2.$ This suggests letting $DJ(u)v = \langle Ku - f, Kv\rangle = \langle K^*(Ku - f), v\rangle$. Indeed

$$
\lim_{\|h\|\rightarrow 0}\frac{\|Kh\|^2_{L^2}}{\|h\|_{L^2}} = 0,
$$

because $K$ is bounded.
```

* $J(\mathbf{v}) = \frac{1}{2} \left\| \partial_t f + \nabla\cdot(f \mathbf{v}) \right\|_{L^2(\Omega \times [0,T])}^2$ 
	where $f$ here represents an image sequence, i.e. $f: \Omega \times [0,T] \rightarrow \mathbb{R}$, and $\mathbf{v}$ denotes a desired vector field, i.e. $\mathbf{v}: \Omega \times [0,T] \rightarrow \mathbb{R}^2$.

```{admonition} Answer
:class: tip, dropdown

Here, we have $J(\mathbf{v} + \mathbf{h}) = J(\mathbf{v}) + \langle \partial_t f + \nabla \cdot (f\mathbf{v}),  \nabla \cdot (f\mathbf{h})\rangle + \textstyle{\frac{1}{2}}\|\nabla \cdot (f\mathbf{h})\|^2$, suggesting

$$
DJ(\mathbf{v})\mathbf{h} = \int_0^T \int_{\Omega} \left(\partial_t f(x,t) + \nabla \cdot (f(x,t)\mathbf{v}(x,t))\right)\left(\nabla \cdot (f(x,t)\mathbf{v}(x,t))\right) \mathrm{d}t\mathrm{d}x.
$$
```

+++

### $\ell_2$-denoising

Consider the Tikhonov functional for denoising (in $\mathbb{R}^n$):

$$
\min\limits_u\textstyle{\frac{1}{2}}\left\|u-f\right\|_2^2 + \frac{\alpha}{2}\left\|u\right\|_2^2.
$$

* Give the solution of this variational problem explicitly.
* Generate in Python a random 1x128 vector with 5 non-zero coefficients (entries) and add aussian noise with standard deviation $\sigma = 0.05$ (see example below)
* Denoise the vector by solving the variational problem. What happens for different regularisation parameters $\alpha = \left\{0.01, 0.05, 0.1, 0.2\right\}$?. Consider in particular $\alpha=0.1$. *Is the solution sparse?*

```{code-cell} ipython3
# import libraries
import numpy as np
import matplotlib.pyplot as plt

# parameters
n = 128
k = 5
sigma = 0.05

# generate spiky signal with random amplitudes
u = np.zeros(n)
u[np.random.randint(128,size=k)] = np.random.randn(k)

# generate noisy signal
f = u + sigma*np.random.randn(n)

# plot
plt.plot(u)
plt.plot(f)
```

```{admonition} Answer
:class: tip, dropdown

* The solution is given by $u = (1 + \alpha)^{-1}f.$
* The results are shown below (click `+` to show the code), showing that (as expected), the result is only scaled down. This obviously reduces the noise level but also effects the amplitude of the spikes.
```

```{code-cell} ipython3
:tags: [hide-cell]

# import libraries
import numpy as np
import matplotlib.pyplot as plt

# parameters
n = 128
k = 5
sigma = 0.05

# random seed
np.random.seed(1)

# generate spiky signal with random amplitudes
u = np.zeros(n)
u[np.random.randint(128,size=k)] = np.random.randn(k)

# generate noisy signal
f = u + sigma*np.random.randn(n)

# denoise
alpha = .1
uhat = f/(1 + alpha)

# plot
plt.plot(u, label='ground truth')
plt.plot(uhat, label='denoised signal')
plt.legend()
plt.show()
```

### $\ell_1$-denoising

Repeat the previous exercise for the $\ell_1$-densoining problem

$$
\min\limits_u\textstyle{\frac{1}{2}}\left\|u-f\right\|_2^2 + \alpha \left\|u\right\|_1.
$$

+++

```{admonition} Answer
:class: tip, dropdown

The exact solution known as *soft tresholding*. A derivation can be found [here](https://math.stackexchange.com/questions/471339/derivation-of-soft-thresholding-operator-proximal-operator-of-l-1-norm)
```

```{code-cell} ipython3
:tags: [hide-cell]

# soft tresholding operation
def soft(y,alpha):
    return np.sign(y)*np.maximum(np.abs(y) - alpha,0)

# import libraries
import numpy as np
import matplotlib.pyplot as plt

# parameters
n = 128
k = 5
sigma = 0.05

# random seed
np.random.seed(1)

# generate spiky signal with random amplitudes
u = np.zeros(n)
u[np.random.randint(128,size=k)] = np.random.randn(k)

# generate noisy signal
f = u + sigma*np.random.randn(n)

# denoise
alpha = 0.1
uhat = soft(f,alpha)

# plot
plt.plot(u, label='ground truth')
plt.plot(uhat, label='denoised signal')
plt.legend()
plt.show()
```
