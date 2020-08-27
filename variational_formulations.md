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

So far, we have seen that inverse problems may generally be formulated as a variational problem

```{math}
:label: variational
\min_{u\in\mathcal{U}} J(u),
```

where the *functional* $J : \mathcal{U} \rightarrow \mathbb{R}$ consists of a *data-fidelity* and *regularisation* term. In this chapter we will discuss how to analyse the well-posedness of {eq}`variational` and lay out the connection between variational problems and PDEs through the *gradient flow*. The contents of this chapter were heavily inspired by the excellent [lecture notes from Matthias J. Ehrhardt and Lukas F. Lang](https://mehrhardt.github.io/data/201803_lecture_notes_invprob.pdf)

---

We will broaden the setting a little and let $\mathcal{U}$ be a Banach space with some topology that is not necessarily induced by the underlying norm. We need this generality to be able to formally tackle some of the more fancy regularisation techniques. As before, we will not focus on the proofs too deeply and focus on the concepts. We will need the following concepts, however.

```{admonition} Definition: *dual space*
:class: important

With every Banach space $\mathcal{U}$ we can associate the dual space consisting of linear, continuous functionals on $\mathcal{U}$. For a given $v \in \mathcal{U}^*$ we denote the application of $v$ on $u$ as the dual product $\langle v,u\rangle$. As such, we can think of this as a way to generalise the concept of an inner product. However, the dual product is not generally symmetric.

The dual product also allows us to define the adjoint of a linear operator $K:\mathcal{U} \rightarrow \mathcal{F}$ as

$$\langle g, Ku\rangle = \langle K^*g, u\rangle \quad \forall u \in \mathcal{U}, g \in \mathcal{F}^*.$$
```

The main technical difficulties will arise when showing convergence of sequences in the usual way. For this, we need to introduce the notion of *weak convergence*.

```{admonition} Definition: *weak convergence*
:class: important

A sequence $\{u_k\}_{k\in\mathbb{N}} \subset \mathcal{U}$ is said to converge weakly to $u \in \mathcal{U}$ iff for all $v \in \mathcal{U}^*$ we have

$$\langle v, u_k\rangle \rightarrow \langle v,u\rangle.$$

We denote weak convergence by $u_k\rightharpoonup u$.
```

---

Some notable examples are highlighted below.

```{admonition} Example: *Sobolev regularisation*

Given a bounded linear operator $K:H^1(\Omega)\rightarrow L^2(\Omega)$ and data $f^\delta$, we let $\nabla$ denote the gradient

$$J(u) = \textstyle{\frac{1}{2}}\|Ku-f^{\delta}\|_{L^2(\Omega)}^2 +  \textstyle{\frac{\alpha}{2}}\|\nabla u\|_{L^2(\Omega)}^2.$$

This functional is well-defined for $u \in H^1(\Omega)$, with $H^1(\Omega)$ denotes the [Sobolev space](https://en.wikipedia.org/wiki/Sobolev_space) of functions $u$ for which both $u$ and $\nabla u$ are square integrable. Thus, thus regularisation generally leads to smooth solutions.
```

```{admonition} Example: *$\ell_1$-regularization*

Consider a forward operator $K:\ell_1 \rightarrow \ell_2$ and let

$$J(u) = \textstyle{\frac{1}{2}}\|Ku - f^\delta\|_{\ell_2}^2 + \alpha \|u\|_{\ell_1}.$$

Such regularisation term generally leads to *sparse* solutions.
```

```{admonition} Example: *Total Variation regularisation*

Consider recovering a function $u: [0,1] \rightarrow \mathbb{R}$ from noisy measurements $f^\delta = Ku + e$. A popular choice in imaging applications is to put an $L^1$-norm on the derivative. For $u\in W^{1,1}([0,1])$ this yields

$$J(u) = \textstyle{\frac{1}{2}}\|Ku-f^{\delta}\|_{L^2([0,1])}^2 + \alpha \|u'\|_{L^1([0,1])}.$$

This can be generalised to include certain non-smooth functions by introducing the space of functions of [bounded variation](https://en.wikipedia.org/wiki/Bounded_variation), denoted by $BV([0,1])$. Functions in $BV([0,1])$ are characterised as having a finite [Total Variation](https://en.wikipedia.org/wiki/Total_variation)

$$TV(u) = \sup_{\phi \in D([0,1],\mathbb{R})} \int_0^1 u(x)\phi'(x)\mathrm{d}x,$$

where $D([0,1],\mathbb{R})$ is the space of smooth test functions with $\|\phi\|_{L^\infty([0,1])}\leq 1$. This space is much larger than $H^{1,1}([0,1])$ as it contains certain discontinuous functions (such as the Heaveside stepfunction) and smaller than $L^1(0,1)$ (which also contains less regular functions). For functions in $H^{1,1}$ we have $TV(u) = \|u'\|_{L^1([0,1])}$.
```

## Analysis

### Well-posedness

To establish existence of minimisers, we first need a few definitions.

```{admonition} Definition: *Minimisers*
:class: important
We say that $\widetilde{u} \in \mathcal{U}$ solves {eq}`variational` iff $E(\widetilde{u}) < \infty$ and $E(\widetilde{u}) \leq E(u)$ for all $u \in \mathcal{U}$.
```

```{admonition} Definition: *Proper functionals*
:class: important

A functional $E$ is called proper if the effective domain is not empty.
```

```{admonition} Definition: *Bounded from below*
:class: important

A functional $J$ is bounded from below if there exists a constant $C > -\infty$ such that $\forall u\in \mathcal{U}$ we have $U(u) \geq C$.
```

```{admonition} Definition: *Coercive functionals*
:class: important

A functional $J$ is called coercive if for all $\{u_j\}_{j\in\mathbb{N}}$ with $\|u_j|_{\mathcal{U}}\rightarrow \infty$ we have $J(u_j) \rightarrow\infty$.
```

```{admonition} Definition: *Lower semi-continuity*
:class: important

A functional $J$ is called lower semi-continuous (with respect to a given topology) at $u\in\mathcal{U}$ if $J(u) \leq \lim\inf_{j\rightarrow \infty} J(u_j)$ for all sequences $\{u_j\}_{j\in\mathbb{N}}$ with $u_j\rightarrow u$ in the given topology of $\mathcal{U}$.
```

With these, we can establish existence.

```{admonition} Theorem: *Fundamental theorem of optimisation*
:class: important

Let $J : \mathcal{U} \rightarrow \mathbb{R}$ be a proper, coercive, bounded from below and lower semi-continuous. Then $J$ has a minimiser.
````

````{admonition} Examples: *existence of minimisers in $\mathbb{R}$*

Consider the following functions $J:\mathbb{R}\rightarrow \mathbb{R}$ (cf {numref}`functional`):

* $J_1(x) = x^3,$
* $J_2(x) = e^x,$
* $J_3(x) = \begin{cases}x^2 & x < 0 \\ 1 + x & x \geq 0\end{cases}$
* $J_4(x) = \begin{cases}x^2 & x \leq 0 \\ 1 + x & x > 0\end{cases}$

We see that $J_1$ is not bounded from below; $J_2$ is not coercive, $J_3$ is not l.s.c while $J_4$ is.

```{glue:figure} functionals
:figwidth: 600px
:name: "functionals"

Examples of various functionals.
```

````

```{code-cell}
:tags: ["hide-cell"]

import numpy as np
import matplotlib.pyplot as plt
from myst_nb import glue

# grid
x = np.linspace(0,5,1000)

# plot
fig,ax = plt.subplots(1,4)

ax[0].plot(-x,-x**3,'b',x,x**3,'b')
ax[0].set_xlim([-2,2])
ax[0].set_ylim([-2,2])
ax[0].set_aspect(1)

ax[1].plot(-x,np.exp(-x),'b',x,np.exp(x),'b')
ax[1].set_xlim([-2,2])
ax[1].set_ylim([-0.5,3.5])
ax[1].set_aspect(1)

ax[2].plot(-x[30:],x[30:]**2,'b',x,1+x,'b')
ax[2].plot(0,0,'bo',fillstyle='none')
ax[2].plot(0,1,'bo')
ax[2].set_xlim([-2,2])
ax[2].set_ylim([-0.5,3.5])
ax[2].set_aspect(1)

ax[3].plot(-x,x**2,'b',x[30:],1+x[30:],'b')
ax[3].plot(0,0,'bo')
ax[3].plot(0,1,'bo',fillstyle='none')
ax[3].set_xlim([-2,2])
ax[3].set_ylim([-0.5,3.5])
ax[3].set_aspect(1)

glue("functionals", fig, display=False)

```

```{admonition} Theorem: *Uniqueness of minimiser*
:class: important

Let $E$ have at least one minimiser and be [strictly convex](https://en.wikipedia.org/wiki/Convex_function) then the minimiser is unique.
```


## The Euler-Lagrange equations

### Derivatives

```{admonition} Definition: *Fréchet derivative*
:class: important

We call a functional $E:\mathcal{U}\rightarrow \mathbb{R}$ Fréchet differentiable (at $u$) if
there exists a linear operator $D$ such that

$$\lim_{h\rightarrow 0} \frac{|E(u+h) - E(u) - Dh|}{\|h\|_{\mathcal{U}}} = 0.$$

If this operator exists for all $u \in\mathcal{U}$ we call $E$ Fréchet differentiable and denote its Fréchet derivative by $E': \mathcal{U} \rightarrow \mathcal{U}^*$.
```

### From functionals to PDEs
* heat equation
* Total Variation
* Perona-Malik

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
