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

where the *functional* $J : \mathcal{U} \rightarrow \mathbb{R}_{\infty}$ consists of a *data-fidelity* and *regularisation* term. Here, $\mathbb{R}_{\infty} = \mathbb{R} \cup \{\infty\}$ denotes the extended real line and $\mathcal{U}$ is a Banach space.

In this chapter we will discuss how to analyse the well-posedness of {eq}`variational` and lay out the connection between variational problems and PDEs through the *gradient flow*. The contents of this chapter were heavily inspired by the excellent [lecture notes from Matthias J. Ehrhardt and Lukas F. Lang](https://mehrhardt.github.io/data/201803_lecture_notes_invprob.pdf)

---

Some notable examples are highlighted below.

```{admonition} Example: *box constraints*

Given a forward operator $K \in \mathbb{R}^{n\times n}$ we can look for a solution in $[0,1]^n$ by solving a constrained minimisation problem

$$\min_{u\in [0,1]^n} \|Ku - f^\delta\|_2^2.$$

However, this does not fall in the class {eq}`variational` since $[0,1]^n$ is not a vectorspace. To circumvent this we can alternatively express it as

$$\min_{u\in \mathbb{R}^n} \|Ku - f^\delta\|_2^2 + \delta_{[0,1]^n}(u),$$

where $\delta_{\mathcal{C}}$ denotes the [characteristic function](https://en.wikipedia.org/wiki/Characteristic_function_(convex_analysis)) of the set $\mathcal{C}$:

$$\delta_{\mathcal{C}}(u) = \begin{cases} 0 & u \in \mathcal{C} \\ \infty & \text{otherwise}\end{cases}.$$

The corresponding functional $J$ now takes values in the extended real line.
```

```{admonition} Example: *Sobolev regularisation*

Given a bounded linear operator $K:H^1(\Omega)\rightarrow L^2(\Omega)$ and data $f^\delta$, we let $\nabla$ denote the gradient

$$J(u) = \textstyle{\frac{1}{2}}\|Ku-f^{\delta}\|_{L^2(\Omega)}^2 +  \textstyle{\frac{\alpha}{2}}\|\nabla u\|_{L^2(\Omega)}^2.$$

This functional is well-defined for $u \in H^1(\Omega)$, with $H^1(\Omega)$ denoting the [Sobolev space](https://en.wikipedia.org/wiki/Sobolev_space) of functions $u$ for which both $u$ and $\nabla u$ are square integrable. Thus, thus regularisation generally leads to smooth solutions.
```

```{admonition} Example: *$\ell_1$-regularization*

Consider a forward operator $K:\ell_1 \rightarrow \ell_2$ and let

$$J(u) = \textstyle{\frac{1}{2}}\|Ku - f^\delta\|_{\ell_2}^2 + \alpha \|u\|_{\ell_1}.$$

Such regularisation is often used to promote *sparse* solutions.
```

```{admonition} Example: *Total Variation regularisation*

Consider recovering a function $u: [0,1] \rightarrow \mathbb{R}$ from noisy measurements $f^\delta = Ku + e$. A popular choice in imaging applications is to put an $L^1$-norm on the derivative. For $u\in W^{1,1}([0,1])$ this yields

$$J(u) = \textstyle{\frac{1}{2}}\|Ku-f^{\delta}\|_{L^2([0,1])}^2 + \alpha \|u'\|_{L^1([0,1])}.$$

This can be generalised to include certain non-smooth functions by introducing the space of functions of [bounded variation](https://en.wikipedia.org/wiki/Bounded_variation), denoted by $BV([0,1])$. Functions in $BV([0,1])$ are characterised as having a finite [Total Variation](https://en.wikipedia.org/wiki/Total_variation)

$$TV(u) = \sup_{\phi \in D([0,1],\mathbb{R})} \int_0^1 u(x)\phi'(x)\mathrm{d}x,$$

where $D([0,1],\mathbb{R})$ is the space of smooth test functions with $\|\phi\|_{L^\infty([0,1])}\leq 1$. This space is much larger than $H^{1,1}([0,1])$ as it contains certain discontinuous functions (such as the Heaveside stepfunction) and smaller than $L^1(0,1)$ (which also contains less regular functions). For functions in $H^{1,1}$ we have $TV(u) = \|u'\|_{L^1([0,1])}$.
```

## Analysis

### Existence and uniqueness

To establish existence of minimisers, we first need a few definitions.

```{admonition} Definition: *Minimisers*
:class: important
We say that $\widetilde{u} \in \mathcal{U}$ solves {eq}`variational` iff $J(\widetilde{u}) < \infty$ and $J(\widetilde{u}) \leq J(u)$ for all $u \in \mathcal{U}$.
```

```{admonition} Definition: *Proper functionals*
:class: important

A functional $J$ is called proper if its effective domain $\text{dom}(J) = \{u\in\mathcal{U} \, | \, J(u) < \infty\}$ is not empty.
```

```{admonition} Definition: *Bounded from below*
:class: important

A functional $J$ is bounded from below if there exists a constant $C > -\infty$ such that $\forall u\in \mathcal{U}$ we have $J(u) \geq C$.
```

```{admonition} Definition: *Coercive functionals*
:class: important

A functional $J$ is called coercive if for all $\{u_j\}_{j\in\mathbb{N}}$ with $\|u_j\|_{\mathcal{U}}\rightarrow \infty$ we have $J(u_j) \rightarrow\infty$.
```

```{admonition} Definition: *Lower semi-continuity*
:class: important

A functional $J$ is lower semi-continuous at $u$ if for every $a < J(u)$ there exists a neighbourhood $\mathcal{X}$ of $u$ such that $a < J(v)$ for all $v \in \mathcal{X}$.

Note that the term *neighbourhood* implies an underlying topology, which may be different (in particular, weaker) than the one induced by the norm on $\mathcal{U}$.
```

With these, we can establish existence.

```{admonition} Theorem: *Fundamental theorem of optimisation*
:class: important

Let $J : \mathcal{U} \rightarrow \mathbb{R}$ be proper, coercive, bounded from below and lower semi-continuous. Then $J$ has a minimiser.
````

````{admonition} Examples: *existence of minimisers in $\mathbb{R}$*

Consider the following functions $J:\mathbb{R}\rightarrow \mathbb{R}$ (cf. {numref}`functionals`):

* $J_1(x) = x^3,$
* $J_2(x) = e^x,$
* $J_3(x) = \begin{cases}x^2 & x < 0 \\ 1 + x & x \geq 0\end{cases}$
* $J_4(x) = \begin{cases}x^2 & x \leq 0 \\ 1 + x & x > 0\end{cases}$

We see that $J_1$ is not bounded from below; $J_2$ is not coercive, $J_3$ is not l.s.c while $J_4$ is.

```{glue:figure} functionals
:figwidth: 600px
:name: "functionals"

Examples of various functions.
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

```{admonition} Theorem: *Uniqueness of minimisers*
:class: important

Let $J$ have at least one minimiser and be [strictly convex](https://en.wikipedia.org/wiki/Convex_function) then the minimiser is unique.
```

### Well-posedness of regularised least-squares problems

In this section we focus in particular on variational problems of the form

```{math}
:label: variational_R
\textstyle{\frac{1}{2}}\|Ku - f^\delta\|_{\mathcal{F}}^2 + \alpha R(u),
```

with $K: \mathcal{U} \rightarrow \mathcal{F}$ a bounded linear operator and $R : \mathcal{U} \rightarrow \mathbb{R}_{\infty}$ is proper and l.s.c. (with respect to an appropriate topology).

We can think of this as defining a (possibly non-linear) regularisation scheme $\widetilde{u}_{\alpha,\delta} = K_{\alpha}^\dagger(f^\delta)$ that generalises the pseudo-inverse approach discussed earlier. Note that the notation $K_{\alpha}^\dagger$ is used very loosely to indicate a mapping from $\mathcal{F}$ to $\mathcal{U}$ that is supposed to approximate the inverse of $K$ in some fashion. In general, this will be a non-linear mapping.

```{admonition} Theorem: *Existence and uniqueness of regularised least-squares solutions*

Let $K$ be injective or $J$ be strictly convex, then the variational problem {eq}`variational_R` has a unique minimiser.

```

```{admonition} Theorem: *Stability of regularised least-squares solutions*

...

```

### Examples


```{admonition} Example: *Tikhonov regularisation in $\mathbb{R}^n$*

Let

$$J(u) = \textstyle{\frac{1}{2}}\|Ku - f^\delta\|_2^2 + \textstyle{\frac{\alpha}{2}}\|u\|_2^2.$$

Here, $J$ is obviously bounded from below and proper. To show that $J$ is coercive, we note that $J(u) \geq \textstyle{\frac{\alpha}{2}}\|u\|_2^2$ and hence that $J(u) \rightarrow \infty$ as $\|u\|_2 \rightarrow \infty$. To show that $J$ is l.s.c., we will show that $J$ is continuous since this implies l.s.c. First note that

$$J(u + d) = J(u) + \textstyle{\frac{1}{2}}\|Kd\|_2^2 - \langle Kd,Ku - f^\delta \rangle + \textstyle{\frac{\alpha}{2}}\|d\|_2^2 + \alpha \langle d,u\rangle,$$

from which we can bound

$$|J(v) - J(u)| \leq \textstyle{\frac{1}{2}}\|Kd\|_2^2 + \|Kd\|_2 \|Ku - f^\delta\|_2 + \alpha \|d\|_2 \|u\|_2 + \textstyle{\frac{\alpha}{2}}\|d\|_2^2 \leq A \|d\|_2^2 + B \|d\|_2.$$

Now, for every $\epsilon$ we can pick a $\delta$ such that $\|u-v\|_2 < \delta$ implies that $|J(v) - J(u)| < \epsilon$.

Finally, we can show that $J$ is *strongly convex* with constant $\alpha$ by showing that $J(u) - \textstyle{\frac{\alpha}{2}}\|u\|_2^2$ is convex. The fact that $\|Ku - f^\delta\|_2^2$ is convex follows easily from the triangle inequality and the fact that the function $\cdot^2$ is convex.

```

```{admonition} Example: *$\ell_1-regularisation$*

Consider

$$J(u) = \textstyle{\frac{1}{2}}\|Ku - f^\delta\|_{\ell_2}^2 + \alpha \|u\|_{\ell_1},$$

with $K : \ell_2 \rightarrow \ell_2$ a bounded operator. Again, we can easily see that $J$ is bounded from below.

Note that the regularised solution is determined for all $f \in \ell_2$, regardless of the Picard condition.

```

```{admonition} Example: *Sobolev regularisation*
Consider

$$J(u) = \textstyle{\frac{1}{2}}\|Ku - f^\delta\|_{L^2(\Omega)}^2 + \textstyle{\frac{\alpha}{2}}\|\nabla u\|_{L^2(\Omega)}^2.$$
```

```{admonition} Example: *Total variation regularisation*

Let

$$J(u) = \textstyle{\frac{1}{2}}\|Ku - f^\delta\|_{L^2(\Omega)}^2 + \alpha TV(u).$$

* existence, uniqueness, stability

```


## Derivatives

Having established well-posedness of {eq}`variational`, we now focus our attention to characterising solutions through the first-order optimality conditions.

````{admonition} Definition: *Fréchet derivative*
:class: important

We call a functional $J:\mathcal{U}\rightarrow \mathbb{R}$ Fréchet differentiable (at $u$) if
there exists a linear operator $D$ such that


```{math}
:label: frechet
\lim_{h\rightarrow 0} \frac{|J(u+h) - J(u) - Dh|}{\|h\|_{\mathcal{U}}} = 0.
```

If this operator exists for all $u \in\mathcal{U}$ we call $J$ Fréchet differentiable and denote its Fréchet derivative by $J': \mathcal{U} \rightarrow \mathcal{U}^*$. Here, $\mathcal{U}^*$ denotes the [dual space](https://en.wikipedia.org/wiki/Dual_space) of $\mathcal{U}$ which consists of bounded linear functionals on $\mathcal{U}$.
````

With this more general notion of differentiation we can pose the first-order optimality conditions.

````{admonition} Definition: *First-order optimality conditions*
:class: important

```{math}
:label: local_minimum

\langle J'(u), v - u\rangle \geq 0.
```

````

```{admonition} Example: *Tikhonov regularisation on \mathbb{R}^n*

```

We need to be careful here, as some important cases $J$ may fail to be Fréchet differentiable at the solution.

```{admonition} Example: *$\ell_1$-regularisation on \mathbb{R}^n*

```

---

A well-known method for solving {eq}`variational` is the *Landweber* iteration

$$u_{k+1} = u_k - \lambda J'(u_k),$$

where $\lambda > 0$ denotes the stepsize. Under certain conditions on $J'(u)$ and $\lambda$ one can show that this converges to a stationary point $u_{*}$ for which $J'(u_*) = 0$. Obviously, this method only applies when $J'(u)$ is well-defined everywhere along the solution path.

---

It turns out that we can make an important distinction between *smooth* and *convex* (non-smooth) functionals.
We will explore optimality conditions and algorithms for these two classes in more detail in a later chapter.

## The Euler-Lagrange equations

An alternative viewpoint on optimality is provided by the Euler-lagrange equations, which establishes the link between certain problems of the form {eq}`variational` and PDEs. In particular, we focus in this section on problems of the form

$$\min_{u\in\mathcal{U}} \textstyle{\frac{1}{2}} \|u - f^\delta\|_{L^2(\Omega)} + \alpha R(\nabla u).$$

Such problems occur for example in image-denoising applications. We will see later that such problems also occur as subproblems when solving more general problems of the form {eq}`variational_R`.

```{admonition} Definition: Euler-Lagrange equations
:class: important

The first-order optimality condition for $u\in\mathcal{U}$ to be a solution to {eq}`variational` is

$$\left.\frac{\mathrm{d}}{\mathrm{d}t} J(u + t\phi)\right|_{t=0} = 0 \quad \forall \phi \in C_c^{\infty},$$

where $..$
```

````{admonition} Example: *The heat equation*

Let

$$R(u) = \|\nabla u\|_{L^2(\Omega)}^2.$$

The corresponding diffusion equation is given by

$$\partial_t u + u - \alpha\nabla^2 u = f^\delta.$$

* definition of underlying spaces, boundary conditions.

A forward Euler discretisation of the PDE leads to

$$u_{k+1} = u_k - \Delta t \left(f^\delta - u + \alpha \nabla^2 u\right),$$

which is in fact a Landweber iteration applied to the corresponding objective.

```{glue:figure} linear_diffusion
:figwidth: 600px
:name: "linear_diffusion"

Example of denoising with linear diffusion.
```

````

```{code-cell}
:tags: ["hide-cell"]

import numpy as np
import matplotlib.pyplot as plt
from myst_nb import glue

from skimage import data
from skimage.util import random_noise
from skimage.transform import resize

# parameters
sigma = 0.1
alpha = 1
dt = 1e-6
niter = 1001
n = 200
coeff = lambda s : 1 + 0*s

# diffusion operator
def L(u,coeff = lambda s : 1 + 0*s):
    ue = np.pad(u,1,mode='edge') # padd edges to get array of size n+2 x n+2

    # diffusion coefficient (central differences)
    grad_norm = ((ue[2:,1:-1] - ue[:-2,1:-1])/(2/n))**2 + ((ue[1:-1,2:] - ue[1:-1,:-2])/(2/n))**2
    c = np.pad(coeff(grad_norm),1,mode='edge')

    # diffusion term (combination of forward and backward differences)
    uxx = ((c[1:-1,1:-1] + c[2:,1:-1])*(ue[2:,1:-1]-ue[1:-1,1:-1]) - (c[:-2,1:-1]+c[1:-1,1:-1])*(ue[1:-1,1:-1]-ue[:-2,1:-1]))/(2/n**2)
    uyy = ((c[1:-1,1:-1] + c[1:-1,2:])*(ue[1:-1,2:]-ue[1:-1,1:-1]) - (c[1:-1,:-2]+c[1:-1,1:-1])*(ue[1:-1,1:-1]-ue[1:-1,:-2,]))/(2/n**2)

    return uxx + uyy

# noisy image
f = resize(data.camera(),(n,n))
f_delta = random_noise(f,var=sigma**2)

# solve evolution equation
u = np.zeros((n,n))

for k in range(niter-1):
    u = u - dt*(u - alpha*L(u,coeff)) + dt*f_delta

# plot
fig,ax = plt.subplots(1,2)

ax[0].imshow(f_delta)
ax[0].set_title('Noisy image')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[1].imshow(u)
ax[1].set_title('Result')
ax[1].set_xticks([])
ax[1].set_yticks([])

glue("linear_diffusion", fig, display=False)

```

````{admonition} Example: *Non-linear diffusion*

Let

$$R(u) = \int_{\Omega} r\left(\|\nabla u(x)\|^2\right) \mathrm{d}x.$$

A popular choice for $r = \log(1 + s/\epsilon^2)$, which leads to the Perona-Malik diffusion equation:

$$\partial_t u + u - \alpha\nabla \cdot \left(\frac{\nabla u}{1 + \epsilon^{-2}\|\nabla u\|_2^2}\right) = f^\delta.$$

We can interpret intuitively why this would preserve edges by looking at the diffusion coefficient. Wherever $\|\nabla u\| \ll \epsilon$ we have linear diffusion, if $\|\nabla u\| \gg \epsilon$, we hardly have any diffusion. This intuition if confirmed by consider the penalty $r(s)$, which for small $s$ behaves like $s^2$ but then flattens out and will thus not increasingly penalise larger gradients.

```{glue:figure} perona_malik
:figwidth: 600px
:name: "perona_malik"

Example of denoising with Perona-Malik regularisation.
```
````

```{code-cell}
:tags: ["hide-cell"]

import numpy as np
import matplotlib.pyplot as plt
from myst_nb import glue

from skimage import data
from skimage.util import random_noise
from skimage.transform import resize

# parameters
sigma = 0.1
alpha = 1
dt = 1e-6
niter = 1001
n = 200
coeff = lambda s : 1/(1+1e6*s)

# diffusion operator
def L(u,coeff = lambda s : 1):
    ue = np.pad(u,1,mode='edge') # padd edges to get array of size n+2 x n+2

    # diffusion coefficient (central differences)
    grad_norm = ((ue[2:,1:-1] - ue[:-2,1:-1])/(2/n))**2 + ((ue[1:-1,2:] - ue[1:-1,:-2])/(2/n))**2
    c = np.pad(coeff(grad_norm),1,mode='edge')

    # diffusion term (combination of forward and backward differences)
    uxx = ((c[1:-1,1:-1] + c[2:,1:-1])*(ue[2:,1:-1]-ue[1:-1,1:-1]) - (c[:-2,1:-1]+c[1:-1,1:-1])*(ue[1:-1,1:-1]-ue[:-2,1:-1]))/(2/n**2)
    uyy = ((c[1:-1,1:-1] + c[1:-1,2:])*(ue[1:-1,2:]-ue[1:-1,1:-1]) - (c[1:-1,:-2]+c[1:-1,1:-1])*(ue[1:-1,1:-1]-ue[1:-1,:-2,]))/(2/n**2)

    return uxx + uyy

# noisy image
f = resize(data.camera(),(n,n))
f_delta = random_noise(f,var=sigma**2)

# solve evolution equation
u = np.zeros((n,n))

for k in range(niter-1):
    u = u - dt*(u - alpha*L(u,coeff)) + dt*f_delta

# plot
fig,ax = plt.subplots(1,2)

ax[0].imshow(f_delta)
ax[0].set_title('Noisy image')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[1].imshow(u)
ax[1].set_title('Result')
ax[1].set_xticks([])
ax[1].set_yticks([])
glue("perona_malik", fig, display=False)

```

```{admonition} Example: *Total variation*

Let

$$R(u) = ...$$

```

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
