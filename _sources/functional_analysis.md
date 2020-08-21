# Functional analysis

Given a linear operator $K:\mathcal{U}\rightarrow\mathcal{V}$ we denote

* the domain of $K$: $\mathcal{D}(K) = \mathcal{U}$,
* the kernel or null-space of $K$: $\mathcal{N}(K) = \{u \in \mathcal{D}(K) \, | \, Ku = 0\}$,
* the range of $K$: $\mathcal{R}(K) = \{Ku\, | \, u \in \mathcal{D}(K)\}$,
* the operator norm of $K$: $\|K\| = \sup_{u\in\mathcal{U} \backslash \{0\}} \frac{\|Ku\|_{\mathcal{V}}}{\|u\|_{\mathcal{U}}}$.

We say that $K$ is continuous if for all $\epsilon > 0$ there exists a $\delta > 0$ such that

$$
\|Ku - Kv\|_{\mathcal{V}} \leq \epsilon, \quad \text{with} \quad \|u - v\|_{\mathcal{U}} \leq \delta.
$$
It can be verified that for linear operators, continuity is equivalent to boundedness ($\|K\| < \infty$).

If in addition, $\mathcal{U}$ and $\mathcal{V}$ are Hilbert spaces we have the following

* the adjoint operator $\adjoint{K}$ is (uniquely) defined by $\langle Ku, v \rangle_{\mathcal{V}} = \langle u, \adjoint{K}v \rangle_{\mathcal{U}}$
* orthogonal complement of a subspace $\mathcal{X} \subset \mathcal{U}$:
$\mathcal{X}^\perp = \{u\in\mathcal{U}\,|\, \langle u, v \rangle_{\mathcal{U}} = 0 \, \forall v \in \mathcal{X}\}$. If $\mathcal{X}$ is closed we have $\mathcal{U} = \mathcal{X} + \mathcal{X}^\perp$.
* orthogonal projection: let $\mathcal{X} \subset \mathcal{U}$ be a closed non-empty subspace, then the orthogonal projection onto $\mathcal{X}$ is denoted by $P_{\mathcal{X}}$ and obeys the following properties:
  * $\adjoint{P_{\mathcal{X}}} = P_{\mathcal{X}}$
  * $\|P_{\mathcal{X}}\| = 1$
  * $I - P_{\mathcal{X}} = P_{\mathcal{X}^{\perp}}$
* $\mathcal{R}(K)^\perp = \mathcal{N}(\adjoint{K})$, $\mathcal{R}(\adjoint{K})^\perp = \mathcal{N}(K)$, $\mathcal{N}(\adjoint{K})^\perp = \overline{\mathcal{R}(K)}$, $\mathcal{N}(K)^\perp = \overline{\mathcal{R}(\adjoint{K})}$
* orthogonal decomposition: $\mathcal{U} = \mathcal{N}(K) \oplus \overline{\mathcal{R}(\adjoint{K})}$ and $\mathcal{V} = \mathcal{N}(\adjoint{K}) \oplus \overline{\mathcal{R}(K)}$


An important class of operators are the \emph{compact operators}. An operator $K \in BL(\mathcal{U},\mathcal{V})$ in the space of bounded linear maps from Hilbert space $\mathcal{U}$ to $\mathcal{V}$ is finite dimensional if its range $\mathcal{R}(K)$ is a finite dimensional subspace of $\mathcal{V}$. $\mathcal{F}$ denotes the space of all finite dimensional operators.

An operator $K$  is compact (completely continuous) if

* the image of a bounded sequence $\{u_j\}$ contains a convergent subsequence $\{Ku_j\}$, or
* the closure of the image $\overline{K(B)}$ of any bounded $B$ set is compact, or
* if it is in the closure $\bar{\mathcal{F}}$ of $\mathcal{F}$, i.e.
$K = \lim_{n \rightarrow \infty} K_n$ with $K_n$ finite dimensional operators.

Hence $\mathcal{C} := \bar{\mathcal{F}}$ is the space of all compact (completely continuous) operators. Compact operators have many nice properties, but also a particularly nasty one; if $\mathcal{R}(K)$ is infinite-dimensional then the pseudo-inverse of $K$ is not continuous.
