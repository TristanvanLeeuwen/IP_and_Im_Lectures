# Linear algebra

## Vector spaces

We consider $\mathbb{C}^n$ as the vector space of tuples $u = (u_1, u_2, \ldots, u_n)$. Linear combinations are defined elementwise in the usual way.

```{admonition} Vector norms

Let $u \in \mathbb{C}^n$. The vector $p$-norm is defined as

$$
\|u\|_p = \left(\sum_{i=1}^n |u_i|^p\right)^{1/p}.
$$

Often-used ones are $p=1$ (Manhattan distance), $p=2$ (Euclidian norm) and $p=\infty$ (supremum norm). The norms are all *equivalent* and obey the following inequalities

* $\|u\|_{\infty} \leq \|u\|_2 \leq \sqrt{n} \|u\|_{\infty},$
* $\|u\|_{2} \leq \|u\|_1 \leq \sqrt{n} \|u\|_{2},$
* $\|u\|_{\infty} \leq \|u\|_1 \leq n \|u\|_{\infty},$
```

```{admonition} Inner product
Let $u, v \in \mathbb{C}^n$. The inner product is defined as

$$\langle u, v\rangle = \sum_{i=1}^n u_i \overline{v_i},$$

with $\overline{\cdot}$ denoting the complex conjugate. The inner product is sometimes denoted by $u^*v$ with $\cdot^*$ denoting the complex-conjugate-transpose.
```

## Matrices

A linear operator $A: \mathbb{C}^n \rightarrow \mathbb{C}^m$ can be represented as a matrix with elements $a_{ij}$, $i \in \{1, 2, \ldots, m\}$, $j \in \{1, 2, \ldots, n\}$.

```{admonition} Invertible matrices

A square matrix $A \in \mathbb{C}^{n\times n}$ is invertible if it has rank $n$, or equivalently, if $\text{det}(A) \not=0$. We have the following useful identities

* $(AB)^{-1} = B^{-1}A^{-1}.$
* $(A + B)^{-1} = A^{-1} - A^{-1}(I + BA^{-1})^{-1}BA^{-1} = A^{-1} - (A + AB^{-1}A)^{-1}.$
* $(I + A)^{-1} = I - A + A^2 - A^3 + \ldots (\text{when}\quad \|A\|\leq 1).$
```

```{admonition} Eigenvalue decomposition
A matrix $A \in \mathbb{C}^{n\times n}$ is called *diagonalizable* if there exists an invertible matrix $U \in \mathbb{C}^{n\times n}$ such that

$$
A = U\Lambda U^{-1},
$$

where $U = [u_1, u_2, \ldots, u_n]$ consists of the eigenvectors of $A$ and $\Lambda$ is a diagonal matrix containing the eigenvalues $\lambda_i$ of $A$. If $A$ is Hermitian (i.e., $a_{ij} = \overline{a_{ji}}$), the eigenvectors are orthogonal and we can normalize them such that $U^*U = I$. The matrix $A$ is invertible if all eigenvalues are non-zero, in which case we have

$$
A^{-1} = U\Lambda^{-1} U^{-1}.
$$

Not all matrices are diagonalizable, however, in which case the matrix is called *defective*.
```

```{admonition} Singular value decomposition
All matrices $A \in \mathbb{C}^{m\times n}$ have a *singular value decomposition*

$$
A = U\Sigma {V}^*,
$$

where $U \in \mathbb{C}^{m\times m}$ and $V \in \mathbb{C}^{n\times n}$ consist of the left and right singular vectors and $\Sigma \in \mathbb{C}^{m\times n}$ is a diagonal matrix containing the singular values $\sigma_i$.
In particular we have $U^*U = I$, $V^*V = I$ and $\sigma_i \geq 0$. If the matrix has rank $k$, the first $k$ singular values are non-zero while the remaining ones are zero. The first $k$ left and right singular vectors form an orthonormal basis for the column space and row space of $A$. The matrix $\Sigma$ is structured as follows

$$
\Sigma =
\left(\begin{matrix}
\Sigma_{k} 				& 0_{k \times (n - k)}     \\
0_{(m - k) \times k}    & 0_{(m - k) \times (n-k)} \\
\end{matrix}\right),
$$

where $k = \min(m,n)$ and $\Sigma_k$ is a diagonal matrix containing the singular values $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_k \geq 0$ and $0_{p\times q}$ denotes an $p\times q$ matrix with all zeros. In particular we have $U^*U = I$, $V^*V = I$.
The singular value decomposition is related to the eigendecompositions of $A^*A$ and $AA^*$ as follows

$$
A^* A = V\Sigma^*\Sigma V^*,
$$

$$
AA^* = U\Sigma{\Sigma}^* U^*.
$$
```

```{admonition} Pseudo-inverse
The *pseudo inverse* of a matrix $A\in\mathbb{C}^{m\times n}$ is defined in terms of its SVD as

$$
A^\dagger = V_k\Sigma_k^{-1}{U_k}^* ,
$$

where $k$ is the rank of the matrix, $V_k \in \mathbb{C}^{n\times k}$ contains the first $k$ right singular vectors, $U_k \in \mathbb{C}^{m \times k}$ contains the first $k$ right singular vectors and $\Sigma_k \in \mathbb{C}^{k\times k}$ contains the non-zero singular values.
When $m > n$ and $A$ has rank $n$ we have

$$
A^{\dagger} = (A^* A)^{-1}A^* ,
$$

in which case $A^\dagger A = I$ (it is a left inverse).
When $m < n$ and $A$ has rank $m$ we have

$$
A^{\dagger} = A(A{A}^* )^{-1},
$$

in which case $AA^\dagger = I$ (it is a right inverse).
```

```{admonition} Matrix functions
We can define a matrix function of a diagonalizable matrix as

$$
f(A) = Uf(\Lambda)U^{-1},
$$

where $A = U\Lambda U^{-1}$ and $f(\Lambda)$ is a diagonal matrix with entries $f(\lambda_i)$.
```

```{admonition} Matrix norms

Based on these we can define the *induced matrix norm*

$$
\|A\|_p = \max_{x} \frac{\|Ax\|_p}{\|x\|_p}.
$$

In particular, we have

* $\|A\|_1 = \max_j \sum_{i=1}^m |a_{ij}|,$
* $\|A\|_2 = \sigma_1,$
* $\|A\|_\infty = \max_i \sum_{j=1}^n |a_{ij}|.$

Besides having the usual properties of a norm, we have the following useful inequality

$$
\|AB\|_p \leq \|A\|_p \|B\|_p,
$$

Two other matrix norms worth mentioning are the *Frobenius norm*

$$
\|A\|_F = \sqrt{\sum_{ij} |a_{ij}|^2} = \sqrt{\sum_{i} \sigma_i^2},
$$

and the *nuclear norm*

$$
\|A\|_* = \sum_{i=1}^k \sigma_i.
$$
```
