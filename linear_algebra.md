# Linear algebra

Invertible matrices
: A square matrix $A \in \mathbb{R}^{n\times n}$ is invertible if it has rank $n$, or equivalently, if $\text{det}(A) \not=0$. We have the following useful identities
\begin{equation}
(AB)^{-1} = B^{-1}A^{-1}.
\end{equation}
\begin{equation}
(A + B)^{-1} = A^{-1} - A^{-1}(I + BA^{-1})^{-1}BA^{-1} = A^{-1} - (A + AB^{-1}A)^{-1}.
\end{equation}
\begin{equation}
(I + A)^{-1} = I - A + A^2 - A^3 + \ldots (\text{when}\quad \|A\|\leq 1).
\end{equation}

Eigenvalue decomposition
: A matrix $A \in \mathbb{R}^{n\times n}$ is called *diagonalizable* if there exists an invertible matrix $U$ such that
\begin{equation}
A = U\Lambda U^{-1},
\end{equation}
where $U = [u_1, u_2, \ldots, u_n]$ consists of the eigenvectors of $A$ and $\Lambda$ is a diagonal matrix containing the eigenvalues $\lambda_i$ of $A$. If $A$ is symmetric, the eigenvectors are orthogonal and we can normalize them such that $U^*U = I$. The matrix $A$ is invertible if all eigenvalues are non-zero, in which case we have
\begin{equation}
A^{-1} = U\Lambda^{-1} U^{-1}.
\end{equation}
Not all matrices are diagonalizable, however, in which case the matrix is called *defective*.

Singular value decomposition
: All matrices $A \in \mathbb{R}^{m\times n}$ have a *singular value decomposition*
\begin{equation}
A = U\Sigma {V}^*,
\end{equation}
where $U \in \mathbb{R}^{m\times m}$ and $V \in \mathbb{R}^{n\times n}$ consist of the left and right singular vectors and $\Sigma \in \mathbb{R}^{m\times n}$ is a diagonal matrix containing the singular values $\sigma_i$.
In particular we have $U^*U = I$, $V^*V = I$ and $\sigma_i \geq 0$. If the matrix has rank $k$, the first $k$ singular values are non-zero while the remaining ones are zero. The first $k$ left and right singular vectors form an orthonormal basis for the column space and row space of $A$. The matrix $\Sigma$ is structured as follows
\begin{equation}
\Sigma =
\left(\begin{matrix}
\Sigma_{k} 				& 0_{k \times (n - k)}     \\
0_{(m - k) \times k}    & 0_{(m - k) \times (n-k)} \\
\end{matrix}\right),
\end{equation}
where $k = \min(m,n)$ and $\Sigma_k$ is a diagonal matrix containing the singular values $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_k \geq 0$ and $0_{p\times q}$ denotes an $p\times q$ matrix with all zeros. In particular we have $U^*U = I$, $V^*V = I$.
The singular value decomposition is related to the eigendecompositions of $A^*A$ and $AA^*$ as follows
\begin{equation}
A^* A = V\Sigma^*\Sigma V^*,
\end{equation}
\begin{equation}
AA^* = U\Sigma{\Sigma}^* U^*.
\end{equation}

Pseudo-inverse
: Using the singular value decomposition of $K$ we may attempt to express its inverse as
\begin{equation}
K^{-1} = V\Sigma^{-1}{U}^*.
\end{equation}
A problem arises when some of the singular values are zero or when $\Sigma$ is not square. For this we introduce the \emph{pseudo inverse}, which is defined as
\begin{equation}
K^\dagger = V_k\Sigma_k^{-1}{U_k}^* ,
\end{equation}
where $k$ is the rank of the matrix, $V_k \in \mathbb{R}^{n\times k}$ contains the first $k$ right singular vectors, $U_k \in \mathbb{R}^{m \times k}$ contains the first $k$ right singular vectors and $\Sigma_k \in \mathbb{R}^{k\times k}$ contains the non-zero singular values.
When $m > n$ and $K$ has rank $n$ we have
\begin{equation}
K^{\dagger} = (K^* K)^{-1}K^* ,
\end{equation}
in which case $K^\dagger K = I$ (it is a left inverse).
When $m < n$ and $K$ has rank $m$ we have
\begin{equation}
K^{\dagger} = K(K{K}^* )^{-1},
\end{equation}
in which case $KK^\dagger = I$ (it is a right inverse).

Norms
: Let $x \in \mathbb{R}^n$ and $A \in \mathbb{R}^{m \times n}$.
The usual vector $p$-norm is defined as
\begin{equation}
\|x\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}.
\end{equation}
Often-used ones are $p=1$ (Manhattan distance), $p=2$ (Euclidian norm) and $p=\infty$ (supremum norm). The norms obey the following inequalities
\begin{equation}
\|x\|_{\infty} \leq \|x\|_2 \leq \sqrt{n} \|x\|_{\infty},
\end{equation}
\begin{equation}
\|x\|_{2} \leq \|x\|_1 \leq \sqrt{n} \|x\|_{2},
\end{equation}
\begin{equation}
\|x\|_{\infty} \leq \|x\|_1 \leq n \|x\|_{\infty},
\end{equation}
Based on these we can define the *induced matrix norm*
\begin{equation}
\|A\|_p = \max_{x} \frac{\|Ax\|_p}{\|x\|_p}.
\end{equation}
We have
\begin{equation}
\|A\|_1 = \max_j \sum_{i=1}^m |a_{ij}|,
\end{equation}
\begin{equation}
\|A\|_2 = \sigma_1,
\end{equation}
and
\begin{equation}
\|A\|_\infty = \max_i \sum_{j=1}^n |a_{ij}|.
\end{equation}
Besides having the usual properties of a norm, we have the following useful inequality
\begin{equation}
\|AB\|_p \leq \|A\|_p \|B\|_p,
\end{equation}
Two other matrix norms worth mentioning are the *Frobenius norm*
\begin{equation}
\|A\|_F = \sqrt{\sum_{ij} |a_{ij}|^2} = \sqrt{\sum_{i} \sigma_i^2},
\end{equation}
and the *nuclear norm*
\begin{equation}
\|A\|_* = \sum_{i=1}^k \sigma_i.
\end{equation}

Matrix functions
: We can define a matrix function of a diagonalizable matrix as
\begin{equation}
f(A) = Uf(\Lambda)U^{-1},
\end{equation}
where $f(\Lambda)$ is a diagonal matrix with entries $f(\lambda_i)$.
