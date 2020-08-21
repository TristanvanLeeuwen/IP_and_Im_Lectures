# Fourier transform, distributions and sampling
We use the folling conventions for the Fourier transforms in space and time. The temporal Fourier transform:
\begin{equation}
\widehat{v}(\omega) = F_tv = \frac{1}{\sqrt{2\pi}}\int v(t)\exp(\imath\omega t)\mathrm{d}t,
\end{equation}
\begin{equation}
v(t) = F_t^{-1}\widehat{v} = \frac{1}{\sqrt{2\pi}}\int \widehat{v}(\omega)\exp(-\imath\omega t)\mathrm{d}\omega
\end{equation}
The spatial Fourier transform
\begin{equation}
\widehat{v}(\xi) = F_xv = \frac{1}{(2\pi)^{n/2}}\int v(x)\exp(\imath\xi\cdot x)\mathrm{d}x,
\end{equation}
\begin{equation}
v(x) = F_x^{-1}\widehat{v}= \frac{1}{(2\pi)^{n/2}}\int \widehat{v}(\xi)\exp(-\imath\xi\cdot x)\mathrm{d}\xi,
\end{equation}
Parseval's theorem ensures that the Fourier transform of a square integrable is also square integrable:
\begin{equation}
\int |v(t)|^2\mathrm{d}t = \int |\widehat{v}(\omega)|^2\mathrm{d}\omega.
\end{equation}
The Fourier transform is defined for distributions as well, in particular
\begin{equation}
\delta(t) = \frac{1}{\sqrt{2\pi}}\int \exp(-\imath\omega t)\mathrm{d}\omega.
\end{equation}

Some other usefull relations:

differentation
: \begin{equation*}
F_t v'(\omega) = (\imath\omega)F_tv(\omega),
\end{equation*}

translation
: \begin{equation*}
F_t v(\cdot + \tau)(\omega) = e^{\imath\omega\tau}F_tv(\omega),
\end{equation*}

convolution
: \begin{equation*}
F_t (u\star v)(\omega) = \left(F_tu\right)\left(F_tv\right)(\omega).
\end{equation*}

correlation
: \begin{equation*}
F_t (u * v)(\omega) = \overline{\left(F_tu\right)}\left(F_tv\right)(\omega),
\end{equation*}
where  $\overline{\cdot}$ denotes complex conjugation.

Finally, recall that we may uniquely reconstruct a *bandlimited* function $v(t)$ from samples $v(n\Delta t)$, with $\Delta t = (2B)^{-1}$ and $B$ is the bandlimit of $v$, i.e., $\widehat{f}(\omega) = 0$ if $|\omega| > B$. Conversely, we need only a discete set of samples of $\widehat{v}$ if $v$ is compactly supported. Remember that a function that is compactly supported is not bandlimited and vice versa.
