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

# Magnetic Resonance Imaging

## A very brief history of MRI

-   1946: Felix Bloch and Edward Purcell independently discover the
    magnetic resonance phenomenon (Nobel Prize in 1952)

-   1971: Raymond Damadian observes different nuclear magnetic
    relaxation times between healthy tissue and tumor. Clinical
    application are envisioned

-   1973/1974: Paul C. Lauterbur and Peter Mansfield apply gradient
    magnetic fields to obtain spatial encoding: first MR images (Nobel
    Prize in 2003)

-   1980's: MRI scanners enter the clinics.

-   1990's: enlarging field of clinical application, higher magnetic
    field magnets

-   2000-present: accelerating acquisition protocols.

At the present, more than 100 million MRI scans are worldwide performed
each year. Most common applications include tumors, multiple sclerosis,
epilepsy, ischemic stroke, stenosis or aneurysms (MR Angiography),
sardiac, musculoskeletal system, brain functioning (functional MRI), MR
guided surgery (e.g. MRI-Linac).

## Physical Principles

Central quantity is the *magnetization* $\mathbf{m} = (m_x ,m_y ,m_z)^T\in\mathbb{R}^3$, which describes the net effect of nuclei magnetic moments (in the body mostly $^1$H) over a small volume at position $\mathbf{r} = (x, y, z)^T$.

Suppose an external magnetic field $\mathbf{b}=(b_x,b_y,b_z)^T$ is present, the behavior of the magnetization is given by the following equation (by Felix Bloch): 

```{math}
:label: bloch
\frac{\text{d}\mathbf{m}}{\text{d}t}=
\left(\begin{array}{ccc}
-1/T_2 &\gamma b_z(t) & -\gamma b_y(t)\\
-\gamma b_z(t)& -1/T_2 & \gamma b_x(t)\\
\gamma b_y(t) & -\gamma b_x(t)&-1/T_1
\end{array}\right)\mathbf{m}+
\left(\begin{array}{c}
0\\0\\ \rho/T_1
\end{array}\right),\quad
\mathbf{m}(0)=\left(\begin{array}{c}
0\\0\\ \rho
\end{array}\right)
```

The quantities $T_1$, $T_2$ and $\rho$ are tissue
properties (parameters) and vary depending on the tissue type and
structure. This makes it possible to spot abnormalities in tissues.
$\gamma$ is a physical constant (gyromagnetic ratio).
The Bloch equation can be written in a more compact form:

```{math}
:label: blochmv
\frac{\text{d}\mathbf{m}}{\text{d}t} = A(t)\mathbf{m}+\mathbf{d},\quad \mathbf{m}(0) = \mathbf{m}_0
```

This equation can look rather complex at a first
glance. To better understand the dynamics involved let's separate the
tissue-parameter (i.e. $T_1,T_2,\rho$) dependent and independent parts.

```{math}
:label: rotation
\frac{\text{d}\mathbf{m}}{\text{d}t}=
\left(\begin{array}{ccc}
0 & \gamma b_z(t) & -\gamma b_y(t)\\
-\gamma b_z(t)& 0 & \gamma b_x(t)\\
\gamma b_y(t) & -\gamma b_x(t) &0
\end{array}\right)\mathbf{m}
```

Note that this differential equation is characterized
by a *skew-symmetric* matrix. This is equivalent to a *rotation* of
$\mathbf{m}$ around the magnetic field vector $\mathbf{b}$. The remaining part is given by

```{math}
:label: decay
\frac{\text{d}\mathbf{m}}{\text{d}t}=
\left(\begin{array}{ccc}
-1/T_2 & 0 & 0\\
0& -1/T_2 & 0\\
0 & 0 &-1/T_1
\end{array}\right)\mathbf{m}+
\left(\begin{array}{c}
0\\0\\ \rho/T_1
\end{array}\right)
```

This differential equation clearly describes two
independent *exponential decays*, one for
$m_{xy}\equiv m_x+\mathrm{i}m_y$ with decay rate $-1/T_2$ and one for
$m_z$ with rate $-1/T_1$.

In conclusion: applied, time-dependent magnetic field $\mathbf{b}$
contributes to rotate the magnetization vector. The tissue properties
contribute to different decay (or relaxation) behavior for different
tissue types. During an MRI scan, the interplay of these two kind of
dynamics makes it possible to generate an image.

## An MRI scan

Let's start by investigating what happens to a single tissue component
during an MRI scan. When we lay inside an MRI scanner, the large
doughnut-shaped magnet generates a very strong magnetic field in the
feet-head direction. By definition, this is the longitudinal, $z$
component. The transverse component $m_{xy}$ is randomly distributed
over a small volume $V$ (zero net sum over $V$). Only the longitudinal component $m_z$ is slightly different than 0, but since we can only measure the $m_{xy}$ component, no signal can be
collected.

To measure the magnetic moments, we need them to attain a net transverse
magnetization component which differs from 0. Since the $\mathbf{m}$ is
initially aligned along $z$ we will need a magnetic field component
perpendicular to that (remember: the magnetic field $\mathbf{b}$
contributes to rotate $\mathbf{m}$ around itself). For simplicity,
suppose we superimpose a time-independent magnetic field $\mathbf{b}$
that is aligned along the $y$ direction: $\mathbf{b}(t) = (0,b_y,0)^T$
for each $t>0$. The Bloch equation {eq}`blochmv`
admits the following solution:

$$\mathbf{m}(t) = e^{At}\left(\mathbf{m}_0+A^{-1}\mathbf{d} \right)-A^{-1}\mathbf{d}$$

A graph of the solution $m_{xy}$ for three different combinations of
tissue parameters $(T_1,T_2)$ is plotted in {numref}`transient`. What would have happened if we had chosen a different value for the magnetic field $b_y$? Let's have a look at the steady-states solution of the Bloch equation for time-constant magnetic fields. To derive the
steady state signal, set $\text{d}\mathbf{m}/\text{d}t = \mathbf{0}$:

$$
\frac{\text{d}\mathbf{m}}{\text{d}t} = 0\Leftrightarrow 0 = A\mathbf{m}+\mathbf{d}\Leftrightarrow \mathbf{m} = -A^{-1}\mathbf{d}
$$

Which leads to

$$
m_{xy} = \frac{T_2\gamma b_y}{T_1T_2 (\gamma b_y)^2+1} \rho.
$$ 

A plot of $m_{xy}(t)$ for different values of the magnetic field component
$b_y$ is shown in {numref}`steady`. Note that not only the absolute amplitude of each
signal changes, but, more importantly, also the *relative* amplitude
(contrast) changes; for instance, for $\gamma b_y=3$, tissue C gives
higher signal than tissue B. At $\gamma b_y=8$ the situation is
inverted. Furthermore, for high magnetic field values, tissue A and B
generate the same signal amplitude, which means they cannot be
differentiated at all.

+++
````{panels}
:container: container-fluid
:column: col-lg-6 col-md-6 col-sm-6 col-xs-12
:card: shadow-none border-0

```{figure} images/magnetic_resonance_imaging/transient.png
:height: 150px
:name: transient

Transient states behavior of $m_x$ for different tissue types. In these simulations, $m_y$ = 0.
```

---

```{figure} images/magnetic_resonance_imaging/steady.png
:height: 150px
:name: steady

Steady states of $m_x$ as a function of the experimental setting $\gamma b_y$ for three different tissue types.
```

````
+++

To better illustrate this point, I simulated the transverse
magnetization for a 2D object made of these three tissue types. See {numref}`contrast`.

```{figure} images/magnetic_resonance_imaging/contrast.png
:height: 150px
:name: contrast

Contrast obtained for dierent experimental settings in a 2D object.
```

Tuning the scanner parameter $b_y$ to different levels results in
different contrast images. A more realistic, in-vivo illustration of how
the scanner settings influence the acquired image is reported in figure
{numref}`contrast_invivo`. Note that the these four images are
acquired from the same slice of the same brain! The ability to capture
different anatomical and functional characteristics just by changing
magnetic field settings makes MRI probably the richest medical imaging
modality.

```{figure} images/magnetic_resonance_imaging/contrast_invivo.png
:height: 150px
:name: contrast_invivo

Four different in-vivo contrast images acquired for the same slice of the same brain.
```

## Imaging

In the previous paragraph we have investigate the behavior of an
individual tissue type. In this section we will see how data from a
whole object (human body) is acquired and processed into an image.\
Look again at {numref}`transient`. Once the steady-state condition is reached (in
that specific case for $t>3$) the value of $m_{xy}$ does no longer
change. This allows to encode the *spatial* information relative to all
position in the image $m_{xy}(\mathbf{r})$. Spatial encoding can take
some time thus it is important that the state of $m_{xy}$ is kept
constant during the data acquisition period. We now make a distinction
between two kinds of superimposed magnetic fields:

*   the Radiofrequency fields are the transverse components $b_x$ and
    $b_y$
*   the Gradient fields describe the longitudinal component $b_z$ and
    have the form of a gradient:
    $b_z(t,\mathbf{r}) = \mathbf{g}(t)\cdot\mathbf{r}$ where
    $\mathbf{g}=(g_x,g_y,g_z)\in\mathbb{R}^3$ can be tuned at the user's
    discretion. Note that the gradient fields are spatially dependent
    (this is essential to localize the signal contribution form
    different positions in the body).

During an MRI scan, these two types of fields are not played out
continuously but keep alternating (see {numref}`sequence`).

```{figure} images/magnetic_resonance_imaging/sequence.png
:height: 250px
:name: sequence

A simple MRI acquisition sequence.
```

This is a slightly different situation than the scenarios we have
previously seen but the physical insights are pretty much the same. The
RF fields are used to drive the magnetization to some kind of desired
state (for instance a steady-sate) as we have seen in the previous
section (remember that, in order to record a signal: $m_{xy}\neq 0$).
The gradient fields are used to encode the information, i.e. to collect
a signal which can be reconstructed into an image. The period of this
alternation is called *repetition time* ($T_R$) and the time-distance
between the radiofrequency and the gradient blocks is the *echo time*
($T_E$); $T_R$ and $T_E$ can be arbitrarily set to modify the contrast
type.

Since each data acquisition block (read-out) lasts a very short time
(few milliseconds) we can ignore the relaxation effects and study their
behavior at the hand of equation {eq}`rotation`.
In particular, suppose that at the start of the acquisition ($t =t_0$)
we are in a steady-state $m_{xy}=u$; under effect of the gradient
fields, the transverse magnetization will have the following form:

```{math}
:label: mss
m_{xy}(t,\mathbf{r}) = u(\mathbf{r})e^{-\text{i}\gamma\int_{t_0}^{t}\mathbf{g}(t')\cdot \mathbf{r}\text{d}t'}
```

which is of course still a localized signal expression. We will now make the final step toward MR Imaging.
In an MRI scanner, radiofrequency receive coils are used to capture the
total amount of transverse magnetization present in the object. By using
Faraday's induction law, it can be shown that the signal $s(t)$ recorded
by the coil is proportional to the volumetric integral of all local
$m_{xy}$ contributions: 

```{math}
:label: signal
s(t) \propto \int_{\mathbb{R}^n}m_{xy}(t,\mathbf{r})\text{d}\mathbf{r} = \int_{\mathbb{R}^n}u(\mathbf{r})e^{-\text{i}\gamma\int_{t_0}^{t}\mathbf{g}(\tau)\cdot \mathbf{r}\text{d}\tau}\text{d}\mathbf{r}
```

where we made use of equation {eq}`mss` for the second equality.
It is important to realize that the measured signal $s(t)$ does not
directly tell us how the spatial distribution of $u$ (i.e. the image)
look like. In order to reconstruct $u(\mathbf{r})$ we need to solve the
integral equation {eq}`signal`. This step is going to be much easier if we rewrite
this equation by introducing the following variable:

```{math}
:label: k
\mathbf{k}(t)\equiv \frac{\gamma}{2\pi}\int_{t_0}^t\mathbf{g}(\tau)\text{d}\tau
```

```{math}
:label: signalk
\Rightarrow s(t) = s(\mathbf{k}(t))= \int_{\mathbb{R}^n}u(\mathbf{r})e^{-2\pi \text{i}\mathbf{k}(t)\cdot \mathbf{r}}\text{d}\mathbf{r}
```

Note that, for ease of notation, we have dropped the
proportionality sign from {eq}`signal` (you
could think of it as a global, space-independent factor which scales
$u$). Obviously, the equation above unveils a Fourier transform relationship
between the signal and the image $u$. To reconstruct the image we can
apply inverse (Discrete) Fourier transform to the data $s$ (see figure
{numref}`kspace`).

```{figure} images/magnetic_resonance_imaging/kspace_image.png
:height: 250px
:name: kspace

$k$-space and image space. The data acquisition process is mathematically equivalent to a Fourier
transform F of the image. To reconstruct the image, inverse Fourier transform $\mathcal{F}^{-1}$ can be applied.
```

The new variable $\mathbf{k}$ denotes a *spatial frequency* term and the
signal can be interpreted as the multi-dimensional frequency
representation of the image. The signal domain is also called the
$k$-space.\
Equation {eq}`signalk` has profound implications not only in the
reconstruction but also in the data *acquisition* process. Fourier
theory tells which sample point in $k$-space need to be acquired in
order to obtain a faithful reconstruction of the true magnetization map
of the body. In particular, the Nyquist criterion states that for an
image of size $L\times L$ and a desired resolution $\Delta_s$ in both
directions, samples in $k$-space should be acquired at distance $1/L$
over the interval $[-1/\Delta_s, + 1/\Delta_s]$. See {numref}`nyquistMRI`.

```{figure} images/magnetic_resonance_imaging/nyquistMRI.png
:height: 250px
:name: nyquistMRI

Sampling in $k$-space
```

In conclusion, the MRI acquisition process is characterized by two main
components:

*   Radiofrequency excitations to drive and keep the magnetization into
    a tissue-dependent state $u(\mathbf{r})$
*   gradient field encoding to collect the signal $s(\mathbf{k})$ as
    spatial frequency coefficients of $u(\mathbf{r})$

The first component is responsible for the different type of contrast
characteristics of the image. The second component make sure that the
acquired data can correctly be reconstructed at the desired resolution.

### Traversing the $k$-space

Let's have a closer look at the data acquisition component (encoding).
This is achieved by ensuring sufficient portion of the $k$-space is
sampled, with the proper density (Nyquist criterion). In practice, there
are several ways to sample the $k$-space. Equation
{eq}`k` shows the direct
relationship between the somehow abstract concept of $k$-space
coordinate and the practical, actually produced gradient field
$\mathbf{g}(t)$; given a desired sampling trajectory, $\mathbf{k}(t)$,
we can easily derive the corresponding scanner's input parameter
$\mathbf{g}(t)$.

As a very simple example, consider a 2D cartesian encoding scheme with
equal time-steps $\Delta_t$ ({numref}`kline`).

```{figure} images/magnetic_resonance_imaging/kline.png
:height: 250px
:name: kline

Top: a very simple 2D cartesian trajectory in red. Bottom: corresponding gradient waveforms in
green and blue.
```

To traverse a single line, parallel to the $k_x$ axis and given
$k_y=\beta$ coordinate, we first impose

$$
k_y = \frac{\gamma}{2\pi}\int_{0}^{\Delta_t}g_y(\tau)\text{d}\tau=\beta
$$

Assuming a constant $g_y$, this leads to
$g_y=\frac{2\pi\beta}{\gamma\Delta_t}$.
For $k_x$ a similar argument shows that, in order to move to the
$k_x = \alpha$ position at $t=\Delta_t$:
$g_x=\frac{2\pi}{\gamma\Delta_t}\alpha$.
Thus, for $t\in[0,\Delta_t]$, we set
$\mathbf{g}=(\frac{2\pi}{\gamma\Delta_t}\beta,\frac{2\pi}{\gamma\Delta_t}\alpha)^T$.
At time $t = \Delta_t$ we have $\mathbf{k}(\Delta_t)=(\alpha,\beta)$. We
start acquiring data keeping $k_y$ constant, thus from now on, $g_y=0$
(why?). Assume that $\Delta_k$ is the desired distance between
consecutive $k$-space locations. For $g_x$ we need to solve:

$$
\Delta_k = \frac{\gamma}{2\pi}\int_{\Delta_t}g_x(\tau)\text{d}\tau \Rightarrow g_x = \frac{2\pi}{\gamma\Delta_t}\Delta_k.
$$

Thus, for each $t>\Delta_t$ we set
$\mathbf{g}=(\frac{2\pi}{\gamma\Delta_t}\Delta_k,0)^T$. See {numref}`kline` for a
pictorial illustration.
Although the cartesian sampling is by far the most commonly used scheme
for clinical MRI exams, other trajectories are gradually entering the
MRI practice. See {numref}`trajectories` for other 2D and 3D examples.

```{figure} images/magnetic_resonance_imaging/trajectories.png
:height: 250px
:name: trajectories

Examples of 2D and 3D non-cartesian $k$-space trajectories
```

### Image reconstruction: beyond the Fast Fourier Transform

We discretize the signal equation {eq}`signalk` as

```{math} 
:label: signalm
\mathbf{s}  = \mathcal{F}\mathbf{u}
```

where we used the following notation:

*   $\mathbf{s}$: the vector of signal samples ($k$-space data)
*   $\mathbf{u}$: the vector of image values (i.e. a discretized version
    of $u(\mathbf{r})$ )
*   $\mathcal{F}$: the $N$-dimensional sampling (Fourier transform)
    operator: $\mathcal{F}_{i,j} = \exp\left(-2\pi\text{i}\mathbf{k}_i\cdot \mathbf{r}_j\right)\nu$
    with volume element $\nu$.

For cartesian, equidistant samples of $k$-space, the operator
$\mathcal{F}$ in equation {eq}`signalm` is unitary and can be implemented by the Fast
Fourier transform (FFT) algorithm. To reconstruct $\mathbf{u}$, the
inverse FFT can be applied: $\mathbf{u} = \mathcal{F}^{-1}\mathbf{s}$
where in this case $\mathcal{F}^{-1}=\mathcal{F}^H$. This is :

* computationally efficient; applying FFT to a $N$-long array scales with $N\log_2 N$ while naive implementation would scale with $N^2$.
* memory efficient, because no operator needs to be stored.

For other, non-cartesian trajectories, the solving strategies are more
complex. First of, all, note that $\mathcal{F}$ in most cases will not
be a unitary, square operator. The problem will be solved by minimizing
some norm of the residual, usually the squared $\ell_2$ norm
(least-squares problems):

$$
\arg\min_{\mathbf{u}}\|\mathcal{F}\mathbf{u}-\mathbf{s}\|_2^2
$$

It is
interesting to give a closer look at this numerical problem. Suppose we
acquire data according to a 2D spiral trajectory, which can be
parametrized as: 

$$
\left(k_x(t),k_y(t)\right)  = \left(\alpha t \cos(\beta t), \alpha t \sin(\beta t)\right).
$$ 

with $\alpha,\beta\in\mathbb{R}$.

{numref}`svd` shows a portion of the samples at the $k$-space center. For design
constraints, points in the middle are mutually much closer. As a
consequence, many rows in the operator $\mathcal{F}$ are almost equal,
leading to problems with respect to linear dependence and orthogonality.
This is more clearly demonstrated in {numref}`svd` which show
the spectrum (singular values) of $F$; the singular values are not equal
(a characteristic of orthogonal operators) but gradually decay to
machine precision level after a while, leading to ill-conditioning.

```{figure} images/magnetic_resonance_imaging/svd.png
:height: 250px
:name: svd

Magnified portion of $k$-space samples for a 2D spiral trajectory and the spectrum of the corresponding $\mathcal{F}$ operator. Note the decaying singular values.
```

To mitigate this problem, regularization needs to be applied. This is
usually done by introducing a penalty $R$ on the computed solution
$\mathbf{u}$:

$$
\arg\min_{\mathbf{u}}\|\mathcal{F}\mathbf{u}-\mathbf{s}\|^2_2+\lambda R(\mathbf{u})
$$

The most common choice for $R$ is $R(\mathbf{u})=\|\mathbf{u}\|_2^2$
(Tikhonov regularization). Other common choices will be shown below.

### Acceleration techniques

Note that the sampling is a time-dependent process; if two consecutive
samples are acquired at $\Delta_t$ distance, acquisition of $N$ samples
along the trajectory will take $N\Delta_t$ amount of time. In practice,
$\Delta_t$ is in the order of $10^{-6}$. Acquiring 2D image of size
$256\times 256$ will require $256^2\approx 65,000$ samples, thus a
fraction of a second.

However, a single 2D image is not sufficient for diagnosis and either
several stacked 2D images (slices) are needed or a full-3D acquisition.
In both cases, the amount of data needed is multiplied by a factor close
to 100 (either 100 2D slices or an additional dimension). This time, the
acquisition would last several seconds or even minutes, depending on the
sequence timings $T_R$ and $T_E$. In this analysis, I consider only the
duration of the *sampling* component, remember (cf. {numref}`sequence`)
that also RF events need to be played out in order to drive and keep the
magnetization in a desired state.

Toward the end of the previous century, a solution strategy was found to
accelerate the scans. It all began around the year 1990 when multiple
receive channels were introduced to increase the signal-to-noise-ratio
(SNR). However, MRI scientists soon realized that using multiple coils
could also allow for shorter data-acquisition protocols.

Parallel imaging in MRI works in the following way. Due to
electromagnetic interference effects (electromagnetism is a wave
phenomenon), each $p$-th receiver coil is characterized by its own
radiofrequency field (sensitivity map) $c^p(\mathbf{r})$. Therefore, the
signal equation has to be slightly modified:

$$
\mathbf{s}^p = \mathcal{F}C^p\mathbf{u},\quad p = 1,\dots, N
$$ 

where $C^p$ is a diagonal matrix with $C^p_{n,n} = c^p(\mathbf{r}_n)$ and
$\mathbf{s}^p$ is the signal from the individual coil. The image is
obviously the same for all receive coils, thus it can be reconstructed
by solving the joint reconstruction system: 

$$
\left[
\begin{array}{c}
\mathbf{s}^1\\\mathbf{s}_2\\ \vdots\\\mathbf{s}^N
\end{array}
\right]=
\left[
\begin{array}{c}
\mathcal{F}C^1\\\mathcal{F}C^2\\ \vdots\\\mathcal{F}C^N
\end{array}
\right]\mathbf{u}
$$ 

which we can be solved as

```{math}
:label: joint
\arg\min_{\mathbf{u}}\|\mathbf{s}-F\mathbf{u}\|_2^2+\lambda R(\mathbf{u})
```
    
where $\mathbf{s}$ now denotes the concatenated
signal from all coils and $F$ the joint linear multi-coil model.

Note that the operator $F$ is a tall operator, that is, there are (many)
more data samples (rows) than unknowns (columns). It makes therefore
sense to *under-sample* $\mathcal{F}$ by deleting rows, that is,
skipping points in $k$-space. Since the sensitivity matrices $C^p$ are
not equal, the resulting joint operator $F$ should still have full rank,
making the inversion possible. Of course, the practical consequence of
under-sampling the $k$-space is a much shorter acquisition protocol.
Although there are limits to the under-sampling level as set by the
number of coils, geometry and trade-off SNR/time, acceleration factors
between 2 and 5 are quite common. For this reason, parallel imaging is
nowadays applied in almost all clinical protocols.

The most recent advance in accelerated protocols is the introduction of
compressed sensing. Basically, the reconstruction problem {eq}`joint` is solved
with a non-linear regularization term $R$, usually in the form
$R(\mathbf{u})=\|\mathbf{u}\|_1$, $R(\mathbf{u})=\|W\mathbf{u}\|_1$ or
$R(\mathbf{u})=TV(\mathbf{u})$ where $W$ denotes a *sparsity* operator
and $TV$ the total variation functional. The sampling strategy needs to
be adapted to guarantee *incoherence*, which can be explained as a
noise-like behavior of under-sampling artifacts. Pseudo-randomization of
the $k$-space trajectory with higher sampling density in the middle is
usually the most practical choice. See {numref}`cs`.

```{figure} images/magnetic_resonance_imaging/CS_philips.png
:height: 250px
:name: cs

Compressed sensing in MRI. Top: illustrative sketch of 2D k-space sampling strategy. Bottom:
Reconstruction by Compressed sensing. Note that the variable-density randomized sampling (right) gives the best reconstruction quality. Image taken from Geerts-Ossevoort et al (Philips). Compressed Sense (2018).
```

Compressed sensing has recently entered clinical practice and showed to
further accelerate exams by about 40%.

## Exercises 

### Rotation
Can you prove that {eq}`rotation` defines a rotation and that the
rotation axis is aligned along $\mathbf{b}$? What is the rotation
angle?

### Decay
Rewrite {eq}`decay` as two independent equations, one for the *transverse* component  $m_{xy}$ and one for the *longitudinal* component $m_z$. Find the solution for both equations for a generic time point $t$ with initial values $m_{xy}(0) =m_{xy}^0$ and $m_z(0) = m_z^0$. What happens in the limit case $t\rightarrow +\infty$?
 
### 
Prove Equation {eq}`mss`. You may use the result from the first exercise.

### 
By using {eq}`k`, can you derive an expression for $\mathbf{g}(t)$ for the trajectories in {numref}`trajectories`? You do not need to assume $\mathbf{g}$ is piece-wise constant as in the cartesian example above.

## Assignment 

### MRI-reconstruction
Set up a regularised inversion for an MRI scan with standard cartesian sampling. As test data you can use `skimage.data.brain`. You may assume that the data are noisy (subsampled) versions of the Fourier transform of the image slices. Can you get rid of aliasing artefacts using appropriate Tikhonov regularisation?