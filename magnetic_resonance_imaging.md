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
tissue parameters $(T_1,T_2)$ is plotted in figure {numref}`transient`. What would have happened if we had chosen a different value for the magnetic field $b_y$? Let's have a look at the steady-states solution of the Bloch equation for time-constant magnetic fields. To derive the
steady state signal, set $\text{d}\mathbf{m}/\text{d}t = \mathbf{0}$:

$$
\frac{\text{d}\mathbf{m}}{\text{d}t} = 0\Leftrightarrow 0 = A\mathbf{m}+\mathbf{d}\Leftrightarrow \mathbf{m} = -A^{-1}\mathbf{d}
$$

Which leads to

$$
m_{xy} = \frac{T_2\gamma b_y}{T_1T_2 (\gamma b_y)^2+1} \rho.
$$ 

A plot of $m_{xy}(t)$ for different values of the magnetic field component
$b_y$ is shown in figure {numref}`steady`. Note that not only the absolute amplitude of each
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
magnetization for a 2D object made of these three tissue types. See figure {numref}`contrast`.

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

## Exercises 

### Rotation
Can you prove that {eq}`rotation` defines a rotation and that the
rotation axis is aligned along $\mathbf{b}$? What is the rotation
angle?
