# Theory

## Introduction

Phenomena guided by partial differential equations are omnipresent in a wide variety of scientific fields. While solutions can be found for simple initial and boundary conditions, their complex nature renders an analytical solution extremely difficult to derive, depending on the physics at hand and the geometry of the problem. In these cases, numerical schemes may be adopted to derive an approximate solution and gain insight into the problem, such as the finite element method. 

We present a finite element method scheme that will solve a transient heat conduction problem in a two-dimensional domain, which can be formulated as[@lienhard2019heat]:

$$
T_{xx} + T_{yy} = \frac{1}{\alpha(T)} T_t + g(x,y) \tag{1}
$$

Where $ T(x,y,t) $ is the temperature profile, $\frac{1}{\alpha(T)}$ is the thermal diffusivity of the medium, and $g(x,y)$ represents any volumetric heat sources (as nromalized by the thermal conductivity). Note that the thermal diffusivity may be formulated as:

$$
\alpha(T) = \dfrac{k}{\rho(T) c(T)} \tag{2}
$$

Where we consider the variations in the thermal diffusivity $k$ to be small compared to the density $\rho$ and specific heat $c$, such that it may be treated as a constant.

## Methodology

As previously shown, the strong form of the FEM problem is formulated as:

$$
T_{xx} + T_{yy} = \frac{1}{\alpha(T)} T_t + g(x,y) \tag{3}
$$

Or written in a more compact form:

$$
\nabla^2 T = \frac{1}{\alpha} \frac{\partial T}{\partial t} + g(x,y) \tag{4}
$$

And, as part of the formulation for the FEM problem, if we consider the two-dimensional domain to be denoted by $\Omega$ and its boundary by $d\Omega$, its weak form may be derived[@comsol_fem] :

$$
\int_\Omega  \nabla^2T v ds = \int_\Omega \frac{1}{\alpha} \frac{\partial T}{\partial t}ds + \int_\Omega g v ds \tag{5}
$$

Or using Green's identity:

$$
-\int_\Omega  \mathbf{\nabla T} \cdot \mathbf{\nabla v} ds = \int_\Omega \frac{1}{\alpha} \frac{\partial T}{\partial t}ds + \int_\Omega g v ds \tag{6}
$$

Where $v=v(x,y)$ is a test function in the Sobolev space $H_0^1$. This may be simply reduced to a a sum of maps $\phi$[@comsol_fem]:

$$
-\left( \phi_1(T,v) + \phi_2(\dot{T},v)\right) = \int_\Omega gv ds \tag{7}
$$

Where each function may be expressed as a basis of the shape functions $N_i(x,y)$ of the problem[@simulate2021fem], namely:

$$
T(x,y,t_c) = \sum_{i=1}^n T_i N_i(x,y)  \\\ \tag{8} 
\dot{T}(x,y,t_c) = \sum_{i=1}^n T_i' N_i(x,y)  \\\
g(x,y) = \sum_{i=1}^n g_i N_i(x,y)  \\\
v(x,y) = \sum_{i=1}^n v_i N_i(x,y) 
$$

For a given time step $t_c$. Substituting the sum, the problem reduces to:

$$
- \sum_{i=1}^n  \sum_{j=1}^n \left( \phi_1(T_i N_i, v_j N_j) + \phi_2(T_i' N_i, v_j N_j) \right) = \sum_{i=1}^n \sum_{j=1}^n \int_{\Omega} g_i v_j N_i N_j ds
$$

Where, using integral and algebraic properties, by linearity:

$$
- \sum_{i=1}^n  \sum_{j=1}^n T_i T_i'\left( \phi_1(N_i, N_j) + \phi_2(N_i, N_j) \right) = \sum_{i=1}^n  \sum_{j=1}^n g_i \int_{\Omega}  N_i N_j ds \tag{10}
$$

Where, if we denote $\phi_1(N_i, N_j)$ by a stiffness matrix $\mathbf{L_1}$, $\phi_2(N_i, N_j)$ by a stiffness matrix $\mathbf{L_2}$, $T_i$ by a vector $\mathbf{T}$, $T_i'$ by a vector $\mathbf{T'}$, the integral $\int_{\Omega}  N_i N_j ds$ by a mass matrix $\mathbf{M}$, and the source term $g_i$ by a vector $\mathbf{g}$[@comsol_fem], we obtain the following linear system:

$$
\mathbf{L_1} \mathbf{T} + \mathbf{L_2} \mathbf{T'}= -\mathbf{M} \mathbf{g} \tag{11}
$$

where the linear system of ordinary differential equations may be solved iteratively for the vector (and function) $\mathbf{T}$ at each time step.





## Implementation

### Linear Mapping and Thermal Diffusivity

### Using the Galerkin Method of Residuals

### Solving the System



## Results

## Conclusion