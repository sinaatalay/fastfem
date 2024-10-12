# __Theory__

## __Introduction__

Phenomena guided by partial differential equations are omnipresent in a wide variety of scientific fields. While solutions can be found for simple initial and boundary conditions, their complex nature renders an analytical solution extremely difficult to derive, depending on the phsyics at hand and the geometry of the problem. In these cases, numerical schemes may be adopted to derive an approximate solution and gain insight into the problem, such as the finite element method. 

In the context of our final project, we present a finite element method scheme that will solve a transient heat conduction problem in a two-dimensional domain, which can be formulated as[@lienhard2019heat]:

$$
T_{xx} + T_{yy} = \frac{1}{\alpha(T)} T_t + g(x,y)
$$

Where $ T(x,y,t) $ is the temperature profile, $\frac{1}{\alpha(T)}$ is the thermal diffusivity of the medium, and $g(x,y)$ represents any volumetric heat sources (as nromalized by the thermal conductivity). Note that the thermal diffusivity may be formulated as:

$$
\alpha(T) = \dfrac{k}{\rho(T) c(T)} 
$$

Where we consider the variations in the thermal diffusivity $k$ to be small compared to the density $\rho$ and specific heat $c$, such that it may be treated as a constant.

## __Methodology__

As previously shown, the strong form of the FEM problem is formulated as:

$$
T_{xx} + T_{yy} = \frac{1}{\alpha(T)} T_t + g(x,y)
$$

Or written in a more compact form:

$$
\mathbf{\nabla^2 T} = \frac{1}{\alpha} \frac{\partial T}{\partial t} + g(x,y)
$$

And, as part of the formulation for the FEM problem, if we consider the two-dimensional domain to be denoted by $\Omega$ and its boundary by $d\Omega$, its weak form may be derived[@wiki:FEM] :

$$
\int_\Omega  \mathbf{\nabla^2T v} ds = \int_\Omega \frac{1}{\alpha} \frac{\partial T}{\partial t}ds + \int_\Omega g v ds \\\
-\int_\Omega  \mathbf{\nabla T} \cdot \mathbf{\nabla v} ds = \int_\Omega \frac{1}{\alpha} \frac{\partial T}{\partial t}ds + \int_\Omega g v ds
$$

Which may be simply reduced to a map $\phi$[@wiki:FEM]:

$$
-\phi(T,v) = \int_\Omega gv ds
$$

Where each function may be expressed as a basis of the shape functions $N_i(x,y)$ of the problem[@simulate2021fem], namely:

$$
T(x,y,t_c) = \sum_{i=1}^n T_i N_i(x,y) \\\
g(x,y) = \sum_{i=1}^n g_i N_i(x,y) \\\
v(x,y) = \sum_{i=1}^n v_i N_i(x,y) 
$$

For a given time step $t_c$. Substituting the sum, the problem reduces to:

$$
- \sum_{i=1}^n  \sum_{j=1}^n \phi(T_i N_i, v_j N_j) = \sum_{i=1}^n  \sum_{j=1}^n \int_{\Omega} g_i v_j N_i N_j ds
$$

Where, using integral and algebraic properties, by linearity:

$$
- \sum_{i=1}^n  \sum_{j=1}^n T_i \phi(N_i, N_j) = \sum_{i=1}^n  \sum_{j=1}^n g_i \int_{\Omega}  N_i N_j ds
$$

Where, if we denote $\phi(N_i, N_j)$ by a stiffness matrix $\mathbf{L}$, $T_i$ by a vector $\mathbf{T}$, the integral $\int_{\Omega}  N_i N_j ds$ by a mass matrix $\mathbf{M}$, and the source term $g_i$ by a vector $\mathbf{g}$[@wiki:FEM], we obtain the following linear system:

$$
-\mathbf{L} \mathbf{T} = \mathbf{M} \mathbf{g} 
$$

where the linear system may be solved iteratively for the vector (and function) $\mathbf{T}$ at each time step.





## __Implementation__

### __Linear Mapping and Thermal Diffusivity__

### __Using the Galerkin Method of Residuals__

### __Solving the System__



## __Results__

## __Conclusion__