from femo.fea.fea_dolfinx import *
from femo.csdl_opt.fea_model import FEAModel
from femo.csdl_opt.state_model import StateModel
from femo.csdl_opt.output_model import OutputModel
import dolfinx.fem as dolfin_fem
import argparse

from ufl import ln, pi

file_path = 'examples/advanced_examples/robotic_fish/meshes/'
output_path = "examples/advanced_examples/robotic_fish/temp/"

# file_path = 'meshes/'
# with XDMFFile(MPI.COMM_WORLD, "examples/advanced_examples/robotic_fish/meshes/module_v1.xdmf", "r") as xdmf:
with XDMFFile(MPI.COMM_WORLD, file_path + "module_v1.xdmf", "r") as xdmf:
       mesh = xdmf.read_mesh(name="Grid")
nel = mesh.topology.index_map(mesh.topology.dim).size_local
nn = mesh.topology.index_map(0).size_local

# mvc = MeshValueCollection("size_t", mesh, 2) 
# with XDMFFile("mf.xdmf") as infile:
#     infile.read(mvc, "name_to_read")
# mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
from dolfinx import mesh as dolfinx_mesh_module

# with XDMFFile(MPI.COMM_WORLD, "examples/advanced_examples/robotic_fish/meshes/segment0.xdmf", "r") as xdmf:
#        left_chamber_mesh = xdmf.read_mesh(name="Grid")

mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
# with XDMFFile(MPI.COMM_WORLD, "examples/advanced_examples/robotic_fish/meshes/module_v1_left_chamber_inner_surfaces.xdmf", "r") as xdmf:
with XDMFFile(MPI.COMM_WORLD, file_path + "module_v1_left_chamber_inner_surfaces.xdmf", "r") as xdmf:
       left_chamber_facet_tags = xdmf.read_meshtags(mesh, name="Grid")
# print(left_chamber_facet_tags.mesh)
# print(left_chamber_facet_tags.dim)
# print(left_chamber_facet_tags.indices)
# print(left_chamber_facet_tags.values)


mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
# with XDMFFile(MPI.COMM_WORLD, "examples/advanced_examples/robotic_fish/meshes/module_v1_right_chamber_inner_surfaces.xdmf", "r") as xdmf:
with XDMFFile(MPI.COMM_WORLD, file_path + "module_v1_right_chamber_inner_surfaces.xdmf", "r") as xdmf:
       right_chamber_facet_tags = xdmf.read_meshtags(mesh, name="Grid")


# mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
# with XDMFFile(comm,'.../input_boundary.xdmf','r') as infile:
#     mt = infile.read_meshtags(mesh, "Grid")

# mf = MeshFunction("size_t", mesh, 1, 0)
# LeftOnBoundary().mark(mf, 1)

fea = FEA(mesh)
# Record the function evaluations during optimization process
fea.record = True

mesh = fea.mesh

parameterization_displacements_function_space = VectorFunctionSpace(mesh, ("CG", 1))
u_hat = Function(parameterization_displacements_function_space)

# Add state to the PDE problem:
state_name = 'u'
state_function_space = VectorFunctionSpace(mesh, ('CG', 1))
# input_function_space = FunctionSpace(mesh, ('DG', 0))
input_function_space = FunctionSpace(mesh, ('CG', 2))   # For some reason, finding the dofs only works if the input function space is CG2
material_properties_function_space = FunctionSpace(mesh, ('DG', 0))
# dummy_function_space = FunctionSpace(mesh, ('CG', 2))
u = Function(state_function_space)
v = TestFunction(state_function_space)
du = Function(state_function_space)
B = Constant(domain=mesh, c=(0.0, 0.0, 0.0))  # Body force per unit volume
T = Constant(domain=mesh, c=(0.0, 0.0, 0.0))  # Traction force on the boundary
# B = Constant(domain=mesh, c=(0.0, -0.5, 0.0))  # Body force per unit volume
# T = Constant(domain=mesh, c=(0.1,  0.0, 0.0))  # Traction force on the boundary


# Time-stepping
t_start = 0.0  # start time
t_end = 0.02  # end time
# t_end = 0.5  # end time
# t_end = 4.  # end time
# t_end = 2.  # end time
# t_end = 1.0  # end time       # Good amount of time for one stroke
t_steps = 3  # number of time steps
# t_steps = 51  # number of time steps
# t_steps = 101  # number of time steps
# t_steps = 201  # number of time steps
# t_steps = 301  # number of time steps
# t_steps = 401  # number of time steps
# t_steps = 801  # number of time steps
# t_steps = 1001  # number of time steps
# t_steps = 51  # number of time steps
# t_steps = 3  # number of time steps

t, dt = np.linspace(t_start, t_end, t_steps, retstep=True)
dt = float(dt)  # time step needs to be converted from class 'numpy.float64' to class 'float' for the .assign() method to work (see below)

p0 = 0.
# pump_max_pressure = 0.
# pump_max_pressure = 1.
# pump_max_pressure = 1.e2
# pump_max_pressure = 3.e4
# pump_max_pressure = 3.5e4
# pump_max_pressure = 4.e4
# pump_max_pressure = 5.e4
# pump_max_pressure = 1.e5
# pump_max_pressure = 2.e5
pump_max_pressure = 3.e5
# pump_max_pressure = 4.e5
# pump_max_pressure = 5.e5
# pump_max_pressure = 6.e5
# pump_max_pressure = 1.e6
# pump_max_pressure = 1.e7
# pump_vacuum_pressure = 0.
# pump_vacuum_pressure = -1.e4
# pump_vacuum_pressure = -3.e4
# pump_vacuum_pressure = -5.e4
pump_vacuum_pressure = -1.e5
actuation_frequency = 0.5 # Hz
# actuation_frequency = 0.125 # Hz
stroke_period = 1./actuation_frequency/2    # 2 strokes per cycle
num_strokes = int(actuation_frequency*t_end)*2+1   # 2 strokes per cycle
# time_constant = 8*actuation_frequency    # in reality this would be a constant from RC circuit analysis, but I want it to be nice for me.
time_constant = 6*actuation_frequency    # in reality this would be a constant from RC circuit analysis, but I want it to be nice for me.
# time_constant = 3*actuation_frequency    # in reality this would be a constant from RC circuit analysis, but I want it to be nice for me.

# u = Function(state_function_space)
# u_bar = Function(state_function_space)
# du = Function(state_function_space)
# ddu = Function(state_function_space)
# ddu_old = Function(state_function_space)

# alpha_m = Constant(mesh, 0.2)
# alpha_f = Constant(mesh, 0.4)
rho_inf = 1.0    # asymptotic spectral radius   # NOTE: This is unstable for MBD simulations (alebraic constraints make this unstable)
# rho_inf = 0.9    # asymptotic spectral radius   # NOTE: Apparently typical values are between 0.6 and 0.9. I want to try adding damping
# rho_inf = 0.8    # asymptotic spectral radius   # NOTE: Currently unstable for rho_infinity = 0.9. Should try jacobian to get stiffness damping
# rho_inf = 0.8 gives stability and better displacement, but still has high frequency oscillations
# rho_inf = 0.7
# rho_inf = 0.6    # asymptoti/c spectral radius   # NOTE: Apparently typical values are between 0.6 and 0.9. I want to try adding damping
# I want to try adding some damping to account for the high frequencies induced from the high stiffness elements (I don't care about higher frequencies)
# rho_inf = 0.5    # asymptotic spectral radius   # NOTE: Jiayao used this
# alpha_m = 0.2
# alpha_f = 0.4
alpha_m = (2*rho_inf - 1)/(rho_inf + 1)
alpha_f = rho_inf/(rho_inf + 1)
gamma   = 0.5+alpha_f-alpha_m   # CHECK THIS!!
beta    = (gamma+0.5)**2/4.     # CHECK THIS!!


# Test and trial functions
# du = TrialFunction(state_function_space)
# u_ = TestFunction(state_function_space)
# Current (unknown) displacement
# u = Function(state_function_space, name="Displacement")
# Fields from previous time step (displacement, velocity, acceleration)
u_old = Function(state_function_space)
v_old = Function(state_function_space)
a_old = Function(state_function_space)


# Generalized alpha method averaging
def linear_combination(start, stop, coefficient):
    # return coefficient*start + (1-coefficient)*stop
    return (1-coefficient)*start + coefficient*stop

# Update formula for acceleration
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_a(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)

    # print('u', u[:])
    # # np.savetxt('u.txt', u[:])
    # # np.savetxt()
    # print('u_old', u_old[:])
    # print('dt_', dt_)
    # print('v_old', v_old[:])
    # print('a_old', a_old[:])
    return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old

# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)
    return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

def update_fields(u, u_old, v_old, a_old):
    """Update fields at the end of each time step.""" 

    # # Get vectors (references)
    # u_vec, u0_vec  = u.vector, u_old.vector
    # v0_vec, a0_vec = v_old.vector, a_old.vector 

    u_vec = getFuncArray(u)
    u_old_vec = getFuncArray(u_old)
    v_old_vec = getFuncArray(v_old)
    a_old_vec = getFuncArray(a_old)


    # use update functions using vector arguments
    # a_new = update_a(u, u_old, v_old, a_old, ufl=True)
    # v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

    a_new_vec = update_a(u_vec, u_old_vec, v_old_vec, a_old_vec, ufl=True)
    v_new_vec = update_v(a_new_vec, u_old_vec, v_old_vec, a_old_vec, ufl=True)

    # print('a_new_vec', a_new_vec)
    # print('v_new_vec', v_new_vec)


    # Update (u_old <- u)
    # v_old.vector[:], a_old.vector[:] = v_vec, a_vec
    # u_old.vector[:] = u.vector
    setFuncArray(v_old, v_new_vec)
    setFuncArray(a_old, a_new_vec)
    setFuncArray(u_old, u_vec)

    # print('---------------------------')
    # print(v_old)
    # print(v_new)
    # print('---------------------------')
    # exit()

    # v_old.interpolate(v_new)
    # a_old.interpolate(a_new)
    # u_old.interpolate(u)

    # print('u', u_old.x.array)
    # print('v', v_old.x.array)
    # print('a', a_old.x.array)


a_new = update_a(u, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

a_weighted = linear_combination(a_new, a_old, alpha_m)  # accelerations at n+1-alpha_m
v_weighted = linear_combination(v_new, v_old, alpha_f)  # velocities at n+1-alpha_f
u_weighted = linear_combination(u, u_old, alpha_f)      # states at n+1-alpha_f

# Define deformation gradient and Green Deformation tensor
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
# deformation_gradient = I + grad(u)       # Deformation gradient
deformation_gradient = I + gradx(u_weighted,u_hat)       # Deformation gradient
C = deformation_gradient.T*deformation_gradient                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
jacobian  = det(deformation_gradient)
# Elasticity parameters
# # E, nu = 1.e6, 0.3
# # E, nu = 2.e2, 0.3
# # E, nu = 10., 0.3
# E_dragon = 1.5e6  # Young's modulus of dragon skin 30 silicone rubber (Pa) NOTE: Github copilot made this up (is it right?)
# E_dragon = 1.e5  # Young's modulus of dragon skin 30 silicone rubber (Pa) NOTE: penalized 100% modulus from data sheet
E_dragon = 6.e5  # Young's modulus of dragon skin 30 silicone rubber (Pa) NOTE: 100% modulus from data sheet
nu_dragon = 0.45  # Poisson's ratio of dragon skin 30 silicone rubber  NOTE: Github copilot made this up (is it right?)
# mu, lmbda = Constant(domain=mesh, c=E/(2*(1 + nu))), Constant(domain=mesh, c=E*nu/((1 + nu)*(1 - 2*nu)))
# E_fr4 = 3.7e9  # Young's modulus of FR-4 (Pa) NOTE: Github copilot made this up (is it right?)
E_fr4 = 3.7e8       # Penalized to account for holes in the material
nu_fr4 = 0.35  # Poisson's ratio of FR-4  NOTE: Github copilot made this up (is it right?)
# NOTE: May need ot penalize fr4 values due to mesh being thicker than 0.015 inches and there being holes in the material.

E = Function(material_properties_function_space)
nu = Function(material_properties_function_space)

centerline_tol = 1.5e-3
# centerline_tol = 1.e-3    # fine mesh
# centerline_tol = 3.e-3    # OLD GEOMETRY
# off_centerline_tol = 2.9e-3
# centerline_tol = 3.001e-3
def off_centerline(x):
    return np.invert(on_centerline(x))
def on_centerline(x):
    return np.abs(x[2]) <= centerline_tol
def on_front_wall(x):
    # return np.isclose(x[0], -0.135 ,atol=0.015)
    # return np.isclose(x[0], -0.135 ,atol=0.009)
    # return np.isclose(x[0], -0.15 ,atol=0.007)    # Old geometry
    return np.isclose(x[0], 0. ,atol=0.006)
def on_back_wall(x):
    # return np.isclose(x[0], -0.335 ,atol=0.015)
    # return np.isclose(x[0], -0.335 ,atol=0.009)
    # return np.isclose(x[0], -0.32 ,atol=0.007)    # Old geometry
    return np.isclose(x[0], 0.08 ,atol=0.006)

# cells_off_centerline = locate_entities(mesh, mesh.topology.dim, off_centerline)
cells_on_centerline = locate_entities(mesh, mesh.topology.dim, on_centerline)
cells_on_front_wall = locate_entities(mesh, mesh.topology.dim, on_front_wall)
cells_on_back_wall = locate_entities(mesh, mesh.topology.dim, on_back_wall)

# E.x.array[cells_off_centerline] = np.full_like(cells_off_centerline, E_dragon, dtype=float)
# E.x.array[cells_on_centerline] = np.full_like(cells_on_centerline, E_fr4, dtype=float)

# nu.x.array[cells_off_centerline] = np.full_like(cells_off_centerline, nu_dragon, dtype=float)
# nu.x.array[cells_on_centerline] = np.full_like(cells_on_centerline, nu_fr4, dtype=float)

E.x.array[:] = E_dragon
E.x.array[cells_on_centerline] = np.full_like(cells_on_centerline, E_fr4, dtype=float)
E.x.array[cells_on_front_wall] = np.full_like(cells_on_front_wall, E_fr4, dtype=float)
E.x.array[cells_on_back_wall] = np.full_like(cells_on_back_wall, E_fr4, dtype=float)

nu.x.array[:] = nu_dragon
nu.x.array[cells_on_centerline] = np.full_like(cells_on_centerline, nu_fr4, dtype=float)
nu.x.array[cells_on_front_wall] = np.full_like(cells_on_front_wall, nu_fr4, dtype=float)
nu.x.array[cells_on_back_wall] = np.full_like(cells_on_back_wall, nu_fr4, dtype=float)


mu = E/(2*(1 + nu))
lmbda = E*nu/((1 + nu)*(1 - 2*nu))

# Dynamics parameters
density_not_fenics = 1080.  # density of dragon skin 30 silicone rubber (kg/m^3) NOTE: From the data sheet
# density = 1.1e6
density = Constant(domain=mesh, c=density_not_fenics)

# Rayleigh damping coefficients
eta_m = Constant(domain=mesh, c=0.)
# eta_m = Constant(domain=mesh, c=5.e0)
# eta_m = Constant(domain=mesh, c=8.e0)   #Seemed to match visually with gravity, but too much oscillation in pressurized setting.
# eta_m = Constant(domain=mesh, c=16.e0)    # Previous was calculated based on frequency, but stiffness term is missing, so multiply by 2.
# eta_k = Constant(domain=mesh, c=0.)
eta_k = Constant(domain=mesh, c=6.)
# eta_k = Constant(domain=mesh, c=8.e0)

# Stored strain energy density (compressible neo-Hookean model)
def elastic_term(u_weighted):
    psi = (mu/2)*(Ic - 3) - mu*ln(jacobian) + (lmbda/2)*(ln(jacobian))**2
    # Total potential energy
    Pi = psi*J(u_hat)*dx - dot(B, u_weighted)*J(u_hat)*dx - dot(T, u_weighted)*ds
    # Compute first variation of Pi (directional derivative about u in the direction of v)
    elastic_forces = 1/alpha_f*derivative(Pi, u, v)     # This is a force term
    # NOTE: Want derivative wrt u_weighted, but not possible in FEniCSx, so take derivative wrt u instead and multiply by 1/alpha_f to cancel chain rule
    # NOTE: This is actually derivative of energy, so it's more like an internal elastic forces term.
    # stiffness_jacobian = 1/alpha_f*derivative(elastic_forces, u, v)  # NOTE: Don't need this for damping because weak form
    return elastic_forces
elastic_forces = elastic_term(u_weighted)

# # Compute Jacobian of F
# J = derivative(F, u, du)

# output_name = 'dPE_du'
# output_form = F         # residual is F == 0 for equilibrium (minimization of potential energy)

'''
3. Define the boundary conditions
'''
############ Strongly enforced boundary conditions #############
ubc_1 = Function(state_function_space)
ubc_1.vector.set(0.)
front_wall_dofs = locate_dofs_geometrical(state_function_space,
                            # lambda x: np.isclose(x[0], -0.335 ,atol=1e-6))  # Want no displacement at x=0
                            # lambda x: np.isclose(x[0], -0.135 ,atol=1e-6))  # Want no displacement at x=0 # Old geometry
                            # lambda x: np.logical_and(
                            #      np.isclose(x[0], 0. ,atol=1e-6),
                            #      np.isclose(x[2], 0. ,atol=1e-6)))  # Want no displacement at x=0
                            # lambda x: np.isclose(x[0], 0. ,atol=1e-6))
                            lambda x: np.isclose(x[0], 0.08 ,atol=1e-6))

midpoint_dof = locate_dofs_geometrical(state_function_space,
                            # lambda x: np.isclose(x[0], -0.335 ,atol=1e-6))  # Want no displacement at x=0
                            # lambda x: np.isclose(x[0], -0.135 ,atol=1e-6))  # Want no displacement at x=0 # Old geometry
                            lambda x: np.logical_and(
                                 np.logical_and(
                                 np.isclose(x[0], 0.04 ,atol=1e-6),
                                 np.isclose(x[1], 0. ,atol=1e-6)),
                                 np.isclose(x[2], 0. ,atol=1e-6)))  # Want no displacement at x=0

# locate_BC2 = locate_dofs_geometrical(state_function_space,
#                             lambda x: np.isclose(x[0], -0.335 ,atol=1e-6))  # Want no displacement at x=0
                            # lambda x: np.isclose(x[0], -0.135 ,atol=1e-6))  # Want no displacement at x=0

fea.add_strong_bc(ubc_1, [front_wall_dofs])
# fea.add_strong_bc(ubc_1, [midpoint_dof])

# centerline_dofs = locate_dofs_geometrical(state_function_space,
#                             lambda x: np.isclose(x[2], 0. ,atol=1e-6))  # Want no x displacement at z=0   # fish centerline
# fea.add_strong_bc(0., [centerline_dofs], state_function_space.sub(0))   # TODO: NOT WORKING AS INTENDED

# fea.add_strong_bc(ubc_1, locate_BC_list_1, state_function_space)
# fea.add_strong_bc(ubc_2, locate_BC_list_2, state_function_space)
body_force = Function(state_function_space)
# f_d = 1.
# f = ufl.as_vector([0,0,-f_d]) # Body force per unit surface area
# body_force.interpolate(f)
# project(f, body_force)
# body_force.vector.set((1., 0., 0.))

# define dynamic resudual form
# gratvity_term = dot(body_force,v)*dx

def intertial_term(a_weighted):
    inertial_forces = inner(density*a_weighted,v)*J(u_hat)*dx
    return inertial_forces
inertial_forces = intertial_term(a_weighted)

mass_damping_term = eta_m*intertial_term(v_weighted)
# stiffness_damping_term = eta_k*dot(density*grad(u),grad(v))*dx    # This works in linear setting
stiffness_damping_term = eta_k*elastic_term(v_weighted)   # This seems weird
damping_forces = mass_damping_term + stiffness_damping_term
# NOTE: Not clear how to apply stiffness term in rayleigh damping with nonlinearity

pressure_input = Function(input_function_space)
# pressure_input = Function(test_function_space)
# locate_left_chamber = locate_dofs_geometrical(input_function_space, lambda x: x[2] > 0)
# locate_right_chamber = locate_dofs_geometrical(input_function_space, lambda x: x[2] < 0)

# # fish_width = 0.05 # got from paraview
# # fish_height = 0.08333333333333333     # got from paraview
# fish_width = 0.0465     # played around with to get everything except the outer boundary of the fish skin
# fish_height = 0.075     # played around with to get everything except the outer boundary of the fish skin
# # locate_left_chamber = locate_dofs_geometrical(input_function_space, lambda x: x[2] > 0 and (x[1]**2/fish_height**2 + x[2]**2/fish_width**2 < 1.))     # idea: try x: x[2] > 0 and x[2] < some ellipse
# # locate_right_chamber = locate_dofs_geometrical(input_function_space, lambda x: x[2] < 0 and (x[1]**2/fish_height**2 + x[2]**2/fish_width**2 < 1.))     # idea: try x: x[2] > 0 and x[2] < some ellipse)

# locate_left_chamber = locate_dofs_geometrical(input_function_space, 
#                                             #   lambda x: #np.logical_and(np.logical_and(x[0] <= -0.15, x[0] >= -0.32),
#                                               lambda x: np.logical_and(np.logical_and(x[0] <= -0.148, x[0] >= -0.322),
#                                                   np.logical_and(x[2] > 0, 
#                                                                        (x[1]**2/fish_height**2 + x[2]**2/fish_width**2 < 1.))))     # idea: try x: x[2] > 0 and x[2] < some ellipse
# locate_right_chamber = locate_dofs_geometrical(input_function_space, 
#                                             #    lambda x: #np.logical_and(np.logical_and(x[0] <= -0.15, x[0] >= -0.32),
#                                               lambda x: np.logical_and(np.logical_and(x[0] <= -0.148, x[0] >= -0.322),
#                                                    np.logical_and(x[2] < 0, 
#                                                                         (x[1]**2/fish_height**2 + x[2]**2/fish_width**2 < 1.))))     # idea: try x: x[2] > 0 and x[2] < some ellipse)
# # NOTE: Alternative idea is to dot the traction term with [1,0,0] to only apply the x-component of normal tractions.

# ds_C = Measure(“ds”, domain=mesh, subdomain_marker=mf)
# ds_LC = Measure("ds", domain=mesh, subdomain_data=left_chamber_facet_tags)
# ds_RC = Measure("ds", domain=mesh, subdomain_data=right_chamber_facet_tags)
# for i in range(0,10000000):
#     please_dont_be_empty = left_chamber_facet_tags.find(i)
#     please_have_dofs = locate_dofs_topological(input_function_space, mesh.topology.dim - 1, please_dont_be_empty)
#     print(i)
#     if len(please_have_dofs):
#         print(i)
#         print(please_have_dofs)
#         exit()

left_chamber_facets = left_chamber_facet_tags.find(507)     # 677 is what GMSH GUI assigned it (see in tools --> visibility)
left_chamber_facet_dofs = locate_dofs_topological(input_function_space, mesh.topology.dim - 1, left_chamber_facets)

right_chamber_facets = right_chamber_facet_tags.find(509)     # 678 is what GMSH GUI assigned it (see in tools --> visibility)
right_chamber_facet_dofs = locate_dofs_topological(input_function_space, mesh.topology.dim - 1, right_chamber_facets)
# NOTE: ONLY WORKS WHEN FUNCTION SPACE IS CG2

# exit()

# print(left_chamber_facet_dofs)
# print(right_chamber_facet_dofs)
# exit()

# def left_chamber_pressure(t):
#     if t < t_end/2:
#         return -(1.e4 - 1.e4*np.exp(-2*t/(t_end/2)))
#     else:
#         return -(-1.e2 + (1.e4+1.e2)*np.exp(-2*(t-t_end/2)/(t_end/2)) - 1.e4*np.exp(-2*t/(t_end/2)))

# def right_chamber_pressure(t):
#     if t < t_end/2:
#         return -(-1.e2 + 1.e2*np.exp(-2*t/(t_end/2)))
#     else:
#         return -(1.e4 - (1.e4+1.e2)*np.exp(-2*(t-t_end/2)/(t_end/2))  + 1.e2*np.exp(-2*t/(t_end/2)))


def compute_chamber_pressure_function(t, pressure_inputs, time_constant, p0, evaluation_t):
    if len(t) != len(pressure_inputs):
        raise ValueError('t and pressure_inputs must have the same length')
    
    # total_t = np.zeros((len(t),len(evaluation_t)))
    chamber_pressure = np.zeros((len(evaluation_t),))
    index = 0
    for i in range(len(t)):   # For each stroke
        # if i < len(t)-1:
        #     evaluation_t = np.linspace(t[i], t[i+1], num_evaluation_points)
        # else:
        #     evaluation_t = np.linspace(t[i], t_end, num_evaluation_points)
        # total_t[i] = evaluation_t
        j = 0
        if i < len(t)-1:
            while evaluation_t[index] < t[i+1]:
                if i == 0:
                    chamber_pressure[index] = stroke_pressure(evaluation_t[index], p0, pressure_inputs[i], time_constant)
                else:
                    chamber_pressure[index] = stroke_pressure(evaluation_t[index] - t[i],
                                                                        chamber_pressure[index-j-1],
                                                                        pressure_inputs[i],
                                                                        time_constant)
                j += 1
                index += 1
        else:
            while index < len(evaluation_t):
                if i == 0:
                    chamber_pressure[index] = stroke_pressure(evaluation_t[index], p0, pressure_inputs[i], time_constant)
                else:
                    chamber_pressure[index] = stroke_pressure(evaluation_t[index] - t[i],
                                                                        chamber_pressure[index-j-1],
                                                                        pressure_inputs[i],
                                                                        time_constant)
                index += 1
                j += 1
        i += 1
            
    return chamber_pressure

def stroke_pressure(t, initial_pressure, final_pressure, time_constant):
    return final_pressure \
        - (final_pressure - initial_pressure)*np.exp(-time_constant*t)\
        # + initial_pressure
    # final pressure is what it should go to. There is a decaying exponential term that goes to zero. Initial pressure is to ensure continuity.
    # t is the time WITHIN THE STROKE. This means that the time previous to the stroke is subtracted off before being passed into this function.

left_chamber_inputs = []
right_chamber_inputs = []
for stroke_index in range(num_strokes):
    if stroke_index % 2 == 0:
        left_chamber_inputs.append(pump_max_pressure)
        right_chamber_inputs.append(pump_vacuum_pressure)
        # left_chamber_inputs.append(pump_vacuum_pressure)
        # right_chamber_inputs.append(pump_max_pressure)
    else:
        left_chamber_inputs.append(pump_vacuum_pressure)
        right_chamber_inputs.append(pump_max_pressure)
        # left_chamber_inputs.append(pump_max_pressure)
        # right_chamber_inputs.append(pump_vacuum_pressure)
# left_chamber_inputs = [pump_max_pressure, pump_vacuum_pressure, pump_max_pressure, pump_vacuum_pressure, pump_max_pressure]
# right_chamber_inputs = [pump_vacuum_pressure, pump_max_pressure, pump_vacuum_pressure, pump_max_pressure, pump_vacuum_pressure]

t_pressure_inputs = np.linspace(0, t_end, int(num_strokes))
t_pressure_inputs[1:] = t_pressure_inputs[1:] - stroke_period/2
left_chamber_pressure = compute_chamber_pressure_function(t_pressure_inputs, left_chamber_inputs, time_constant, p0, t)
right_chamber_pressure = compute_chamber_pressure_function(t_pressure_inputs, right_chamber_inputs, time_constant, p0, t)

# import matplotlib.pyplot as plt
# t_test = np.linspace(0, t_end, 100)
# plt.plot(t_test, left_chamber_pressure(t_test))
# plt.plot(t_test, right_chamber_pressure(t_test))
# plt.show()

# import matplotlib.pyplot as plt
# t_test = np.linspace(0, t_end, 100)
# left_chamber_pressure_vector = np.vectorize(left_chamber_pressure)
# right_chamber_pressure_vector = np.vectorize(right_chamber_pressure)
# plt.plot(t_test, left_chamber_pressure_vector(t_test))
# plt.plot(t_test, right_chamber_pressure_vector(t_test))
# plt.show()

# plt.plot(t_test, left_chamber_pressure(t_test))
# plt.plot(t_test, right_chamber_pressure(t_test))
# plt.show()

# pressure_input.x.array[:] = 1.e6
# pressure_input.x.array[:] = 1.e3
# pressure_input.x.array[locate_left_chamber] = -4.e3
# pressure_input.x.array[locate_left_chamber] = -5.e3
# pressure_input.x.array[locate_left_chamber] = -1.e4
# pressure_input.x.array[locate_right_chamber] = 0.
# pressure_input.x.array[left_chamber_facet_dofs] = -left_chamber_pressure[0]
# pressure_input.x.array[right_chamber_facet_dofs] = -right_chamber_pressure[0]
pressure_input.x.array[left_chamber_facet_dofs] = left_chamber_pressure[-1]
pressure_input.x.array[right_chamber_facet_dofs] = right_chamber_pressure[-1]

# def projectLocalBasis(PATH):
#     VT = VectorFunctionSpace(mesh, ("CG", 1), dim=3)
#         #local frame looks good when exported one at a time, paraview doing something funny when all 3 basis vectors included
#     with XDMFFile(MPI.COMM_SELF, PATH+"a0.xdmf", "w") as xdmf:
#         xdmf.write_mesh(mesh)
#         e0 = Function(VT, name="e0")
#         # e0.interpolate(ufl.Expression(n, VT.element.interpolation_points()))
#         e0.interpolate(dolfin_fem.Expression(n, VT.element.interpolation_points()))
#         xdmf.write_function(e0)

# with XDMFFile(MPI.COMM_SELF, "examples/advanced_examples/temp/pressure_input.xdmf", "w") as xdmf:
with XDMFFile(MPI.COMM_SELF, output_path + "temp/pressure_input.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(pressure_input)

# with XDMFFile(MPI.COMM_SELF, "examples/advanced_examples/temp/E.xdmf", "w") as xdmf:
with XDMFFile(MPI.COMM_SELF, output_path + "temp/E.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(E)

# with XDMFFile(MPI.COMM_SELF, "examples/advanced_examples/temp/nu.xdmf", "w") as xdmf:
with XDMFFile(MPI.COMM_SELF, output_path + "temp/nu.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(nu)
    # exit()

# projectLocalBasis("examples/advanced_examples/temp/")

n = FacetNormal(mesh)
# transform normal and area element by Nanson's formula:
dsx_dsy_n_x = J(u_hat)*inv(F(u_hat).T)*n
# norm_dsx_dsy_n_x = ufl.sqrt(ufl.dot(dsx_dsy_n_x, dsx_dsy_n_x))

# internal_pressure_traction = pressure_input*v*n*dx
# internal_pressure_traction = pressure_input*dot(v,n)*dx
# internal_pressure_forces = pressure_input*dot(v,n)*ds
internal_pressure_forces = pressure_input*dot(v,dsx_dsy_n_x)*ds
# pressure_term = pressure_input*dot(v,n)
# internal_pressure_traction = pressure_term("+")*dS + pressure_term("-")*dS

# left_chamber_pressure_term = left_chamber_pressure[0]*dot(v,n)*ds_LC
# right_chamber_pressure_term = right_chamber_pressure[0]*dot(v,n)*ds_RC

# residual_form = strain_energy_term - body_term + intertial_term + damping_term - internal_pressure_traction
residual_form = inertial_forces + damping_forces + elastic_forces - internal_pressure_forces
# residual_form = strain_energy_term + intertial_term + damping_term - left_chamber_pressure_term# - right_chamber_pressure_term
# residual_form = strain_energy_term - body_term
# import ufl
# my_rhs = ufl.rhs(residual_form)

def static_solve(residual_form, u, ubc):
    # u_bar.assign(u + dt*du + 0.25*dt*dt*ddu)
    # u_bar = u + dt*du + 0.25*dt*dt*ddu

    # fea.solve(residual_form, u, ubc)
    # solveNonlinear(residual_form, u, ubc, solver="Newton", report=True, initialize=False)
    solveNonlinear(residual_form, u, ubc, solver="SNES", report=True, initialize=False)

    # np.savetxt('u_vec.txt', u.vector[:])

    # ddu_old.assign(ddu)
    # ddu.assign(4/(dt*dt)*(u - u_bar))
    # du.assign(du + 0.5*dt*(ddu + ddu_old))
    # ddu_old = ddu
    # ddu = 4/(dt*dt)*(u - u_bar)
    # du = du + 0.5*dt*(ddu + ddu_old)

# path = "temp"
xdmf_file = XDMFFile(comm, output_path+"u.xdmf", "w")
xdmf_file.write_mesh(mesh)

xdmf_file.write_function(u, t[0])


def dynamic_solve(residual_form, u, ubc, report=False):
    time_step_index = 0
    for time_step_index in range(len(t)-1):
        print(f't={t[time_step_index]:.3f}/{t_end:.3f}')
        print(f'time_step={time_step_index+1}/{t_steps}')
        left_chamber_pressure_input = linear_combination(-left_chamber_pressure[time_step_index+1], -left_chamber_pressure[time_step_index], alpha_f)
        right_chamber_pressure_input = linear_combination(-right_chamber_pressure[time_step_index+1], -right_chamber_pressure[time_step_index], alpha_f)

        pressure_input.x.array[left_chamber_facet_dofs] = left_chamber_pressure_input
        pressure_input.x.array[right_chamber_facet_dofs] = right_chamber_pressure_input

        # left_chamber_pressure_term = left_chamber_pressure[time_step_index]*dot(v,n)*ds_LC
        # right_chamber_pressure_term = right_chamber_pressure[time_step_index]*dot(v,n)*ds_RC
        # residual_form = strain_energy_term + intertial_term + damping_term - left_chamber_pressure_term# - right_chamber_pressure_term
        static_solve(residual_form, u, ubc)

        update_fields(u, u_old, v_old, a_old)

        xdmf_file.write_function(u, t[time_step_index+1])
        time_step_index += 1


# make a function that performs the time stepping (dynamic solution)
# each time step performs the static solve using fea.solve()
fea.custom_solve = dynamic_solve

input_name = 'structural_mesh_displacements'
# fea.add_input(input_name, body_force)
fea.add_input(input_name, u_hat)
fea.add_state(name=state_name,
                function=u,
                residual_form=residual_form,
                arguments=[input_name])
# fea.add_output(name=output_name,
#                 type='scalar',
#                 form=output_form,
#                 arguments=[input_name,state_name])


'''
4. Set up the CSDL model
'''
fea.PDE_SOLVER = 'Newton'
# fea.REPORT = True
fea_model = FEAModel(fea=[fea])
# fea_model.create_input("{}".format(input_name),
#                             shape=fea.inputs_dict[input_name]['shape'],
#                             val=0.1*np.ones(fea.inputs_dict[input_name]['shape']) * 0.86)
fea_model.create_input(input_name,
                            shape=fea.inputs_dict[input_name]['shape'],
                            val=-0.1*np.ones(fea.inputs_dict[input_name]['shape']))

# fea_model.connect('f','u_state_model.f')
# fea_model.connect('f','l2_functional_output_model.f')
# fea_model.connect('u_state_model.u','l2_functional_output_model.u')

# fea_model.add_design_variable(input_name)
# fea_model.add_objective(output_name, scaler=1e5)


sim = Simulator(fea_model)
# sim = om_simulator(fea_model)
########### Test the forward solve ##############
body_force_input = Function(state_function_space)
# f_d = 10.
# f = ufl.as_vector([0,0,-f_d]) # Body force per unit surface area
# f_d = density_not_fenics*9.81
f_d = 0.
f = ufl.as_vector([0,0,-f_d]) # Body force per unit surface area
# body_force.interpolate(f)
# project(f, body_force_input)

# sim[input_name] = getFuncArray(body_force_input)

sim.run()

xdmf_file = XDMFFile(comm, output_path+"u_hat.xdmf", "w")
xdmf_file.write_mesh(mesh)
xdmf_file.write_function(fea.inputs_dict[input_name]['function'])
print(u_hat.x.array[:])