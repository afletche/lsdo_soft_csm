
import csdl
from python_csdl_backend import Simulator

from femo.fea.fea_dolfinx import *
from femo.csdl_opt.fea_model import FEAModel
from femo.csdl_opt.state_model import StateModel
from femo.csdl_opt.output_model import OutputModel
import dolfinx.fem as dolfin_fem
import argparse
from ufl import ln, pi

class RoboticFishStructuralResidualModel(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='robotic_fish_3d_finite_element_model')

    def compute(self):
        csdl_model = construct_csdl_model()
        return csdl_model

    def evaluate(self, mesh_displacements:m3l.Variable) -> tuple:
        # self.name = f'robotic_fish_3d_finite_element_model'

        self.arguments = {'mesh_displacements':mesh_displacements}

        structural_displacements = m3l.Variable(name='structural_displacements', shape=(10907*3), operation=self)
        # NOTE: SHAPE HARDCODED FOR MESH FOR NOW
        operation_csdl = self.compute()
        
        # create csdl model for in-line evaluations
        sim = Simulator(operation_csdl)
        sim['mesh_displacements'] = mesh_displacements.value
        sim.run()
        structural_displacements.value = sim['structural_displacements']

        return structural_displacements




def construct_csdl_model():
    file_path = 'examples/advanced_examples/robotic_fish/meshes/'
    output_path = "examples/advanced_examples/robotic_fish/temp/"
    # file_path = 'meshes/'

    with XDMFFile(MPI.COMM_WORLD, file_path + "module_v1.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    with XDMFFile(MPI.COMM_WORLD, file_path + "module_v1_left_chamber_inner_surfaces.xdmf", "r") as xdmf:
        left_chamber_facet_tags = xdmf.read_meshtags(mesh, name="Grid")

    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    with XDMFFile(MPI.COMM_WORLD, file_path + "module_v1_right_chamber_inner_surfaces.xdmf", "r") as xdmf:
        right_chamber_facet_tags = xdmf.read_meshtags(mesh, name="Grid")

    fea = FEA(mesh)
    # Record the function evaluations during optimization process
    fea.record = True

    mesh = fea.mesh

    parameterization_displacements_function_space = VectorFunctionSpace(mesh, ("CG", 1))
    u_hat = Function(parameterization_displacements_function_space)

    # Add state to the PDE problem:
    state_name = 'structural_displacements'
    state_function_space = VectorFunctionSpace(mesh, ('CG', 1))
    # input_function_space = FunctionSpace(mesh, ('DG', 0))
    input_function_space = FunctionSpace(mesh, ('CG', 2))   # For some reason, finding the dofs only works if the input function space is CG2
    material_properties_function_space = FunctionSpace(mesh, ('DG', 0))
    u = Function(state_function_space)
    v = TestFunction(state_function_space)
    B = Constant(domain=mesh, c=(0.0, 0.0, 0.0))  # Body force per unit volume
    T = Constant(domain=mesh, c=(0.0, 0.0, 0.0))  # Traction force on the boundary

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
    pump_vacuum_pressure = -3.e5
    actuation_frequency = 0.5 # Hz
    # actuation_frequency = 0.125 # Hz
    stroke_period = 1./actuation_frequency/2    # 2 strokes per cycle
    num_strokes = int(actuation_frequency*t_end)*2+1   # 2 strokes per cycle
    # time_constant = 8*actuation_frequency    # in reality this would be a constant from RC circuit analysis, but I want it to be nice for me.
    time_constant = 6*actuation_frequency    # in reality this would be a constant from RC circuit analysis, but I want it to be nice for me.
    # time_constant = 3*actuation_frequency    # in reality this would be a constant from RC circuit analysis, but I want it to be nice for me.

    rho_inf = 1.0    # asymptotic spectral radius   # NOTE: This is unstable for MBD simulations (alebraic constraints make this unstable)
    # rho_inf = 0.9    # asymptotic spectral radius   # NOTE: Apparently typical values are between 0.6 and 0.9. I want to try adding damping
    # rho_inf = 0.8    # asymptotic spectral radius   # NOTE: Currently unstable for rho_infinity = 0.9. Should try jacobian to get stiffness damping
    # rho_inf = 0.8 gives stability and better displacement, but still has high frequency oscillations
    # rho_inf = 0.7
    # rho_inf = 0.6    # asymptoti/c spectral radius   # NOTE: Apparently typical values are between 0.6 and 0.9. I want to try adding damping
    # I want to try adding some damping to account for the high frequencies induced from the high stiffness elements (I don't care about higher frequencies)
    # rho_inf = 0.5    # asymptotic spectral radius   # NOTE: Jiayao used this
    alpha_m = (2*rho_inf - 1)/(rho_inf + 1)
    alpha_f = rho_inf/(rho_inf + 1)
    gamma   = 0.5+alpha_f-alpha_m
    beta    = (gamma+0.5)**2/4.

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

        u_vec = getFuncArray(u)
        u_old_vec = getFuncArray(u_old)
        v_old_vec = getFuncArray(v_old)
        a_old_vec = getFuncArray(a_old)

        a_new_vec = update_a(u_vec, u_old_vec, v_old_vec, a_old_vec, ufl=True)
        v_new_vec = update_v(a_new_vec, u_old_vec, v_old_vec, a_old_vec, ufl=True)

        setFuncArray(v_old, v_new_vec)
        setFuncArray(a_old, a_new_vec)
        setFuncArray(u_old, u_vec)

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
        return np.isclose(x[0], 0.08 ,atol=0.006)
    def on_back_wall(x):
        # return np.isclose(x[0], -0.335 ,atol=0.015)
        # return np.isclose(x[0], -0.335 ,atol=0.009)
        # return np.isclose(x[0], -0.32 ,atol=0.007)    # Old geometry
        return np.isclose(x[0], 0. ,atol=0.006)

    # cells_off_centerline = locate_entities(mesh, mesh.topology.dim, off_centerline)
    cells_on_centerline = locate_entities(mesh, mesh.topology.dim, on_centerline)
    cells_on_front_wall = locate_entities(mesh, mesh.topology.dim, on_front_wall)
    cells_on_back_wall = locate_entities(mesh, mesh.topology.dim, on_back_wall)

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
    density = Constant(domain=mesh, c=density_not_fenics)

    # Rayleigh damping coefficients
    eta_m = Constant(domain=mesh, c=0.)
    # eta_m = Constant(domain=mesh, c=5.e0)
    # eta_m = Constant(domain=mesh, c=8.e0)   #Seemed to match visually with gravity, but too much oscillation in pressurized setting.
    # eta_m = Constant(domain=mesh, c=16.e0)    # Previous was calculated based on frequency, but stiffness term is missing, so multiply by 2.
    # eta_k = Constant(domain=mesh, c=0.)
    eta_k = Constant(domain=mesh, c=6.)     # Converted damping to stiffness damping because stiff centerline added high frequency oscillations
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


    def intertial_term(a_weighted):
        inertial_forces = inner(density*a_weighted,v)*J(u_hat)*dx
        return inertial_forces
    inertial_forces = intertial_term(a_weighted)

    mass_damping_term = eta_m*intertial_term(v_weighted)
    stiffness_damping_term = eta_k*elastic_term(v_weighted)   # This seems weird
    damping_forces = mass_damping_term + stiffness_damping_term

    pressure_input = Function(input_function_space)

    left_chamber_facets = left_chamber_facet_tags.find(507)     # 677 is what GMSH GUI assigned it (see in tools --> visibility)
    left_chamber_facet_dofs = locate_dofs_topological(input_function_space, mesh.topology.dim - 1, left_chamber_facets)

    right_chamber_facets = right_chamber_facet_tags.find(509)     # 678 is what GMSH GUI assigned it (see in tools --> visibility)
    right_chamber_facet_dofs = locate_dofs_topological(input_function_space, mesh.topology.dim - 1, right_chamber_facets)
    # NOTE: ONLY WORKS WHEN FUNCTION SPACE IS CG2

    def compute_chamber_pressure_function(t, pressure_inputs, time_constant, p0, evaluation_t):
        if len(t) != len(pressure_inputs):
            raise ValueError('t and pressure_inputs must have the same length')
        
        # total_t = np.zeros((len(t),len(evaluation_t)))
        chamber_pressure = np.zeros((len(evaluation_t),))
        index = 0
        for i in range(len(t)):   # For each stroke
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

    t_pressure_inputs = np.linspace(0, t_end, int(num_strokes))
    t_pressure_inputs[1:] = t_pressure_inputs[1:] - stroke_period/2
    left_chamber_pressure = compute_chamber_pressure_function(t_pressure_inputs, left_chamber_inputs, time_constant, p0, t)
    right_chamber_pressure = compute_chamber_pressure_function(t_pressure_inputs, right_chamber_inputs, time_constant, p0, t)

    # pressure_input.x.array[left_chamber_facet_dofs] = -left_chamber_pressure[0]
    # pressure_input.x.array[right_chamber_facet_dofs] = -right_chamber_pressure[0]
    pressure_input.x.array[left_chamber_facet_dofs] = left_chamber_pressure[-1]
    pressure_input.x.array[right_chamber_facet_dofs] = right_chamber_pressure[-1]

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

    # internal_pressure_forces = pressure_input*dot(v,n)*ds
    internal_pressure_forces = pressure_input*dot(v,dsx_dsy_n_x)*ds


    residual_form = inertial_forces + damping_forces + elastic_forces - internal_pressure_forces


    def static_solve(residual_form, u, ubc):
        solveNonlinear(residual_form, u, ubc, solver="SNES", report=True, initialize=False)


    # path = "temp"
    xdmf_file = XDMFFile(comm, output_path+"/u.xdmf", "w")
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

            static_solve(residual_form, u, ubc)
            update_fields(u, u_old, v_old, a_old)

            xdmf_file.write_function(u, t[time_step_index+1])
            time_step_index += 1

    fea.custom_solve = dynamic_solve

    input_name = 'mesh_displacements'
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
    fea_model.declare_variable(input_name,
                                shape=fea.inputs_dict[input_name]['shape'],
                                val=-0.1*np.ones(fea.inputs_dict[input_name]['shape']))
    
    return fea_model