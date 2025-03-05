import m3l
import csdl_alpha as csdl
from python_csdl_backend import Simulator
import numpy as np

from femo.fea.fea_dolfinx import XDMFFile, MPI, FEA, VectorFunctionSpace, Function, FunctionSpace, \
    TestFunction, Constant, grad, Identity, tr, det, inv, J, derivative, dx, ds, dot, solveNonlinear, \
    locate_entities, locate_dofs_geometrical, locate_dofs_topological, gradx, F, FacetNormal
import femo
from femo.csdl_alpha_opt.fea_model import FEAModel
from femo.csdl_alpha_opt.state_operation import StateOperation
from femo.csdl_alpha_opt.output_operation import OutputOperation
import dolfinx.fem as dolfin_fem
import argparse
from ufl import ln, pi

# class RoboticFishStaticStructuralModel:
#     def evaluate(self, mesh_displacements:m3l.Variable) -> m3l.Variable:
#         # self.name = f'robotic_fish_3d_finite_element_model'

#         self.arguments = {'mesh_displacements':mesh_displacements}

#         # structural_displacements = m3l.Variable(name='structural_displacements', shape=(10907*3,), operation=self)
#         structural_displacements = m3l.Variable(name='structural_displacements', shape=(mesh_displacements.value.size,), operation=self)
#         # NOTE: SHAPE HARDCODED FOR MESH FOR NOW
#         operation_csdl = construct_csdl_model(record=False)
        
#         # create csdl model for in-line evaluations
#         sim = Simulator(operation_csdl)
#         sim['mesh_displacements'] = mesh_displacements.value
#         sim.run()
#         structural_displacements.value = sim['structural_displacements']

#         # with XDMFFile(MPI.COMM_SELF, output_path + "mesh_displacements.xdmf", "w") as xdmf:
#         #     xdmf.write_mesh(mesh)
#         #     xdmf.write_function(u_hat)

#         return structural_displacements




def robotic_fish_static_structural_model_front_fixed(mesh_displacements:csdl.Variable, pressure_input_coefficients:csdl.Variable) -> tuple[csdl.Variable,csdl.Variable]:
    file_path = 'examples/advanced_examples/robotic_fish/meshes/'
    output_path = "examples/advanced_examples/robotic_fish/temp/"
    # file_path = 'meshes/'

    # mesh_name = 'module_v1_fine'
    # mesh_name = 'module_v1'
    # mesh_name = 'module_v1_refined'
    # mesh_name = 'module_v1_refined_2'
    # mesh_name = 'module_v1_refined_3'
    # mesh_name = 'module_v1_refinement_study_20mm'
    mesh_name = 'module_v1_refinement_study_2point5mm'

    with XDMFFile(MPI.COMM_WORLD, file_path + mesh_name + ".xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    # print(mesh.geometry.input_global_indices)
    # import pickle
    # file_name = file_path + mesh_name + "_fenics_mesh_indices.pickle"
    # with open(file_name, 'wb+') as handle:
    #     pickle.dump(mesh.geometry.input_global_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)


    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    with XDMFFile(MPI.COMM_WORLD, file_path + mesh_name + "_left_chamber_inner_surfaces.xdmf", "r") as xdmf:
        left_chamber_facet_tags = xdmf.read_meshtags(mesh, name="Grid")

    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    with XDMFFile(MPI.COMM_WORLD, file_path + mesh_name + "_right_chamber_inner_surfaces.xdmf", "r") as xdmf:
        right_chamber_facet_tags = xdmf.read_meshtags(mesh, name="Grid")

    fea = FEA(mesh)
    # Record the function evaluations during optimization process
    record = True
    fea.record = record  # TODO: Should this be an argument? It was before.

    mesh = fea.mesh

    parameterization_displacements_function_space = VectorFunctionSpace(mesh, ("CG", 1))
    u_hat = Function(parameterization_displacements_function_space)
    # pump_pressure_space = FunctionSpace(mesh, ('CG', 2))
    # pump_pressure_function = Function(pump_pressure_space)

    # pump_max_pressure = pump_pressure_function.x.array[0]
    # pump_vacuum_pressure = -pump_max_pressure

    # Add state to the PDE problem:
    state_name = 'structural_displacements'
    state_function_space = VectorFunctionSpace(mesh, ('CG', 1))
    # state_function_space = VectorFunctionSpace(mesh, ('CG', 2))
    # input_function_space = FunctionSpace(mesh, ('DG', 0))
    input_function_space = FunctionSpace(mesh, ('CG', 2))   # For some reason, finding the dofs only works if the input function space is CG2
    material_properties_function_space = FunctionSpace(mesh, ('DG', 0))
    u = Function(state_function_space)
    v = TestFunction(state_function_space)
    B = Constant(domain=mesh, c=(0.0, 0.0, 0.0))  # Body force per unit volume
    T = Constant(domain=mesh, c=(0.0, 0.0, 0.0))  # Traction force on the boundary

    u_last_load_step = Function(state_function_space)

    # pump_max_pressure = 0.
    # pump_max_pressure = 1.
    # pump_max_pressure = 2.e2
    # pump_max_pressure = 1.e3        # Good for softer fine mesh (optimization)? NOTE: Still haven't gotten robustness through opt for fine mesh
    # pump_max_pressure = 5.e3      # Good for softer course mesh (optimization or fine model evaluation)
    # pump_max_pressure = 8.e3
    # pump_max_pressure = 1.e4    # Was using this a lot
    # pump_max_pressure = 1.5e4    # Was using this a lot
    # pump_max_pressure = 2.e4
    # pump_max_pressure = 3.e4
    # pump_max_pressure = 3.5e4
    # pump_max_pressure = 4.e4
    # pump_max_pressure = 5.e4
    # pump_max_pressure = 8.e4
    # pump_max_pressure = 1.e5
    # pump_max_pressure = 2.e5
    # pump_max_pressure = 3.e5
    # pump_max_pressure = 4.e5
    # pump_max_pressure = 5.e5
    # pump_max_pressure = 6.e5
    # pump_max_pressure = 1.e6
    # pump_max_pressure = 1.e7
    # pump_vacuum_pressure = -pump_max_pressure
    # pump_vacuum_pressure = 0.
    # pump_vacuum_pressure = -2.e2
    # pump_vacuum_pressure = -1.e3
    # pump_vacuum_pressure = -5.e3
    # pump_vacuum_pressure = -8.e3
    # pump_vacuum_pressure = -1.e4
    # pump_vacuum_pressure = -2.e4
    # pump_vacuum_pressure = -3.e4
    # pump_vacuum_pressure = -4.e4
    # pump_vacuum_pressure = -5.e4
    # pump_vacuum_pressure = -3.e5


    # Define deformation gradient and Green Deformation tensor
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    # deformation_gradient = I + grad(u)       # Deformation gradient
    deformation_gradient = I + gradx(u,u_hat)       # Deformation gradient
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
    # E_dragon = 6.e5  # Young's modulus of dragon skin 30 silicone rubber (Pa) NOTE: 100% modulus from data sheet
    # nu_dragon = 0.45  # Poisson's ratio of dragon skin 30 silicone rubber  NOTE: Github copilot made this up (is it right?)
    # Use dragon skin 10 which has a mu of 0.0425 MPa
    

    # Tuning notes: 1.e5 too soft, 6e5 too stiff
    # E_dragon = 3.5e5  # Parameter estimated to approximate experimental data
    # E_dragon = 2.e5  # Parameter estimated to approximate experimental data
    E_dragon = 2.2e5  # Parameter estimated to approximate experimental data    # THIS LOOKS GOOD   (just doesn't capture geometric softening)
    # E_dragon = 1.5e5
    # E_dragon = 6.e5  # Parameter estimated to approximate experimental data
    nu_dragon = 0.499  # Set to make the material effectively incompressible    # THIS LOOKS GOOD   (just doesn't capture geometric softening)
    # nu_dragon = 0.4999
    # nu_dragon = 0.4  # Set to make the material effectively incompressible
    # nu_dragon = 0.3  # Set to make the material effectively incompressible
    
    # E_dragon = 6.33e4  # Young's modulus of dragon skin 10 silicone rubber (Pa) NOTE: I guessed this to match the mu value (given nu)
    # nu_dragon = 0.49  # Poisson's ratio of dragon skin 10 silicone rubber  NOTE: I guessed this to match the mu value and make it mostly incompressible
    
    # E_dragon = 6.15e4  # Young's modulus of dragon skin 10 silicone rubber (Pa) NOTE: I guessed this to match the mu value (given nu)
    # nu_dragon = 0.45  # Poisson's ratio of dragon skin 10 silicone rubber  NOTE: I guessed this to match the mu value and make it mostly incompressible
    
    # E_dragon = 6.2e4  # Young's modulus of dragon skin 10 silicone rubber (Pa) NOTE: I guessed this to match the mu value (given nu)
    # nu_dragon = 0.48  # Poisson's ratio of dragon skin 10 silicone rubber  NOTE: I guessed this to match the mu value and make it mostly incompressible

    # mu, lmbda = Constant(domain=mesh, c=E/(2*(1 + nu))), Constant(domain=mesh, c=E*nu/((1 + nu)*(1 - 2*nu)))
    # E_fr4 = 3.7e9  # Young's modulus of FR-4 (Pa) NOTE: Github copilot made this up (is it right?)
    # E_fr4 = 3.7e8       # Penalized to account for holes in the material and mesh being thicker
    # E_fr4 = 3.7e7       # Penalized to account for holes in the material and mesh being thicker
    E_fr4 = 3.7e6       # Actual value according to wikipedia BEEN USING THIS
    # E_fr4 = 1.e6       # Penalized to account for holes and mesh being thicker
    # nu_fr4 = 0.35  # Poisson's ratio of FR-4  NOTE: Github copilot made this up (is it right?)
    nu_fr4 = 0.12  # Wikipedia value
    # NOTE: May need ot penalize fr4 values due to mesh being thicker than 0.015 inches and there being holes in the material.

    E = Function(material_properties_function_space)
    nu = Function(material_properties_function_space)

    if mesh_name == 'module_v1_fine':
        centerline_tol = 1.e-3    # fine mesh
    elif mesh_name == 'module_v1':
        centerline_tol = 3.e-3    # coarse mesh
    elif mesh_name == 'module_v1_refined':
        centerline_tol = 1.e-3
    elif mesh_name == 'module_v1_refined_2':
        centerline_tol = 1.e-3
    elif 'module_v1_refine' in mesh_name:
        centerline_tol = 1.e-3
    else:
        raise Exception('SET A CENTERLINE TOLERANCE FOR THIS MESH')
    # centerline_tol = 1.5e-3     # Coarse mesh
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


    # Stored strain energy density (compressible neo-Hookean model)
    def elastic_term(u_weighted):
        psi = (mu/2)*(Ic - 3) - mu*ln(jacobian) + (lmbda/2)*(ln(jacobian))**2
        # Total potential energy
        Pi = psi*J(u_hat)*dx # - dot(B, u_weighted)*J(u_hat)*dx - dot(T, u_weighted)*ds
        # Compute first variation of Pi (directional derivative about u in the direction of v)
        elastic_forces = derivative(Pi, u, v)     # This is a force term
        # NOTE: Want derivative wrt u_weighted, but not possible in FEniCSx, so take derivative wrt u instead and multiply by 1/alpha_f to cancel chain rule
        # NOTE: This is actually derivative of energy, so it's more like an internal elastic forces term.
        # stiffness_jacobian = 1/alpha_f*derivative(elastic_forces, u, v)  # NOTE: Don't need this for damping because weak form
        return elastic_forces
    elastic_forces = elastic_term(u)


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

    # midplane_dofs = locate_dofs_geometrical(state_function_space,
    #                             # lambda x: np.isclose(x[0], 0.04 ,atol=1e-6) )# Want no displacement at x=0
    #                             lambda x: np.isclose(x[0], 0.04 ,atol=1e-4) )# Want no displacement at x=0
    
    # midplane_dofs = locate_dofs_geometrical(state_function_space,
    #                             lambda x: np.logical_and(
    #                                 np.isclose(x[0], 0.04 ,atol=1e-6),
    #                                 np.isclose(x[2], 0. ,atol=1e-2)))  # Want no displacement at x=0
    
    # midplane_dofs = locate_dofs_geometrical((state_function_space.sub(0), state_function_space.sub(0).collapse()[0]),
    #                                         lambda x: np.isclose(x[0], 0.04 ,atol=1e-6))# Want no x displacement at x=0
    
    # midpoint_dof = locate_dofs_geometrical(state_function_space,
    #                             lambda x: np.logical_and(
    #                                 np.logical_and(
    #                                 np.isclose(x[0], 0.04 ,atol=1e-6),
    #                                 np.isclose(x[1], 0. ,atol=1e-6)),
    #                                 np.isclose(x[2], 0. ,atol=1e-6)))  # Want no displacement at x=0
    
    # print('midplane_dofs', midplane_dofs)
    # print('midpoint_dof', midpoint_dof)
    # exit()

    fea.add_strong_bc(ubc_1, [front_wall_dofs])
    # fea.add_strong_bc(ubc_1, [midplane_dofs])
    # fea.add_strong_bc(ubc_1, [list(midplane_dofs[1])])


    pressure_input = Function(input_function_space)

    # left_chamber_facets = left_chamber_facet_tags.find(507)     # module_v1: 507 is what GMSH GUI assigned it (see in tools --> visibility)
    left_chamber_facets = left_chamber_facet_tags.find(508)     # module_v1_refined_2: 508 is what GMSH GUI assigned it (see in tools --> visibility)
    left_chamber_facet_dofs = locate_dofs_topological(input_function_space, mesh.topology.dim - 1, left_chamber_facets)

    # right_chamber_facets = right_chamber_facet_tags.find(509)     # module_v1: 509 is what GMSH GUI assigned it (see in tools --> visibility)
    right_chamber_facets = right_chamber_facet_tags.find(509)     # module_v1_refined_2: 509 is what GMSH GUI assigned it (see in tools --> visibility)
    right_chamber_facet_dofs = locate_dofs_topological(input_function_space, mesh.topology.dim - 1, right_chamber_facets)
    # NOTE: ONLY WORKS WHEN FUNCTION SPACE IS CG2

    # pump_max_pressure = pump_pressure_function.x.array[right_chamber_facet_dofs]
    # pump_vacuum_pressure = -pump_pressure_function.x.array[left_chamber_facet_dofs]

    # pressure_input.x.array[left_chamber_facet_dofs] = pump_vacuum_pressure
    # pressure_input.x.array[right_chamber_facet_dofs] = pump_max_pressure

    # with XDMFFile(MPI.COMM_SELF, "examples/advanced_examples/temp/pressure_input.xdmf", "w") as xdmf:
    with XDMFFile(MPI.COMM_SELF, output_path + "pressure_input.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(pressure_input)

    # with XDMFFile(MPI.COMM_SELF, "examples/advanced_examples/temp/E.xdmf", "w") as xdmf:
    with XDMFFile(MPI.COMM_SELF, output_path + "E.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(E)

    # with XDMFFile(MPI.COMM_SELF, "examples/advanced_examples/temp/nu.xdmf", "w") as xdmf:
    with XDMFFile(MPI.COMM_SELF, output_path + "nu.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(nu)
        # exit()


    n = FacetNormal(mesh)
    # transform normal and area element by Nanson's formula:
    F_test = I + grad(u_hat) + grad(u)
    J_test = det(F_test)

    dsx_dsy_n_x = J_test*inv(F_test.T)*n
    # dsx_dsy_n_x = J(u_hat)*inv(F(u_hat).T)*n
    # norm_dsx_dsy_n_x = ufl.sqrt(ufl.dot(dsx_dsy_n_x, dsx_dsy_n_x))

    # internal_pressure_forces = pressure_input*dot(v,n)*ds
    internal_pressure_forces = pressure_input*dot(v,dsx_dsy_n_x)*ds

    residual_form = elastic_forces - internal_pressure_forces

    displacement_along_normal = dot(u, dsx_dsy_n_x)
    applied_work = pressure_input*displacement_along_normal*ds
    # applied_work = (pressure_input*dot(dot(v,dsx_dsy_n_x), u))*ds
    # applied_work = (pressure_input.dot(u))*ds

    # xdmf_file = XDMFFile(comm, output_path+"displacements.xdmf", "w")
    # xdmf_file.write_mesh(mesh)

    # # num_load_steps = 1
    # # num_load_steps = 2
    # # num_load_steps = 6
    # num_load_steps = 50
    # # load_stepping_coefficient = 3.5
    # load_stepping_coefficient = 3
    # def static_solve(residual_form, u, ubc, report=False):
    #     # use load stepping to solve the problem
    #     for i in range(num_load_steps):
    #         # u_hat.x.array[:] = desired_u_hat_coefficients/(load_stepping_coefficient**(num_load_steps-i-1))

    #         pressure_input.x.array[left_chamber_facet_dofs] = pump_vacuum_pressure/(load_stepping_coefficient**(num_load_steps-i-1))
    #         pressure_input.x.array[right_chamber_facet_dofs] = pump_max_pressure/(load_stepping_coefficient**(num_load_steps-i-1))
    #         converged_reason = solveNonlinear(residual_form, u, ubc, solver="SNES", report=report, initialize=False)
                

    # load_stepping_coefficient = 3.5
    # NOTE: Keep u_old that stores last converged solution (for initialization after load step)
    def static_solve(residual_form, u, ubc, report=False):
        initialize = False
        # load_stepping_coefficient = 3
        # adaptivity_coefficient = 1.1

        # NOTE: Increase load stepping coefficient if initial convergence takes too long or if it always converges once it starts converging
        load_stepping_coefficient = 2.
        
        # NOTE: This is a subdividing coefficient. 
        # NOTE: Increase if it takes many passes to converge, decrease if it always converges after the first subdivision
        # adaptivity_coefficient = 3.
        adaptivity_coefficient = 2
        aggressively_adapt = True
        num_load_steps = 0  # always start with no load steps
        # use adaptive load stepping to solve the problem
        # pressure_input.x.array[left_chamber_facet_dofs] = pump_vacuum_pressure
        # pressure_input.x.array[right_chamber_facet_dofs] = pump_max_pressure
        converged_reason = solveNonlinear(residual_form, u, ubc, solver="SNES", report=report, initialize=False)

        if converged_reason > 0:
            return

        full_pressure = pressure_input.x.array[:].copy()
        while converged_reason < 0:
            # Apply load step
            initialize = True
            print("Applying load step")
            num_load_steps += 1

            # pressure_input.x.array[left_chamber_facet_dofs] = pump_vacuum_pressure/(load_stepping_coefficient**(num_load_steps))
            # pressure_input.x.array[right_chamber_facet_dofs] = pump_max_pressure/(load_stepping_coefficient**(num_load_steps))
            # pressure_input = pressure_input/(load_stepping_coefficient**(num_load_steps))
            pressure_input.x.array[:] = full_pressure/(load_stepping_coefficient**(num_load_steps))
            converged_reason = solveNonlinear(residual_form, u, ubc, solver="SNES", report=report, initialize=True)
            # NOTE: Initialize=True is means that if it fails, it throws out the previous state solution as initial guess (back to 0)

            if converged_reason > 0:
                initialize = False  # Don't reset initial guess to 0 once it starts to converge
                u_last_load_step.x.array[:] = u.x.array[:]  # Store last converged solution
                while num_load_steps >= 0:
                    
                    if converged_reason < 0: # If it fails, subdivide the load step
                        num_load_steps += 1 # take a step back, then subdivide
                        # num_load_steps = int(num_load_steps*np.log(load_stepping_coefficient)/np.log(load_stepping_coefficient**(1/adaptivity_coefficient))) + 1
                        if aggressively_adapt:
                            # Here we don't guarauntee the load is smaller than the last converged
                            # The idea is that we should be close enough to the last value to still get convergence though
                            # num_load_steps = int(num_load_steps*adaptivity_coefficient)
                            num_load_steps = int(num_load_steps*adaptivity_coefficient) - 1
                            # NOTE: With integer coefficients, +0 repeats last converged solution unnecessarily
                        else:
                            # Here we guarauntee the load is smaller than the last converged but requires more load steps that seem unnecessary
                            num_load_steps = int(num_load_steps*adaptivity_coefficient) + 1
                        load_stepping_coefficient**=(1/adaptivity_coefficient)
                        u.x.array[:] = u_last_load_step.x.array[:]  # Reset initial guess to last converged solution
                    else:
                        if num_load_steps == 0:
                            return
                        num_load_steps -= 1
                        u_last_load_step.x.array[:] = u.x.array[:]  # Store last converged solution

                    print('num load steps left: ', num_load_steps)
                    print('load stepping coefficient: ', load_stepping_coefficient)
                    print('divisor: ', load_stepping_coefficient**(num_load_steps))

                    # pressure_input.x.array[left_chamber_facet_dofs] = pump_vacuum_pressure/(load_stepping_coefficient**(num_load_steps))
                    # pressure_input.x.array[right_chamber_facet_dofs] = pump_max_pressure/(load_stepping_coefficient**(num_load_steps))
                    # pressure_input = pressure_input/(load_stepping_coefficient**(num_load_steps))
                    pressure_input.x.array[:] = full_pressure/(load_stepping_coefficient**(num_load_steps))
                    converged_reason = solveNonlinear(residual_form, u, ubc, solver="SNES", report=report, initialize=False)



        # xdmf_file.write_function(u)


    fea.custom_solve = static_solve

    fea.add_input('mesh_displacements', u_hat, record=record)
    # fea.add_input('pump_pressure', pump_pressure_function, record=record)
    fea.add_input('pressure_input', pressure_input, record=record)
    fea.add_state(name=state_name,
                    function=u,
                    residual_form=residual_form,
                    arguments=['mesh_displacements', 'pressure_input'])
                    # arguments=['mesh_displacements'])
    fea.add_output(name='applied_work',
                    type='scalar',
                    form=applied_work,
                    arguments=['mesh_displacements','pressure_input', state_name])
                    # arguments=['mesh_displacements', state_name])


    '''
    4. Set up the CSDL model
    '''
    fea.PDE_SOLVER = 'Newton'
    # fea.REPORT = True
    fea_model = FEAModel(fea=[fea])

    input_group = csdl.VariableGroup()
    input_group.mesh_displacements = mesh_displacements
    input_group.pressure_input = pressure_input_coefficients

    fea_output = fea_model.evaluate(input_group)
    structural_displacements = fea_output.structural_displacements
    applied_work = fea_output.applied_work
    return structural_displacements, applied_work