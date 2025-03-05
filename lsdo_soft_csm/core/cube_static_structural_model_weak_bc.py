
import csdl_alpha as csdl

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




def cube_static_structural_model(mesh_displacements:csdl.Variable) -> csdl.Variable:
    file_path = 'examples/example_geometries/'
    output_path = "examples/example_geometries/temp/"

    mesh_name = 'cube_14_node_mesh'

    with XDMFFile(MPI.COMM_WORLD, file_path + mesh_name + ".xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    # print(mesh.geometry.input_global_indices)
    # import pickle
    # file_name = file_path + mesh_name + "_fenics_mesh_indices.pickle"
    # with open(file_name, 'wb+') as handle:
    #     pickle.dump(mesh.geometry.input_global_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)


    fea = FEA(mesh)
    # Record the function evaluations during optimization process
    record = True
    fea.record = record  # TODO: Should this be an argument? It was before.

    mesh = fea.mesh

    parameterization_displacements_function_space = VectorFunctionSpace(mesh, ("CG", 1))
    u_hat = Function(parameterization_displacements_function_space)

    # Add state to the PDE problem:
    state_name = 'structural_displacements'
    state_function_space = VectorFunctionSpace(mesh, ('CG', 1))
    # input_function_space = FunctionSpace(mesh, ('DG', 0))
    input_function_space = FunctionSpace(mesh, ('CG', 1))   # For some reason, finding the dofs only works if the input function space is CG2
    # input_function_space = FunctionSpace(mesh, ('CG', 2))   # For some reason, finding the dofs only works if the input function space is CG2
    material_properties_function_space = FunctionSpace(mesh, ('DG', 0))
    u = Function(state_function_space)
    v = TestFunction(state_function_space)
    B = Constant(domain=mesh, c=(0.0, 0.0, 0.0))  # Body force per unit volume
    T = Constant(domain=mesh, c=(0.0, 0.0, 0.0))  # Traction force on the boundary

    u_last_load_step = Function(state_function_space)

    pressure=-2e4


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
    E_dragon = 6.33e4  # Young's modulus of dragon skin 10 silicone rubber (Pa) NOTE: I guessed this to match the mu value (given nu)
    nu_dragon = 0.49  # Poisson's ratio of dragon skin 10 silicone rubber  NOTE: I guessed this to match the mu value and make it mostly incompressible
    

    E = Function(material_properties_function_space)
    nu = Function(material_properties_function_space)

    E.x.array[:] = E_dragon
    nu.x.array[:] = nu_dragon


    mu = E/(2*(1 + nu))
    lmbda = E*nu/((1 + nu)*(1 - 2*nu))


    # Stored strain energy density (compressible neo-Hookean model)
    def compute_elastic_term(u_weighted):
        psi = (mu/2)*(Ic - 3) - mu*ln(jacobian) + (lmbda/2)*(ln(jacobian))**2
        # Total potential energy
        Pi = psi*J(u_hat)*dx # - dot(B, u_weighted)*J(u_hat)*dx - dot(T, u_weighted)*ds
        # Compute first variation of Pi (directional derivative about u in the direction of v)
        elastic_forces = derivative(Pi, u, v)     # This is a force term

        return elastic_forces
    elastic_forces = compute_elastic_term(u)


    '''
    3. Define the boundary conditions
    '''
    ############ Strongly enforced boundary conditions #############
    ubc_1 = Function(state_function_space)
    ubc_1.vector.set(0.)
    front_wall_dofs = locate_dofs_geometrical(state_function_space, lambda x: np.isclose(x[0], -1.27 ,atol=1e-6))
    back_wall_dofs = locate_dofs_geometrical(state_function_space, lambda x: np.isclose(x[0], 1.27 ,atol=1e-6))

    fea.add_strong_bc(ubc_1, [front_wall_dofs])


    pressure_input = Function(input_function_space)
    pressure_input.x.array[back_wall_dofs] = pressure


    with XDMFFile(MPI.COMM_SELF, output_path + "pressure_input.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(pressure_input)


    n = FacetNormal(mesh)
    # transform normal and area element by Nanson's formula:
    dsx_dsy_n_x = J(u_hat)*inv(F(u_hat).T)*n
    # norm_dsx_dsy_n_x = ufl.sqrt(ufl.dot(dsx_dsy_n_x, dsx_dsy_n_x))

    # internal_pressure_forces = pressure_input*dot(v,n)*ds
    internal_pressure_forces = pressure_input*dot(v,dsx_dsy_n_x)*ds

    residual_form = elastic_forces - internal_pressure_forces

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
        pressure_input.x.array[back_wall_dofs] = pressure
        converged_reason = solveNonlinear(residual_form, u, ubc, solver="SNES", report=report, initialize=False)

        if converged_reason > 0:
            return

        while converged_reason < 0:
            # Apply load step
            initialize = True
            print("Applying load step")
            num_load_steps += 1

            pressure_input.x.array[back_wall_dofs] = pressure/(load_stepping_coefficient**(num_load_steps))
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

                    pressure_input.x.array[back_wall_dofs] = pressure/(load_stepping_coefficient**(num_load_steps))
                    converged_reason = solveNonlinear(residual_form, u, ubc, solver="SNES", report=report, initialize=False)



        # xdmf_file.write_function(u)


    fea.custom_solve = static_solve

    input_name = 'mesh_displacements'
    fea.add_input(input_name, u_hat, record=record)
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

    input_group = csdl.VariableGroup()
    input_group.mesh_displacements = mesh_displacements

    fea_output = fea_model.evaluate(input_group)
    structural_displacements = fea_output.structural_displacements
    return structural_displacements
