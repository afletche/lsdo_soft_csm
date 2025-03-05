import csdl_alpha as csdl
import lsdo_geo
import lsdo_function_spaces as lfs
import lsdo_soft_csm
import pickle
import meshio
import numpy as np
# import m3l

from modopt import CSDLAlphaProblem
from modopt import PySLSQP
import vedo




'''
Objective: Maximize actuator angle or average curvature
Design variables: 
1. Length
2. Change in Width
3. Height

ideas:
1. width profile
2. chamber thicknesses (with total chamber volume constraint)

Subject to: 
1. Length <= 0.7
2. Change in width <= 0.02
3. Height <= 0.1, Height >= 0.05
4. Chamber thickness <= 0.02, Chamber thickness >= 0.01
5. Total chamber volume <= 0.1
6. Total fish volume = initial fish volume?
'''

recorder = csdl.Recorder(inline=True)
recorder.start()

# region Import and Setup
def import_geometry() -> lfs.Function:
    with open("examples/advanced_examples/robotic_fish/pickle_files/fishy_volume_geometry_fine.pickle", 'rb') as handle:
        fishy = pickle.load(handle)
        fishy.coefficients = csdl.Variable(value=fishy.coefficients, name='fishy_coefficients')   # Remake variables because can't pickle variables
        return fishy

fishy = import_geometry()
fishy.name = 'fishy'

# fishy.plot(opacity=0.3)
# region -Structural Mesh Projection
mesh_file_path = "examples/advanced_examples/robotic_fish/meshes/"
# mesh_name = "module_v1_fine"
# mesh_name = "module_v1"
# mesh_name = "module_v1_refined"
mesh_name = "module_v1_refined_2"
structural_mesh = meshio.read(mesh_file_path + mesh_name + ".msh")
structural_mesh_nodes = structural_mesh.points/1000 + np.array([0.04, 0, 0.])   # Shift the mesh to the right to make it the middle module
structural_elements = structural_mesh.cells_dict['tetra']

# vedo_mesh = vedo.Mesh([structural_mesh_nodes, structural_elements]).wireframe()
# plotter = vedo.Plotter()
# plotter.show(vedo_mesh, axes=1, viewup='y')

# Reorder the nodes to match the FEniCS mesh
try:
    fenics_mesh_indices = pickle.load(open(mesh_file_path + mesh_name + "_fenics_mesh_indices.pickle", "rb"))
except FileNotFoundError:
    from dolfinx.io import XDMFFile
    from mpi4py import MPI
    with XDMFFile(MPI.COMM_WORLD, mesh_file_path + mesh_name + ".xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    fenics_mesh_indices = mesh.geometry.input_global_indices

    file_name = mesh_file_path + mesh_name + "_fenics_mesh_indices.pickle"
    with open(file_name, 'wb+') as handle:
        pickle.dump(fenics_mesh_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fenics_mesh_indices = pickle.load(open(mesh_file_path + mesh_name + "_fenics_mesh_indices.pickle", "rb"))
fenics_mesh_indices_mapping = dict(zip(np.arange(len(fenics_mesh_indices)), fenics_mesh_indices))
fenics_mesh_indices_inverse_mapping = dict((v,k) for k,v in fenics_mesh_indices_mapping.items())
reordered_structural_mesh_nodes = structural_mesh_nodes[fenics_mesh_indices]
reordered_structural_elements = np.zeros_like(structural_elements)
for i in range(structural_elements.shape[0]):
    for j in range(structural_elements.shape[1]):
        # reordered_structural_elements[i,j] = np.where(fenics_mesh_indices == structural_elements[i,j])[0][0]
        reordered_structural_elements[i,j] = fenics_mesh_indices_inverse_mapping[structural_elements[i,j]]

# vedo_mesh = vedo.Mesh([reordered_structural_mesh_nodes, reordered_structural_elements]).wireframe()
# plotter = vedo.Plotter()
# plotter.show(vedo_mesh, axes=1, viewup='y')
structural_mesh_nodes = reordered_structural_mesh_nodes
structural_elements = reordered_structural_elements

structural_mesh_parametric = fishy.project(points=structural_mesh_nodes, grid_search_density_parameter=10., plot=False)
# Plot projection result
# fishy_plot = fishy.plot(show=False, opacity=0.3)
# structural_mesh_node_values = fishy.evaluate(structural_mesh_parametric).value
# vedo_mesh = vedo.Mesh([structural_mesh_node_values, structural_elements]).wireframe()
# plotter = vedo.Plotter()
# plotter.show([fishy_plot, vedo_mesh], axes=1, viewup='y')


structural_module_front_parametric = fishy.project(points=np.array([[0.08 + 0.04, 0., 0.]]), plot=False)
structural_module_mid_parametric = fishy.project(points=np.array([[0.04 + 0.04, 0., 0.]]), plot=False)
structural_mesh_nodes = fishy.evaluate(structural_mesh_parametric).value.reshape((-1,3))
# endregion -Structural Mesh Projection

# region -Projections for Design Variables (Parameterization Solver Inputs)
fishy_nose_parametric = fishy.project(points=np.array([[0.3, 0., 0.]]), plot=False)
fishy_tail_tip_parametric = fishy.project(points=np.array([[-0.2, 0., 0.]]), plot=False)

fishy_left_parametric = fishy.project(points=np.array([[0., 0., -0.05]]), plot=False)
fishy_right_parametric = fishy.project(points=np.array([[0., 0., 0.05]]), plot=False)

fishy_top_parametric = fishy.project(points=np.array([[0., 0.1, 0.]]), plot=False)
fishy_bottom_parametric = fishy.project(points=np.array([[0., -0.09, 0.]]), plot=False)
# endregion -Projections for Design Variables (Parameterization Solver Inputs)
# endregion Import and Setup

# region Geometry Parameterization
# region -Create Parameterization Objects
num_ffd_sections = 2
ffd_block = lsdo_geo.construct_ffd_block_around_entities(entities=fishy, num_coefficients=(num_ffd_sections,2,2), degree=(1,1,1))
# ffd_block.plot()

ffd_sectional_parameterization = lsdo_geo.VolumeSectionalParameterization(
    parameterized_points=ffd_block.coefficients,
    principal_parametric_dimension=0,
    parameterized_points_shape=ffd_block.coefficients.shape,
    name='ffd_sectional_parameterization',
)
# plotting_elements = fishy.plot(show=False, opacity=0.3, color='#FFCD00')

linear_2_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
constant_1_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=0, coefficients_shape=(1,))


length_sectional_translations_b_spline_coefficients = csdl.ImplicitVariable(
    name='length_delta_translations_b_spline_coefficients',
    value=np.array([-0., 0.]),
)
length_sectional_translations_b_spline_parameterization = lfs.Function(
    name='length_delta_translations_b_spline',
    space=linear_2_dof_space,
    coefficients=length_sectional_translations_b_spline_coefficients,
)

sectional_delta_width = csdl.ImplicitVariable(value=-0., name='sectional_delta_width')
sectional_delta_height = csdl.ImplicitVariable(value=0., name='sectional_delta_height')

# endregion -Create Parameterization Objects

# region -Define Parameterization For Solver
parameterization_b_spline_input = np.linspace(0., 1., num_ffd_sections).reshape((-1,1))

length_sectional_translations = length_sectional_translations_b_spline_parameterization.evaluate(parameterization_b_spline_input)
sectional_widths = csdl.expand(sectional_delta_width, (num_ffd_sections,))
sectional_heights = csdl.expand(sectional_delta_height, (num_ffd_sections,))

ffd_sectional_parameterization_inputs = lsdo_geo.VolumeSectionalParameterizationInputs()
ffd_sectional_parameterization_inputs.add_sectional_translation(axis=0, translation=length_sectional_translations)
ffd_sectional_parameterization_inputs.add_sectional_stretch(axis=2, stretch=sectional_widths)
ffd_sectional_parameterization_inputs.add_sectional_stretch(axis=1, stretch=sectional_heights)
ffd_block_coefficients = ffd_sectional_parameterization.evaluate(ffd_sectional_parameterization_inputs, plot=False)

fishy_coefficients = ffd_block.evaluate(coefficients=ffd_block_coefficients, plot=False)

fishy.coefficients = fishy_coefficients
# fishy.plot()


# endregion -Evaluate Parameterization For Solver

# region -Evaluate Parameterization Solver
# region -Evaluate Parameterization Solver Inputs
fishy_nose = fishy.evaluate(fishy_nose_parametric)
fishy_tail_tip = fishy.evaluate(fishy_tail_tip_parametric)
computed_fishy_length = csdl.norm(fishy_nose - fishy_tail_tip)

fishy_left = fishy.evaluate(fishy_left_parametric)
fishy_right = fishy.evaluate(fishy_right_parametric)
computed_fishy_width = csdl.norm(fishy_left - fishy_right)

fishy_top = fishy.evaluate(fishy_top_parametric)
fishy_bottom = fishy.evaluate(fishy_bottom_parametric)
computed_fishy_height = csdl.norm(fishy_top - fishy_bottom)
# endregion -Evaluate Parameterization Solver Inputs

# region Geometric Design Variables
length = csdl.Variable(value=computed_fishy_length.value, name='length')
width = csdl.Variable(value=computed_fishy_width.value, name='width')
height = csdl.Variable(value=computed_fishy_height.value, name='height')

# length = csdl.Variable(value=1.1, name='length')
# width = csdl.Variable(value=0.02, name='width')
# height = csdl.Variable(value=0.07, name='height')

# endregion Geometric Design Variables

geometry_parameterization_solver = lsdo_geo.ParameterizationSolver()

geometry_parameterization_solver.add_parameter(length_sectional_translations_b_spline_coefficients)
geometry_parameterization_solver.add_parameter(sectional_delta_width)
geometry_parameterization_solver.add_parameter(sectional_delta_height)

geometric_parameterization_variables = lsdo_geo.GeometricVariables()
geometric_parameterization_variables.add_variable(computed_fishy_length, length)
geometric_parameterization_variables.add_variable(computed_fishy_width, width)
geometric_parameterization_variables.add_variable(computed_fishy_height, height)

geometry_parameterization_solver.evaluate(geometric_variables=geometric_parameterization_variables)

# fishy.plot()
# exit()
# endregion Geometry Parameterization

# region Evaluate Meshes
# - Evaluate Structural Mesh
structural_mesh = fishy.evaluate(structural_mesh_parametric).reshape((-1,3))

# - Shift structural mesh displacements so there is no displacement at the fixed BC
front_point_in_structural_solver = csdl.Variable(value=np.array([0.08 + 0.04, 0., 0.]))
front_point_in_structural_solver_expanded = csdl.expand(front_point_in_structural_solver, structural_mesh.shape, 'i->ji')
mid_point_in_structural_solver = csdl.Variable(value=np.array([0.04 + 0.04, 0., 0.]))
mid_point_in_structural_solver_expanded = csdl.expand(mid_point_in_structural_solver, structural_mesh.shape, 'i->ji')
current_front_point = fishy.evaluate(structural_module_front_parametric)
current_front_point_expanded = csdl.expand(current_front_point, structural_mesh.shape, 'i->ji')
current_mid_point = fishy.evaluate(structural_module_mid_parametric)
current_mid_point_expanded = csdl.expand(current_mid_point, structural_mesh.shape, 'i->ji')
# structural_mesh_displacements = structural_mesh - structural_mesh_nodes + front_point_in_structural_solver_expanded - current_front_point_expanded
structural_mesh_displacements = structural_mesh - structural_mesh_nodes + mid_point_in_structural_solver_expanded - current_mid_point_expanded
structural_mesh_displacements = structural_mesh_displacements.flatten()

# endregion Evaluate Meshes
# endregion Geoemetry Parameterization

# region Structural Solver
structural_displacements_flattened = lsdo_soft_csm.robotic_fish_static_structural_model(structural_mesh_displacements)
structural_displacements = structural_displacements_flattened.reshape((structural_displacements_flattened.size//3,3))

displaced_mesh = structural_mesh + structural_displacements

initial_displaced_mesh = displaced_mesh.value

# Plot structural solver result
fishy_plot = fishy.plot(show=False, opacity=0.3)
vedo_mesh = vedo.Mesh([displaced_mesh.value, structural_elements]).wireframe()
plotter = vedo.Plotter()
plotter.show([fishy_plot, vedo_mesh], axes=1, viewup='y')

# endregion Structural Solver

# region Fit B-Spline to Displacements and Construct Deformed Geometry
displacement_space = lfs.BSplineSpace(
    num_parametric_dimensions=3,
    degree=(2,2,2),
    coefficients_shape=(5,3,3))

mesh_parametric_coordinates_in_displacement_space = structural_mesh_parametric.copy()
module_2_min_u = np.min(mesh_parametric_coordinates_in_displacement_space[:,0]) # Use the mesh projections to guarantee mesh is in module bounds
module_2_max_u = np.max(mesh_parametric_coordinates_in_displacement_space[:,0])
for i, old_u in enumerate(structural_mesh_parametric[:,0]):
    new_u = (old_u - module_2_min_u)/(module_2_max_u - module_2_min_u)
    mesh_parametric_coordinates_in_displacement_space[i,0] = new_u

structural_displacements_b_spline = displacement_space.fit_function(structural_displacements,
                                                                    parametric_coordinates=mesh_parametric_coordinates_in_displacement_space)

deformed_module = displacement_space.fit_function(structural_mesh + structural_displacements, 
                                                  parametric_coordinates=mesh_parametric_coordinates_in_displacement_space)
# deformed_module.plot()
# exit()

# displacement_evaluation_map = displacement_space.compute_basis_matrix(mesh_parametric_coordinates_in_displacement_space, expand_map_for_physical=True)
# fitting_matrix = (displacement_evaluation_map.T).dot(displacement_evaluation_map) 
#                 # + regularization_parameter*sps.identity(displacement_evaluation_map.shape[1])
# fitting_rhs = m3l.matvec(displacement_evaluation_map.T.tocsc(), structural_displacements_flattened)
# structural_displacements_coefficients = m3l.solve(fitting_matrix, fitting_rhs)

# structural_displacements_b_spline = BSpline(
#     name='structural_displacements_b_spline',
#     space=displacement_space,
#     coefficients=structural_displacements_coefficients,
#     num_physical_dimensions=3,
# )
# structural_displacements_b_spline.plot()

coefficients_parametric = fishy.project(points=fishy.coefficients, grid_search_density_parameter=5., plot=False)
points_in_module_2_indices = list(np.where((coefficients_parametric[:,0] <= module_2_max_u) & (coefficients_parametric[:,0] >= module_2_min_u))[0])
# coefficient_indices = np.arange(fishy.coefficients.value.size).reshape((-1,3))

deformed_fishy = fishy.copy()
# for coefficients in module, evaluate displacements at corresponding location and add them to the current coefficients
coefficients_parametric_in_module = coefficients_parametric[points_in_module_2_indices,:]
coefficients_parametric_in_module[:,0] = (coefficients_parametric_in_module[:,0] - module_2_min_u)/(module_2_max_u - module_2_min_u)
deformations_in_module = structural_displacements_b_spline.evaluate(coefficients_parametric_in_module)
# NOTE: For line above, a temporary structural displacement B-spline will have to be created for each time step. (where it's coefficients change)
# indices = list(coefficient_indices[points_in_module_2_indices].reshape((-1)))
deformed_fishy.coefficients = deformed_fishy.coefficients.set(csdl.slice[points_in_module_2_indices], 
                                                              fishy.coefficients[points_in_module_2_indices] + deformations_in_module)

# deformed_fishy.plot()
# exit()

derivative_at_module_edge1 = deformed_module.evaluate(np.array([[0., 0.5, 0.5]]), parametric_derivative_orders=(1,0,0))
derivative_at_module_edge = derivative_at_module_edge1/csdl.norm(derivative_at_module_edge1)    # normalize
module_edge = deformed_module.evaluate(np.array([[0., 0.5, 0.5]]))
# endregion Fit B-Spline to Displacements and Construct Deformed Geometry

# # region Compute Surface Area
# num_elements_per_dimension = 50
# parametric_mesh_2, parametric_mesh_1 = \
#     np.meshgrid(np.linspace(0., 1., num_elements_per_dimension), np.linspace(0., 1., num_elements_per_dimension))
# # parametric_grid_1 = np.zeros((num_elements_per_dimension**2, 3))
# # parametric_grid_2 = np.zeros((num_elements_per_dimension**2, 3))
# # parametric_grid_3 = np.zeros((num_elements_per_dimension**2, 3))
# # parametric_grid_4 = np.zeros((num_elements_per_dimension**2, 3))
# parametric_grid_5 = np.zeros((num_elements_per_dimension, num_elements_per_dimension, 3))
# parametric_grid_6 = np.zeros((num_elements_per_dimension, num_elements_per_dimension, 3))

# # parametric_grid_1[:,1] = parametric_mesh_1.flatten()
# # parametric_grid_1[:,2] = parametric_mesh_2.flatten()
# # parametric_grid_2[:,0] = np.ones_like(parametric_mesh_1.flatten())
# # parametric_grid_2[:,1] = parametric_mesh_1.flatten()
# # parametric_grid_2[:,2] = parametric_mesh_2.flatten()
# # parametric_grid_3[:,0] = parametric_mesh_1.flatten()
# # # parametric_grid_3[:,1] = np.zeros(parametric_mesh_1.flatten().shape)
# # parametric_grid_3[:,2] = parametric_mesh_2.flatten()
# # parametric_grid_4[:,0] = parametric_mesh_1.flatten()
# # parametric_grid_4[:,1] = np.ones_like(parametric_mesh_1.flatten())
# # parametric_grid_4[:,2] = parametric_mesh_2.flatten()
# parametric_grid_5[:,:,0] = parametric_mesh_1
# parametric_grid_5[:,:,1] = parametric_mesh_2
# parametric_grid_6[:,:,0] = parametric_mesh_1
# parametric_grid_6[:,:,1] = parametric_mesh_2
# parametric_grid_6[:,:,2] = np.ones_like(parametric_mesh_1)

# # parametric_grids = [parametric_grid_1, parametric_grid_2, parametric_grid_3, parametric_grid_4, parametric_grid_5, parametric_grid_6]
# parametric_grids = [parametric_grid_5, parametric_grid_6]

# surface_area = m3l.Variable(value=0, shape=(1, ))
# # for i in range(6):
# for i in range(2):
#     surface_grid = fishy.evaluate(parametric_grids[i])

#     u_vectors = surface_grid[1:,:] - surface_grid[:-1,:]
#     v_vectors = surface_grid[:,1:] - surface_grid[:,:-1]

#     u_vectors_low_v = u_vectors[:,:-1]
#     u_vectors_high_v = u_vectors[:,1:]
#     v_vectors_low_u = v_vectors[:-1,:]
#     v_vectors_high_u = v_vectors[1:,:]

#     area_vectors_left_lower = csdl.cross(u_vectors_low_v, v_vectors_high_u, axis=2)
#     area_vectors_right_upper = csdl.cross(v_vectors_low_u, u_vectors_high_v, axis=2)
#     area_magnitudes_left_lower = csdl.norm(area_vectors_left_lower, ord=2, axes=(2,))
#     area_magnitudes_right_upper = csdl.norm(area_vectors_right_upper, ord=2, axes=(2,))
#     area_magnitudes = (area_magnitudes_left_lower + area_magnitudes_right_upper)/2
#     wireframe_area = csdl.sum(area_magnitudes, axes=(0, 1))
#     surface_area = surface_area + wireframe_area
# # endregion Compute Surface Area


# region Objective Model
max_tip_displacement_node = np.argmax(structural_displacements.value.reshape((-1,3))[:,2])
flattened_index = max_tip_displacement_node*3 + 2
# surface_area_penalty = 4.       # shrinks the length and height a bit smaller (width lower bound)
# surface_area_penalty = 2.     # just shrinks the width
surface_area_penalty = 1.     # Makes the length and height a bit larger (width lower bound)
# surface_area_penalty = 1.e-10

initial_angle = csdl.Variable(shape=(3,), value=np.array([1., 0., 0.]))
angle = csdl.arccos(csdl.vdot(initial_angle, derivative_at_module_edge))
# system_model.register_output(angle)

objective = -angle


# parametric_coordinate_at_module_edge = np.array([[module_2_min_u, 0.5, 0.5]])
# derivative_at_module_edge_old = deformed_fishy.evaluate(parametric_coordinate_at_module_edge, parametric_derivative_orders=(1,0,0))
# derivative_at_module_edge_old = derivative_at_module_edge_old/csdl.norm(derivative_at_module_edge_old)    # normalize
# old_angle = csdl.arccos(csdl.vdot(initial_angle, derivative_at_module_edge_old))
# endregion Objective Model

# region Optimization
length.set_as_design_variable(lower=0.3, upper=1., scaler=1.e1)
width.set_as_design_variable(lower=0.02, upper=0.08, scaler=1.e2)
height.set_as_design_variable(lower=0.03, upper=0.14, scaler=1.e2)

objective.set_as_objective()


sim = csdl.experimental.PySimulator(recorder=recorder)

# exit()
optimization_problem = CSDLAlphaProblem(problem_name='fishy_optimization', simulator=sim)
# optimizer = SLSQP(optimization_problem, maxiter=100, ftol=1.e-7)
optimizer = PySLSQP(optimization_problem, solver_options={'maxiter':100, 'acc':1.e-9}, readable_outputs=['x'])

initial_objective_value = objective.value
initial_length = length.value
initial_width = width.value
initial_height = height.value
# initial_surface_area = surface_area.value
initial_angle = angle.value

# from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
# verify_derivatives_inline([csdl.norm(fishy.coefficients)], [length, width, height], 1.e-6, raise_on_error=False)
# verify_derivatives_inline([structural_mesh_displacements[list(np.arange(0, structural_displacements_flattened.size, 100))]], [length, width, height], 1.e-6, raise_on_error=False)
# verify_derivatives_inline([structural_displacements_flattened[list(np.arange(0, structural_displacements_flattened.size, 100))]], [length, width, height], 1.e-6, raise_on_error=False)
# verify_derivatives_inline([structural_displacements_flattened[10000]], [length, width, height], 1.e-6, raise_on_error=False)
# verify_derivatives_inline([deformed_module.coefficients], [length, width, height], 1.e-6, raise_on_error=False)
# verify_derivatives_inline([derivative_at_module_edge1], [length, width, height], 1.e-6, raise_on_error=False)
# verify_derivatives_inline([derivative_at_module_edge], [length, width, height], 1.e-6, raise_on_error=False)
# verify_derivatives_inline([objective], [length, width, height], 1.e-6, raise_on_error=False)
# optimizer.check_first_derivatives(optimization_problem.x0)

# exit()

# d_objective_d_length = csdl.derivative(objective, length)
# d_objective_d_width = csdl.derivative(objective, width)
# d_objective_d_height = csdl.derivative(objective, height)

# print('d_objective_d_length', d_objective_d_length.value)
# print('d_objective_d_width', d_objective_d_width.value)
# print('d_objective_d_height', d_objective_d_height.value)
# exit()

# recorder.execute()


# video = vedo.Video('fishy_width_sweep_fixed_midpoint.mp4', duration=5, backend='cv')
# width_values = np.linspace(0.015, 0.05, 5)
# direction_values = np.zeros((len(width_values), 3))
# old_direction_values = np.zeros((len(width_values), 3))
# objective_values = np.zeros_like(width_values)
# old_objective_values = np.zeros_like(width_values)
# for i, width_value in enumerate(width_values):
#     print(i)
#     width.value = width_value
#     recorder.execute()
#     objective_values[i] = objective.value
#     old_objective_values[i] = -old_angle.value
#     direction_values[i] = derivative_at_module_edge.value
#     old_direction_values[i] = derivative_at_module_edge_old.value

#     fishy_plot = fishy.plot(show=False, opacity=0.3)
#     vedo_mesh = vedo.Mesh([displaced_mesh.value, structural_elements]).wireframe()
#     arrow = vedo.Arrow(tuple(module_edge.value.reshape((-1,))), 
#                                    tuple((module_edge.value - derivative_at_module_edge.value/10).reshape((-1,))), s=0.0005)
#     plotter = vedo.Plotter(offscreen=True)
#     plotter.show([fishy_plot, vedo_mesh, arrow], axes=1, viewup='y')
#     video.add_frame()

# video.close()

# print(width_values)
# print(objective_values)
# print(direction_values)
# print(old_direction_values)
# import matplotlib.pyplot as plt
# plt.plot(width_values, -objective_values, label='Angle')
# plt.title('Angle vs Width')
# plt.xlabel('Width')
# plt.ylabel('Angle')
# # plt.plot(width_values, old_objective_values, label='Old Objective')
# plt.legend()
# plt.show()
# exit()

optimizer.solve()
optimizer.print_results()


print('Initial Objective: ', initial_objective_value)
print('Initial Length', initial_length)
print('Initial Width', initial_width)
print('Initial Height', initial_height)
# print('Initial Surface Area', initial_surface_area)
print('Initial Angle: ', initial_angle)

print('Optimized Objective: ', objective.value)
print('Optimized Length', length.value)
print('Optimized Width', width.value)
print('Optimized Height', height.value)
# print('Optimized Surface Area', surface_area.value)
print('Optimized Angle: ', angle.value)

print('Percent Change in Objective', (objective.value - initial_objective_value)/initial_objective_value*100)
print("Percent Change in Length: ", (length.value - initial_length)/initial_length*100)
print("Percent Change in Width: ", (width.value - initial_width)/initial_width*100)
print("Percent Change in Height: ", (height.value - initial_height)/initial_height*100)

# from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
# verify_derivatives_inline([csdl.norm(fishy.coefficients)], [length, width, height], 1.e-6, raise_on_error=False)
# verify_derivatives_inline([csdl.norm(structural_displacements_flattened)], [length, width, height], 1.e-6, raise_on_error=False)
# verify_derivatives_inline([csdl.norm(structural_displacements_b_spline.coefficients)], [length, width, height], 1.e-6, raise_on_error=False)
# verify_derivatives_inline([csdl.norm(deformations_in_module)], [length, width, height], 1.e-6, raise_on_error=False)
# verify_derivatives_inline([csdl.norm(deformed_fishy.coefficients)], [length, width, height], 1.e-6, raise_on_error=False)
# optimizer.check_first_derivatives(optimization_problem.x0)

# Plot structural solver result
fishy_plot = fishy.plot(show=False, opacity=0.3)
vedo_mesh_initial = vedo.Mesh([initial_displaced_mesh, structural_elements]).wireframe().color('yellow').opacity(0.4)
vedo_mesh = vedo.Mesh([displaced_mesh.value, structural_elements]).wireframe().color('green').opacity(0.8)
plotter = vedo.Plotter()
plotter.show([fishy_plot, vedo_mesh_initial, vedo_mesh], axes=1, viewup='y')

# endregion Optimization