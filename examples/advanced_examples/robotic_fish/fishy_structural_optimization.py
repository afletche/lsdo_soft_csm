import lsdo_geo
from lsdo_geo.splines.b_splines.b_spline import BSpline
import pickle
import meshio
import numpy as np
import scipy.sparse as sps
import m3l
import vedo
import python_csdl_backend
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem

from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver
from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
from lsdo_geo.core.parameterization.volume_sectional_parameterization import VolumeSectionalParameterization
from lsdo_geo.splines.b_splines.b_spline_space import BSplineSpace

from lsdo_soft_csm.core.robotic_fish_static_structural_model import RoboticFishStaticStructuralModel
from lsdo_soft_csm.core.robotic_fish_dynamic_structural_model import RoboticFishDynamicStructuralModel

'''
Objective: Maximize tail tip displacement
Design variable: Length
Subject to: Length <= 0.5
'''

# region Import and Setup
def import_geometry() -> BSpline:
    with open("examples/advanced_examples/robotic_fish/pickle_files/fishy_volume_geometry_fine.pickle", 'rb') as handle:
        fishy = pickle.load(handle)
        return fishy

fishy = import_geometry()
fishy.name = 'fishy'

# fishy.plot(opacity=0.3)
# region -Structural Mesh Projection
structural_mesh = meshio.read("examples/advanced_examples/robotic_fish/meshes/module_v1.msh")
structural_mesh_nodes = structural_mesh.points/1000 + np.array([0.08 + 0.04, 0, 0.])   # Shift the mesh to the right to make it the front module

fenics_mesh_indices = pickle.load(open("examples/advanced_examples/robotic_fish/meshes/module_v1_fenics_mesh_indices.pickle", "rb"))
structural_mesh_nodes = structural_mesh_nodes[fenics_mesh_indices]

structural_mesh_parametric = fishy.project(points=structural_mesh_nodes, grid_search_density=100, max_iterations=500, plot=False)
structural_module_front_parametric = fishy.project(points=np.array([[0.08 + 0.12, 0., 0.]]), grid_search_density=150, plot=False)
structural_module_back_parametric = fishy.project(points=np.array([[0.12, 0., 0.]]), grid_search_density=150, plot=False)
# print(structural_module_back_parametric)
# exit()
structural_mesh_nodes = fishy.evaluate(structural_mesh_parametric).value.reshape((-1,3))
# endregion -Structural Mesh Projection

# region -Projections for Design Variables (Parameterization Solver Inputs)
fishy_nose_parametric = fishy.project(points=np.array([[0.3, 0., 0.]]), grid_search_density=100, plot=False)
fishy_tail_tip_parametric = fishy.project(points=np.array([[-0.2, 0., 0.]]), grid_search_density=100, plot=False)
# endregion -Projections for Design Variables (Parameterization Solver Inputs)

# endregion Import and Setup

system_model = m3l.Model()

# region Geometry Parameterization
# region -Create Parameterization Objects
geometry_parameterization_solver = ParameterizationSolver()

num_ffd_sections = 2
ffd_block = construct_ffd_block_around_entities(name='ffd_block', entities=fishy, num_coefficients=(num_ffd_sections,2,2), order=(2,2,2))
# ffd_block.plot()

ffd_sectional_parameterization = VolumeSectionalParameterization(
    name='ffd_sectional_parameterization',
    parameterized_points=ffd_block.coefficients,
    principal_parametric_dimension=0,
    parameterized_points_shape=ffd_block.coefficients_shape,
)
ffd_sectional_parameterization.add_sectional_translation(name='length_sectional_translations', axis=0)
ffd_sectional_parameterization.add_sectional_stretch(name='width_sectional_stretches', axis=2)

linear_2_dof_space = BSplineSpace(name='linear_2_dof_space', order=2, parametric_coefficients_shape=(2,))
constant_1_dof_space = BSplineSpace(name='linear_1_dof_space', order=1, parametric_coefficients_shape=(1,))


length_sectional_translations_b_spline_coefficients = m3l.Variable(
    name='length_sectional_translations_b_spline_coefficients',
    shape=(2,),
    value=np.array([0., 0.]),
)
length_sectional_translations_b_spline_parameterization = BSpline(
    name='length_sectional_translations_b_spline_parameterization',
    space=linear_2_dof_space,
    coefficients=length_sectional_translations_b_spline_coefficients,
    num_physical_dimensions=1,
)

width_sectional_stretches_b_spline_coefficients = m3l.Variable(
    name='width_sectional_stretches_b_spline_coefficients',
    shape=(1,),
    value=np.array([0.]),
)
width_sectional_stretches_b_spline_parameterization = BSpline(
    name='width_sectional_stretches_b_spline_parameterization',
    space=constant_1_dof_space,
    coefficients=width_sectional_stretches_b_spline_coefficients,
    num_physical_dimensions=1,
)

geometry_parameterization_solver.declare_state(name='length_sectional_translations_b_spline_coefficients', 
                                               state=length_sectional_translations_b_spline_coefficients)
# endregion -Create Parameterization Objects

# region -Define Parameterization For Solver
parameterization_b_spline_input = np.linspace(0., 1., num_ffd_sections).reshape((-1,1))

length_sectional_translations = length_sectional_translations_b_spline_parameterization.evaluate(parameterization_b_spline_input)
width_sectional_stretches = width_sectional_stretches_b_spline_parameterization.evaluate(parameterization_b_spline_input)

sectional_parameters = {
    'length_sectional_translations' : length_sectional_translations,
    'width_sectional_stretches' : width_sectional_stretches
    }
ffd_block_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

fishy_coefficients = ffd_block.evaluate(coefficients=ffd_block_coefficients, plot=False)

fishy.coefficients = fishy_coefficients
# fishy.plot()

# region -Evaluate Parameterization Solver Inputs
fishy_nose = fishy.evaluate(fishy_nose_parametric)
fishy_tail_tip = fishy.evaluate(fishy_tail_tip_parametric)
fishy_length = m3l.norm(fishy_nose - fishy_tail_tip)

geometry_parameterization_solver.declare_input(name='fishy_length', input=fishy_length)

# endregion -Evaluate Parameterization Solver Inputs

# endregion -Evaluate Parameterization For Solver

# region -Evaluate Parameterization
fishy_length_input = system_model.create_input(name='fishy_length', shape=(1,), val=fishy_length.value, dv_flag=True, upper=0.7)
# fishy_length_input = system_model.create_input(name='fishy_length', shape=(1,), val=0.7, dv_flag=True, upper=1.)
optimization_inputs = {'fishy_length': fishy_length_input}
parameterization_solver_states = geometry_parameterization_solver.evaluate(optimization_inputs)

length_sectional_translations_b_spline_parameterization.coefficients = parameterization_solver_states['length_sectional_translations_b_spline_coefficients']
width_scaling_input = system_model.create_input(name='width_scaling', shape=(1,), val=-0.0, dv_flag=False, upper=0.05, lower=-0.02)
width_sectional_stretches_b_spline_parameterization.coefficients = width_scaling_input

length_sectional_translations = length_sectional_translations_b_spline_parameterization.evaluate(parameterization_b_spline_input)
width_sectional_stretches = width_sectional_stretches_b_spline_parameterization.evaluate(parameterization_b_spline_input)

sectional_parameters = {
    'length_sectional_translations':length_sectional_translations,
    'width_sectional_stretches':width_sectional_stretches,
    }
ffd_block_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

fishy_coefficients = ffd_block.evaluate(coefficients=ffd_block_coefficients, plot=False)
fishy.coefficients = fishy_coefficients
# fishy.plot()
# endregion Geometry Parameterization

# region Evaluate Meshes
# fishy_length = m3l.norm(fishy.evaluate(fishy_nose_parametric) - fishy.evaluate(fishy_tail_tip_parametric))

structural_mesh = fishy.evaluate(structural_mesh_parametric).reshape((-1,3))

# geometry_plot = fishy.plot(show=False, opacity=0.3)
# mesh_points = vedo.Points(structural_mesh.value, r=5, c='gold').opacity(0.6)
# plotter = vedo.Plotter()
# plotter.show([geometry_plot, mesh_points], axes=1, viewup='y')

front_point_in_structural_solver = m3l.Variable(shape=(3,), value=np.array([0.08 + 0.12, 0., 0.]))
current_front_point = fishy.evaluate(structural_module_front_parametric)
structural_mesh_displacements = structural_mesh - structural_mesh_nodes + front_point_in_structural_solver - current_front_point
# structural_mesh_displacements.value = np.zeros_like(structural_mesh_displacements.value)
# structural_mesh_displacements.value = np.ones_like(structural_mesh_displacements.value)*0.01
# mesh_points = vedo.Points(structural_mesh_displacements.value, r=5, c='gold').opacity(0.6)
# plotter = vedo.Plotter()
# plotter.show([mesh_points], axes=1, viewup='y')
# endregion Evaluate Meshes
# endregion Geoemetry Parameterization

# region Structural Solver
structural_model = RoboticFishStaticStructuralModel()
# structural_model = RoboticFishDynamicStructuralModel()
structural_displacements_flattened = structural_model.evaluate(structural_mesh_displacements.reshape((-1,)))
structural_displacements = structural_displacements_flattened.reshape((-1,3))

# Reconstruct displacement field
displacement_space = BSplineSpace(
    name='structural_displacements_b_spline_space',
    order=(3,1,1),
    parametric_coefficients_shape=(10,1,1))
mesh_parametric_coordinates_in_displacement_space = structural_mesh_parametric.copy()
min_u = np.min(mesh_parametric_coordinates_in_displacement_space[:,0])
max_u = np.max(mesh_parametric_coordinates_in_displacement_space[:,0])
for i, old_u in enumerate(structural_mesh_parametric[:,0]):
    new_u = (old_u - min_u)/(max_u - min_u)
    mesh_parametric_coordinates_in_displacement_space[i,0] = new_u
dummy_b_spline = BSpline(
    name='dummy_b_spline',
    space=displacement_space,
    coefficients=np.zeros(displacement_space.parametric_coefficients_shape + (3,)),
    num_physical_dimensions=3,
)
displacement_evaluation_map = dummy_b_spline.compute_evaluation_map(mesh_parametric_coordinates_in_displacement_space, expand_map_for_physical=True)
fitting_matrix = (displacement_evaluation_map.T).dot(displacement_evaluation_map) 
                # + regularization_parameter*sps.identity(displacement_evaluation_map.shape[1])
fitting_rhs = m3l.matvec(displacement_evaluation_map.T.tocsc(), structural_displacements_flattened)
structural_displacements_coefficients = m3l.solve(fitting_matrix, fitting_rhs)

structural_displacements_b_spline = BSpline(
    name='structural_displacements_b_spline',
    space=displacement_space,
    coefficients=structural_displacements_coefficients,
    num_physical_dimensions=3,
)

displacement_space.create_function(name='structural_displacements_b_spline', coefficients=structural_displacements_coefficients)

# structural_mesh_evaluation_map = fishy.compute_evaluation_map(structural_mesh_parametric, expand_map_for_physical=True)
# regularization_parameter = 1e-4

# region -Map Displacements To Geometry
# Perform initial fitting to get the derivative at the edge of the structural module
# fitting_matrix = (structural_mesh_evaluation_map.T).dot(structural_mesh_evaluation_map) + \
#     regularization_parameter*sps.identity(structural_mesh_evaluation_map.shape[1])
# fitting_rhs = m3l.matvec(structural_mesh_evaluation_map.T.tocsc(), structural_displacements_flattened)
# structural_displacements_coefficients = m3l.solve(fitting_matrix, fitting_rhs)

# structural_displacements_b_spline = \
#     fishy.space.create_function(name='structural_displacements_b_spline', coefficients=structural_displacements_coefficients)

# Use derivative to stuff data with rigid body displacement
num_u_stuff = 50
num_v_stuff = 25
num_w_stuff = 25

# edge_u_coordinate = 0.66
edge_u_coordinate = structural_module_back_parametric[0][0]
w_coordinates, v_coordinates = np.meshgrid(np.linspace(0., 1., num_v_stuff), np.linspace(0., 1., num_w_stuff))
u_coordinates = np.ones_like(v_coordinates)*0.
derivative_parametric_coordinates = np.stack((u_coordinates.flatten(), v_coordinates.flatten(), w_coordinates.flatten()), axis=1)
derivative_values_at_edge = structural_displacements_b_spline.evaluate(derivative_parametric_coordinates, parametric_derivative_order=(1,0,0))
# derivative_values_at_edge = m3l.expand(derivative_values_at_edge.reshape((-1,3)), new_shape=(num_u_stuff, num_v_stuff*num_w_stuff, 3), 
#                                        indices='ij->kij')

displacement_values_at_edge = structural_displacements_b_spline.evaluate(derivative_parametric_coordinates)

# ##########    For trying to decide where to call the edge
# fishy.coefficients = fishy.coefficients + structural_displacements_b_spline.coefficients
# fishy.evaluate(derivative_parametric_coordinates, plot=True)
# ##########

# displacement_values_at_edge = m3l.expand(displacement_values_at_edge.reshape((-1,3)), new_shape=(num_u_stuff, num_v_stuff*num_w_stuff, 3),
#                                          indices='ij->kij')

u_stuff = np.linspace(0., edge_u_coordinate, num_u_stuff, endpoint=False)
stuffing_points = m3l.Variable(shape=(num_u_stuff*num_v_stuff*num_w_stuff*3,), value=np.zeros(num_u_stuff*num_v_stuff*num_w_stuff*3))
stuffing_parametric_coordinates = np.zeros((num_u_stuff*num_v_stuff*num_w_stuff, 3))

# # This should be faster but currently doesn't work
# u_stuff_tiled = np.tile(u_stuff.reshape((num_u_stuff,1,1)), (1,num_v_stuff*num_w_stuff,3))
# delta_u = u_stuff_tiled - edge_u_coordinate
# stuffing_points = (displacement_values_at_edge + derivative_values_at_edge*delta_u).reshape((-1,))
# stuffing_parametric_coordinates = np.meshgrid(np.linspace(0., edge_u_coordinate, num_u_stuff, endpoint=False),
#                                               np.linspace(0., 1., num_v_stuff), np.linspace(0., 1., num_w_stuff))
# stuffing_parametric_coordinates = np.stack((stuffing_parametric_coordinates[0].flatten(), 
#                                             stuffing_parametric_coordinates[1].flatten(), 
#                                             stuffing_parametric_coordinates[2].flatten()), axis=1)

# This almost works, but this is slow because of lots of small-ish indexing
for i, u in enumerate(u_stuff):
    delta_u = (u - edge_u_coordinate)/(max_u - min_u)
    flattened_indices = np.arange(i*num_v_stuff*num_w_stuff*3, (i+1)*num_v_stuff*num_w_stuff*3)
    stuffing_points[flattened_indices] = displacement_values_at_edge + (derivative_values_at_edge*delta_u).reshape((-1,))
    stuffing_parametric_coordinates_at_this_u = derivative_parametric_coordinates.copy()
    stuffing_parametric_coordinates_at_this_u[:,0] = u
    indices = np.arange(i*num_v_stuff*num_w_stuff, (i+1)*num_v_stuff*num_w_stuff)
    stuffing_parametric_coordinates[indices] = stuffing_parametric_coordinates_at_this_u

# Fit displacements B-spline to structural displcaements + stuffing points
combined_parametric_coordinates = np.concatenate((structural_mesh_parametric, stuffing_parametric_coordinates))
combined_values = m3l.vstack((structural_displacements_flattened, stuffing_points))

# regularization_parameter = 1.e-3
regularization_parameter = 1.e-3
combined_evaluation_matrix = fishy.compute_evaluation_map(combined_parametric_coordinates, expand_map_for_physical=True)
fitting_matrix = (combined_evaluation_matrix.T).dot(combined_evaluation_matrix) \
    + regularization_parameter*sps.identity(combined_evaluation_matrix.shape[1])
fitting_rhs = m3l.matvec(combined_evaluation_matrix.T.tocsc(), combined_values)
combined_structural_displacements_coefficients = m3l.solve(fitting_matrix, fitting_rhs)

structural_displacements_b_spline = \
    fishy.space.create_function(name='structural_displacements_b_spline', coefficients=combined_structural_displacements_coefficients)


structural_displacements_magnitude_b_spline = BSpline(
    name='structural_displacements_magnitude_b_spline',
    space=fishy.space,
    coefficients=m3l.norm(structural_displacements_b_spline.coefficients.reshape((-1,3)), axes=(1,)),
    num_physical_dimensions=1,
)
# fishy.plot(opacity=0.8, color=structural_displacements_magnitude_b_spline)

deformed_fishy = fishy.copy()
deformed_fishy.coefficients = deformed_fishy.coefficients + structural_displacements_b_spline.coefficients
deformed_fishy.plot(color=structural_displacements_magnitude_b_spline)
# deformed_fishy.plot()

opposite_deformed_fishy = deformed_fishy.copy()
coefficients_shape = opposite_deformed_fishy.space.parametric_coefficients_shape + (3,)
indices = np.arange(np.prod(coefficients_shape)).reshape(coefficients_shape)
# want to flip the z component of the coefficients and invert the ordering of the x component along the 3rd axis of coefficients
z_indicies = indices[:,:,:,2].reshape((-1,))
opposite_deformed_fishy.coefficients[z_indicies] = -deformed_fishy.coefficients[z_indicies]
for i in range(coefficients_shape[2]):
    # set_indices = [:,:,i,0]
    # get_indices = [:,:,coefficients_shape[2]-i-1,0]
    set_indices = indices[:,:,i,0].reshape((-1,))
    get_indices = indices[:,:,coefficients_shape[2]-i-1,0].reshape((-1,))
    # set_indices = np.arange(i*coefficients_shape[0]*coefficients_shape[1], (i+1)*coefficients_shape[0]*coefficients_shape[1])
    # get_indices = np.arange((coefficients_shape[2]-i-1)*coefficients_shape[0]*coefficients_shape[1], (coefficients_shape[2]-i)*coefficients_shape[0]*coefficients_shape[1])
    opposite_deformed_fishy.coefficients[set_indices] = deformed_fishy.coefficients[get_indices]
opposite_deformed_fishy.plot(color=structural_displacements_magnitude_b_spline)
exit()

# endregion -Map Displacements To Geometry

# region -Perform interpolation to get dynamic profile
# NOTE: interpolation is a linear combination between displaced geometry and the displacement flipped around z. Interpolation is exponential in time.
displaced_mesh = structural_mesh + structural_displacements
structural_displacements_z_flipped = structural_displacements_flattened.copy()
indices = np.arange(2, structural_displacements_flattened.shape[0], 3)
structural_displacements_z_flipped[indices] = -structural_displacements_z_flipped[indices]
structural_displacements_z_flipped = structural_displacements_z_flipped.reshape((-1,3))
displaced_mesh_other_direction = structural_mesh + structural_displacements_z_flipped

time_constant = 3
actuation_frequency = 1.
actuation_period = 1/actuation_frequency
num_vast_time_steps = 41
time_segment_1 = np.linspace(0, actuation_period/2, num_vast_time_steps//2, endpoint=False)
time_segment_2 = np.linspace(actuation_period/2, actuation_period, num_vast_time_steps//2 + 1, endpoint=True)
time = np.concatenate((time_segment_1, time_segment_2))

start_weights_segment_1 = np.exp(-time_constant*time_segment_1)
stop_weights_segment_1 = 1 - start_weights_segment_1
start_weights_segment_2 = np.exp(-time_constant*(time_segment_2 - actuation_period/2))
stop_weights_segment_2 = 1 - start_weights_segment_2

structural_mesh_segment_1 = m3l.linear_combination(displaced_mesh, displaced_mesh_other_direction, 
                                                   start_weights=start_weights_segment_1, stop_weights=stop_weights_segment_1)
structural_mesh_segment_2 = m3l.linear_combination(displaced_mesh_other_direction, displaced_mesh,
                                                    start_weights=start_weights_segment_2, stop_weights=stop_weights_segment_2)
# endregion -Perform interpolation to get dynamic profile

# import csdl
# csdl.solve()

fishy_plot = fishy.plot(show=False, opacity=0.3)
video = vedo.Video('examples/advanced_examples/robotic_fish/temp/fishy_optimization.mp4', fps=41, backend='cv')
for i in range(num_vast_time_steps):
    if i < num_vast_time_steps//2:
        structural_mesh = structural_mesh_segment_1.value[i]
    else:
        structural_mesh = structural_mesh_segment_2.value[i - num_vast_time_steps//2]

    mesh_points = vedo.Points(structural_mesh, r=5, c='gold').opacity(0.6)
    plotter = vedo.Plotter(offscreen=True)
    plotter.show([fishy_plot, mesh_points], axes=1, viewup='y')
    video.add_frame()
video.close()
exit()
# fishy_plot = fishy.plot(show=False, opacity=0.3)
# mesh_points = vedo.Points(displaced_mesh.value, r=5, c='gold').opacity(0.6)
# plotter = vedo.Plotter()
# plotter.show([fishy_plot, mesh_points], axes=1, viewup='y')

# endregion Structural Solver

# region Objective Model
max_tip_displacement_node = np.argmax(structural_displacements.value.reshape((-1,3))[:,2])
flattened_index = max_tip_displacement_node*3 + 2
objective = -(structural_displacements_flattened[int(flattened_index)]**2)

system_model.register_output(objective)
system_model.add_objective(objective, scaler=1e5)
# endregion Objective Model

csdl_model = system_model.assemble()
sim = python_csdl_backend.Simulator(csdl_model)
# sim.run()

# region Optimization
prob = CSDLProblem(problem_name='fishy_optimization', simulator=sim)
optimizer = SLSQP(prob, maxiter=1000, ftol=1E-6)
optimizer.solve()
optimizer.print_results()

print('Initial Objective: ', objective.value*1e5)
print('Initial Length', fishy_length_input.value)

print('Optimized Objective: ', sim[objective.operation.name + '.' + objective.name]*1e5)
print('Optimized Length', sim[fishy_length_input.name])
# print('Optimized Width Scaling', sim[width_scaling_input.name])
# endregion Optimization
