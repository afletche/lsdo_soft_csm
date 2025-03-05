import lsdo_geo
from lsdo_geo.splines.b_splines.b_spline import BSpline
import pickle
import meshio
import numpy as np

import vedo

# from modopt.scipy_library import SLSQP
# from modopt.csdl_library import CSDLProblem

from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver
from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
from lsdo_geo.core.parameterization.volume_sectional_parameterization import VolumeSectionalParameterization
from lsdo_geo.splines.b_splines.b_spline_space import BSplineSpace

from lsdo_soft_csm.core.robotic_fish_static_structural_model import RoboticFishStaticStructuralModel
import matplotlib.pyplot as plt

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

# region Import and Setup
def import_geometry() -> BSpline:
    with open("examples/advanced_examples/robotic_fish/pickle_files/fishy_volume_geometry_fine.pickle", 'rb') as handle:
        fishy = pickle.load(handle)
        return fishy

fishy = import_geometry()
fishy.name = 'fishy'

# fishy.plot(opacity=0.3)
# region -Structural Mesh Projection
mesh_file_path = "examples/advanced_examples/robotic_fish/meshes/"
# mesh_name = "module_v1_fine"
mesh_name = "module_v1"
structural_mesh = meshio.read(mesh_file_path + mesh_name + ".msh")
structural_mesh_nodes = structural_mesh.points/1000 + np.array([0.04, 0, 0.])   # Shift the mesh to the right to make it the front module

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
structural_mesh_nodes = structural_mesh_nodes[fenics_mesh_indices]

if mesh_name == "module_v1":
    grid_search_density = 100
elif mesh_name == "module_v1_fine":
    grid_search_density = 100
else:
    raise Exception("Specify grid search for this mesh.")
structural_mesh_parametric = fishy.project(points=structural_mesh_nodes, grid_search_density=grid_search_density, max_iterations=500, plot=False)
structural_module_front_parametric = fishy.project(points=np.array([[0.08 + 0.04, 0., 0.]]), grid_search_density=150, plot=False)
structural_mesh_nodes = fishy.evaluate(structural_mesh_parametric).value.reshape((-1,3))
# endregion -Structural Mesh Projection

# region -Projections for Design Variables (Parameterization Solver Inputs)
fishy_nose_parametric = fishy.project(points=np.array([[0.3, 0., 0.]]), grid_search_density=100, plot=False)
fishy_tail_tip_parametric = fishy.project(points=np.array([[-0.2, 0., 0.]]), grid_search_density=100, plot=False)

fishy_top_parametric = fishy.project(points=np.array([[0., 0.1, 0.]]), grid_search_density=150, plot=False)
fishy_bottom_parametric = fishy.project(points=np.array([[0., -0.09, 0.]]), grid_search_density=150, plot=False)
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
ffd_sectional_parameterization.add_sectional_stretch(name='height_sectional_stretches', axis=1)

plotting_elements = fishy.plot(show=False, opacity=0.3, color='#FFCD00')

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

height_sectional_stretches_b_spline_coefficients = m3l.Variable(
    name='height_sectional_stretches_b_spline_coefficients',
    shape=(1,),
    value=np.array([0.]),
)
height_sectional_stretches_b_spline_parameterization = BSpline(
    name='height_sectional_stretches_b_spline_parameterization',
    space=constant_1_dof_space,
    coefficients=height_sectional_stretches_b_spline_coefficients,
    num_physical_dimensions=1,
)

geometry_parameterization_solver.declare_state(name='length_sectional_translations_b_spline_coefficients', 
                                               state=length_sectional_translations_b_spline_coefficients)
geometry_parameterization_solver.declare_state(name='height_sectional_stretches_b_spline_coefficients', 
                                               state=height_sectional_stretches_b_spline_coefficients)
# endregion -Create Parameterization Objects

# region -Define Parameterization For Solver
parameterization_b_spline_input = np.linspace(0., 1., num_ffd_sections).reshape((-1,1))

length_sectional_translations = length_sectional_translations_b_spline_parameterization.evaluate(parameterization_b_spline_input)
width_sectional_stretches = width_sectional_stretches_b_spline_parameterization.evaluate(parameterization_b_spline_input)
height_sectional_stretches = height_sectional_stretches_b_spline_parameterization.evaluate(parameterization_b_spline_input)

sectional_parameters = {
    'length_sectional_translations' : length_sectional_translations,
    'width_sectional_stretches' : width_sectional_stretches,
    'height_sectional_stretches' : height_sectional_stretches
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

fishy_top = fishy.evaluate(fishy_top_parametric)
fishy_bottom = fishy.evaluate(fishy_bottom_parametric)
fishy_height = m3l.norm(fishy_top - fishy_bottom)
geometry_parameterization_solver.declare_input(name='fishy_height', input=fishy_height)

# endregion -Evaluate Parameterization Solver Inputs

# endregion -Evaluate Parameterization For Solver

# region -Evaluate Parameterization
fishy_length_input = system_model.create_input(name='fishy_length', shape=(1,), val=fishy_length.value, dv_flag=True, upper=0.7, lower=0.3, scaler=1.e1)
# fishy_length_input = system_model.create_input(name='fishy_length', shape=(1,), val=0.7, dv_flag=True, upper=1.)
fishy_height_input = system_model.create_input(name='fishy_height', shape=(1,), val=fishy_height.value, dv_flag=True, upper=0.09, lower=0.045, scaler=1.e2)
# fishy_height_input = system_model.create_input(name='fishy_height', shape=(1,), val=0.1, dv_flag=True, upper=0.1, lower=0.04)
optimization_inputs = {'fishy_length': fishy_length_input, 'fishy_height': fishy_height_input}
parameterization_solver_states = geometry_parameterization_solver.evaluate(optimization_inputs)

length_sectional_translations_b_spline_parameterization.coefficients = parameterization_solver_states['length_sectional_translations_b_spline_coefficients']
width_scaling_input = system_model.create_input(name='width_scaling', shape=(1,), val=-0.0, dv_flag=True, upper=0.01, lower=-0.02, scaler=1.e2)
width_sectional_stretches_b_spline_parameterization.coefficients = width_scaling_input
height_sectional_stretches_b_spline_parameterization.coefficients = parameterization_solver_states['height_sectional_stretches_b_spline_coefficients']

length_sectional_translations = length_sectional_translations_b_spline_parameterization.evaluate(parameterization_b_spline_input)
width_sectional_stretches = width_sectional_stretches_b_spline_parameterization.evaluate(parameterization_b_spline_input)
height_sectional_stretches = height_sectional_stretches_b_spline_parameterization.evaluate(parameterization_b_spline_input)

sectional_parameters = {
    'length_sectional_translations':length_sectional_translations,
    'width_sectional_stretches':width_sectional_stretches,
    'height_sectional_stretches':height_sectional_stretches
    }
ffd_block_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

fishy_coefficients = ffd_block.evaluate(coefficients=ffd_block_coefficients, plot=False)
fishy.coefficients = fishy_coefficients

# fishy.plot()
# exit()
# endregion Geometry Parameterization

# region Evaluate Meshes
# fishy_length = m3l.norm(fishy.evaluate(fishy_nose_parametric) - fishy.evaluate(fishy_tail_tip_parametric))

structural_mesh = fishy.evaluate(structural_mesh_parametric).reshape((-1,3))

# geometry_plot = fishy.plot(show=False, opacity=0.3)
# mesh_points = vedo.Points(structural_mesh.value, r=5, c='gold').opacity(0.6)
# plotter = vedo.Plotter()
# plotter.show([geometry_plot, mesh_points], axes=1, viewup='y')

front_point_in_structural_solver = m3l.Variable(shape=(3,), value=np.array([0.08 + 0.04, 0., 0.]))
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

displaced_mesh = structural_mesh + structural_displacements

# fishy_plot = fishy.plot(show=False, opacity=0.3)
# mesh_points = vedo.Points(displaced_mesh.value, r=5, c='gold').opacity(0.6)
# plotter = vedo.Plotter()
# plotter.show([fishy_plot, mesh_points], axes=1, viewup='y')

# endregion Structural Solver

# region Fit B-Spline to Displacements and Construct Deformed Geometry
displacement_space = BSplineSpace(
    name='structural_displacements_b_spline_space',
    order=(3,1,1),
    parametric_coefficients_shape=(10,1,1))

dummy_b_spline = BSpline(
    name='dummy_b_spline',
    space=displacement_space,
    coefficients=np.zeros(displacement_space.parametric_coefficients_shape + (3,)),
    num_physical_dimensions=3,
)
mesh_parametric_coordinates_in_displacement_space = structural_mesh_parametric.copy()
module_2_min_u = np.min(mesh_parametric_coordinates_in_displacement_space[:,0]) # Use the mesh projections to guarantee mesh is in module bounds
module_2_max_u = np.max(mesh_parametric_coordinates_in_displacement_space[:,0])
for i, old_u in enumerate(structural_mesh_parametric[:,0]):
    new_u = (old_u - module_2_min_u)/(module_2_max_u - module_2_min_u)
    mesh_parametric_coordinates_in_displacement_space[i,0] = new_u
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


coefficients_parametric = fishy.project(points=fishy.coefficients, grid_search_density=50, plot=False)
points_in_module_2_indices = np.where((coefficients_parametric[:,0] <= module_2_max_u) & (coefficients_parametric[:,0] >= module_2_min_u))[0]
coefficient_indices = np.arange(fishy.coefficients.value.size).reshape((-1,3))

deformed_fishy = fishy.copy()
# for coefficients in module, evaluate displacements at corresponding location and add them to the current coefficients
coefficients_parametric_in_module = coefficients_parametric[points_in_module_2_indices,:]
coefficients_parametric_in_module[:,0] = (coefficients_parametric_in_module[:,0] - module_2_min_u)/(module_2_max_u - module_2_min_u)
deformations_in_module = structural_displacements_b_spline.evaluate(coefficients_parametric_in_module)
# NOTE: For line above, a temporary structural displacement B-spline will have to be created for each time step. (where it's coefficients change)
indices = coefficient_indices[points_in_module_2_indices].reshape((-1))
deformed_fishy.coefficients[indices] = fishy.coefficients[indices] + deformations_in_module

parametric_coordinate_at_module_edge = np.array([[module_2_min_u, 0.5, 0.5]])
derivative_at_module_edge = deformed_fishy.evaluate(parametric_coordinate_at_module_edge, parametric_derivative_order=(1,0,0))
derivative_at_module_edge = derivative_at_module_edge/m3l.norm(derivative_at_module_edge)    # normalize
# endregion Fit B-Spline to Displacements and Construct Deformed Geometry

# region Compute Surface Area
num_elements_per_dimension = 50
parametric_mesh_2, parametric_mesh_1 = \
    np.meshgrid(np.linspace(0., 1., num_elements_per_dimension), np.linspace(0., 1., num_elements_per_dimension))
# parametric_grid_1 = np.zeros((num_elements_per_dimension**2, 3))
# parametric_grid_2 = np.zeros((num_elements_per_dimension**2, 3))
# parametric_grid_3 = np.zeros((num_elements_per_dimension**2, 3))
# parametric_grid_4 = np.zeros((num_elements_per_dimension**2, 3))
parametric_grid_5 = np.zeros((num_elements_per_dimension**2, 3))
parametric_grid_6 = np.zeros((num_elements_per_dimension**2, 3))

# parametric_grid_1[:,1] = parametric_mesh_1.flatten()
# parametric_grid_1[:,2] = parametric_mesh_2.flatten()
# parametric_grid_2[:,0] = np.ones_like(parametric_mesh_1.flatten())
# parametric_grid_2[:,1] = parametric_mesh_1.flatten()
# parametric_grid_2[:,2] = parametric_mesh_2.flatten()
# parametric_grid_3[:,0] = parametric_mesh_1.flatten()
# # parametric_grid_3[:,1] = np.zeros(parametric_mesh_1.flatten().shape)
# parametric_grid_3[:,2] = parametric_mesh_2.flatten()
# parametric_grid_4[:,0] = parametric_mesh_1.flatten()
# parametric_grid_4[:,1] = np.ones_like(parametric_mesh_1.flatten())
# parametric_grid_4[:,2] = parametric_mesh_2.flatten()
parametric_grid_5[:,0] = parametric_mesh_1.flatten()
parametric_grid_5[:,1] = parametric_mesh_2.flatten()
parametric_grid_6[:,0] = parametric_mesh_1.flatten()
parametric_grid_6[:,1] = parametric_mesh_2.flatten()
parametric_grid_6[:,2] = np.ones_like(parametric_mesh_1.flatten())

# parametric_grids = [parametric_grid_1, parametric_grid_2, parametric_grid_3, parametric_grid_4, parametric_grid_5, parametric_grid_6]
parametric_grids = [parametric_grid_5, parametric_grid_6]

surface_area = m3l.Variable(value=0, shape=(1, ))
# for i in range(6):
for i in range(2):
    surface_grid = fishy.evaluate(parametric_grids[i])

    indices = np.arange(num_elements_per_dimension**2 * 3).reshape((num_elements_per_dimension, num_elements_per_dimension, 3))
    u_end_indices = indices[1:, :, :].flatten()
    u_start_indices = indices[:-1, :, :].flatten()

    v_end_indices = indices[:, 1:, :].flatten()
    v_start_indices = indices[:, :-1, :].flatten()
    
    coords_u_end = surface_grid[u_end_indices].reshape((num_elements_per_dimension-1, num_elements_per_dimension, 3))
    coords_u_start = surface_grid[u_start_indices].reshape((num_elements_per_dimension-1, num_elements_per_dimension, 3))

    coords_v_end = surface_grid[v_end_indices].reshape((num_elements_per_dimension-1, num_elements_per_dimension, 3))
    coords_v_start = surface_grid[v_start_indices].reshape((num_elements_per_dimension-1, num_elements_per_dimension, 3))


    indices = np.arange(num_elements_per_dimension* (num_elements_per_dimension-1)  * 3).reshape((num_elements_per_dimension-1, num_elements_per_dimension, 3))
    v_start_indices = indices[:, :-1, :].flatten()
    v_end_indices = indices[:, 1:, :].flatten()
    u_vectors = coords_u_end - coords_u_start
    u_vectors_start = u_vectors.reshape((-1, ))
    u_vectors_1 = u_vectors_start[v_start_indices].reshape((num_elements_per_dimension-1, num_elements_per_dimension-1, 3))
    u_vectors_2 = u_vectors_start[v_end_indices].reshape((num_elements_per_dimension-1, num_elements_per_dimension-1, 3))

    indices = np.arange(num_elements_per_dimension*(num_elements_per_dimension-1) * 3).reshape((num_elements_per_dimension, num_elements_per_dimension-1, 3))
    u_start_indices = indices[:-1, :, :].flatten()
    u_end_indices = indices[1:, :, :].flatten()
    v_vectors = coords_v_end - coords_v_start
    v_vectors_start = v_vectors.reshape((-1, ))
    v_vectors_1 = v_vectors_start[u_start_indices].reshape((num_elements_per_dimension-1, num_elements_per_dimension-1, 3))
    v_vectors_2 = v_vectors_start[u_end_indices].reshape((num_elements_per_dimension-1, num_elements_per_dimension-1, 3))

    area_vectors_left_lower = m3l.cross(u_vectors_1, v_vectors_2, axis=2)
    area_vectors_right_upper = m3l.cross(v_vectors_1, u_vectors_2, axis=2)
    area_magnitudes_left_lower = m3l.norm(area_vectors_left_lower, order=2, axes=(-1, ))
    area_magnitudes_right_upper = m3l.norm(area_vectors_right_upper, order=2, axes=(-1, ))
    area_magnitudes = (area_magnitudes_left_lower + area_magnitudes_right_upper)/2
    wireframe_area = m3l.sum(area_magnitudes, axes=(0, 1)).reshape((1, ))
    surface_area =  surface_area + wireframe_area


# endregion Compute Surface Area


# region Objective Model
max_tip_displacement_node = np.argmax(structural_displacements.value.reshape((-1,3))[:,2])
flattened_index = max_tip_displacement_node*3 + 2
# surface_area_penalty = 4.       # shrinks the length and height a bit smaller (width lower bound)
# surface_area_penalty = 2.     # just shrinks the width
surface_area_penalty = 1.     # Makes the length and height a bit larger (width lower bound)
# surface_area_penalty = 1.e-10

initial_angle = m3l.Variable(shape=(3,), value=np.array([1., 0., 0.]))
angle = m3l.arccos(m3l.dot(initial_angle, derivative_at_module_edge))
system_model.register_output(angle)

objective = -angle

# new_module_length = fishy_length_input*(0.08/fishy_length.value)    # use initial values to find ratio of module length to total length
# objective = -angle/new_module_length

module_tip_displacement = structural_displacements_flattened[int(flattened_index)]
module_average_curvature = (derivative_at_module_edge[2]/derivative_at_module_edge[0]) / fishy_length_input
system_model.register_output(module_average_curvature)
# objective = -module_average_curvature

# objective = -((structural_displacements_flattened[int(flattened_index)]))/(fishy_length_input**2)# + surface_area_penalty*surface_area**2
# objective = -((structural_displacements_flattened[int(flattened_index)]))/(fishy_length_input)# + surface_area_penalty*surface_area**2
# objective = -(structural_displacements_flattened[int(flattened_index)]**2) + surface_area_penalty*surface_area**2

system_model.register_output(objective)
# system_model.add_objective(objective, scaler=1.e0)
system_model.add_objective(objective, scaler=1.e1)
# system_model.add_objective(objective, scaler=1.e6)
# endregion Objective Model

csdl_model = system_model.assemble()
sim = python_csdl_backend.Simulator(csdl_model, display_scripts=True)

import time
t1 = time.time()
sim.run()
t2 = time.time()
print('Elapsed Time: ', t2 - t1)
sim.check_totals()
exit()

# height_values = np.linspace(0.04, 0.09, 20)
# angle = np.zeros_like(height_values)
# average_curvature = np.zeros_like(height_values)
# for i, height in enumerate(height_values):
#     print(i)
#     sim[fishy_height_input.name] = height
#     sim.run()
#     angle[i] = -sim[objective.operation.name + '.' + objective.name]*180/np.pi
#     # average_curvature[i] = -sim[objective.operation.name + '.' + objective.name] / sim[fishy_length_input.name]
#     average_curvature[i] = sim[module_average_curvature.operation.name + '.' + module_average_curvature.name]


# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(height_values, angle)
# plt.title('Actuator Angle vs Height (No optimization)')
# plt.xlabel('Height')
# plt.ylabel('Actuator Angle')
# plt.figure()
# plt.plot(height_values, average_curvature)
# plt.title('Average Curvature vs Height (No optimization)')
# plt.xlabel('Height')
# plt.ylabel('Average Curvature')
# plt.show()
# # exit()

# width_values = np.linspace(-0.02, 0.01, 10)
# angle = np.zeros_like(width_values)
# average_curvature = np.zeros_like(width_values)
# for i, width in enumerate(width_values):
#     print(i)
#     sim[width_scaling_input.name] = width
#     sim.run()
#     angle[i] = -sim[objective.operation.name + '.' + objective.name]*180/np.pi
#     # average_curvature[i] = -sim[objective.operation.name + '.' + objective.name] / sim[fishy_length_input.name]
#     average_curvature[i] = sim[module_average_curvature.operation.name + '.' + module_average_curvature.name]


# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(width_values, angle)
# plt.title('Actuator Angle vs Change In Width (No optimization)')
# plt.xlabel('Change In Width')
# plt.ylabel('Actuator Angle')
# plt.figure()
# plt.plot(width_values, average_curvature)
# plt.title('Average Curvature vs Change In Width (No optimization)')
# plt.xlabel('Change In Width')
# plt.ylabel('Average Curvature')
# plt.show()
# exit()


# length_values = np.linspace(0.3, 1., 15)
# angle = np.zeros_like(length_values)
# average_curvature = np.zeros_like(length_values)
# for i, length in enumerate(length_values):
#     print(i)
#     sim[fishy_length_input.name] = length
#     sim.run()
#     angle[i] = -sim[objective.operation.name + '.' + objective.name]*180/np.pi
#     # average_curvature[i] = -sim[objective.operation.name + '.' + objective.name] / sim[fishy_length_input.name]
#     average_curvature[i] = sim[module_average_curvature.operation.name + '.' + module_average_curvature.name]

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(length_values, angle)
# plt.title('Actuator Angle vs Length (No optimization)')
# plt.xlabel('Length')
# plt.ylabel('Actuator Angle')
# plt.figure()
# plt.plot(length_values, average_curvature)
# plt.title('Average Curvature vs Length (No optimization)')
# plt.xlabel('Length')
# plt.ylabel('Average Curvature')
# plt.show()
# exit()

sim[fishy_length_input.name] = 0.389
sim[fishy_height_input.name] = 0.09
sim[width_scaling_input.name] = -0.02
sim.run()
new_coefficients = sim[fishy.coefficients.operation.name + '.' + fishy.coefficients.name]
fishy.coefficients = m3l.Variable(value=new_coefficients, shape=fishy.coefficients.shape)
fishy.plot(additional_plotting_elements=plotting_elements, opacity=0.8)
exit()

# region Optimization
prob = CSDLProblem(problem_name='fishy_optimization', simulator=sim)
optimizer = SLSQP(prob, maxiter=40, ftol=1E-6)
import time
t1 = time.time()
optimizer.solve()
t2 = time.time()
print('Elapsed Time: ', t2 - t1)
optimizer.print_results()

print('Initial Objective: ', objective.value)
print('Initial Length', fishy_length_input.value)
print('Initial Height', fishy_height_input.value)
print('Initial Surface Area', surface_area.value)
print('Initial Angle: ', angle.value)

print('Optimized Objective: ', sim[objective.operation.name + '.' + objective.name])
print('Optimized Length', sim[fishy_length_input.name])
print('Optimized Width Scaling', sim[width_scaling_input.name])
print('Optimized Height', sim[fishy_height_input.name])
# print('Optimized Surface Area', sim[surface_area.operation.name + '.' + surface_area.name])
print('Optimized Angle: ', sim[angle.operation.name + '.' + angle.name])
# endregion Optimization

# import time
# t1 = time.time()
# width_values = np.linspace(-0.02, 0.01, 15)
# angle = np.zeros_like(width_values)
# average_curvature = np.zeros_like(width_values)
# for i, width in enumerate(width_values):
#     print('-----------------------------------------------')
#     print(i)
#     print('-----------------------------------------------')
#     sim[width_scaling_input.name] = width
#     optimizer.solve()
#     angle[i] = -sim[objective.operation.name + '.' + objective.name]*180/np.pi
#     # average_curvature[i] = -sim[objective.operation.name + '.' + objective.name] / sim[fishy_length_input.name]
#     average_curvature[i] = sim[module_average_curvature.operation.name + '.' + module_average_curvature.name]

#     optimizer.print_results()
#     print('Initial Objective: ', objective.value)
#     print('Initial Length', fishy_length_input.value)
#     print('Initial Height', fishy_height_input.value)
#     print('Initial Surface Area', surface_area.value)
#     # print('Initial Angle: ', angle.value)

#     print('Optimized Objective: ', sim[objective.operation.name + '.' + objective.name])
#     print('Optimized Length', sim[fishy_length_input.name])
#     print('Optimized Width Scaling', sim[width_scaling_input.name])
#     print('Optimized Height', sim[fishy_height_input.name])
#     # print('Optimized Surface Area', sim[surface_area.operation.name + '.' + surface_area.name])
#     # print('Optimized Angle: ', sim[angle.operation.name + '.' + angle.name])
# t2 = time.time()
# print('Elapsed Time: ', t2 - t1)


# plt.figure()
# plt.plot(width_values, angle, '-o')
# plt.title('Actuator Angle vs Change In Width (With optimization)')
# plt.xlabel('Change In Width')
# plt.ylabel('Actuator Angle')
# plt.figure()
# plt.plot(width_values, average_curvature, '-o')
# plt.title('Average Curvature vs Change In Width (With optimization)')
# plt.xlabel('Change In Width')
# plt.ylabel('Average Curvature')
# plt.show()
# exit()