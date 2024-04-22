import lsdo_geo
from lsdo_geo.splines.b_splines.b_spline import BSpline
import pickle
import meshio
import numpy as np
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
Design variables: 
1. Length
2. Change in Width
3. Height
4. Chamber thickness

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
    with open("examples/advanced_examples/robotic_fish/pickle_files/fishy_volume_geometry.pickle", 'rb') as handle:
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

structural_mesh_parametric = fishy.project(points=structural_mesh_nodes, grid_search_density=65, max_iterations=500, plot=False)
structural_module_front_parametric = fishy.project(points=np.array([[0.08 + 0.12, 0., 0.]]), grid_search_density=150, plot=False)
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
fishy_height_input = system_model.create_input(name='fishy_height', shape=(1,), val=fishy_height.value, dv_flag=True, upper=0.09, lower=0.04, scaler=1.e2)
# fishy_height_input = system_model.create_input(name='fishy_height', shape=(1,), val=0.1, dv_flag=True, upper=0.1, lower=0.04)
optimization_inputs = {'fishy_length': fishy_length_input, 'fishy_height': fishy_height_input}
parameterization_solver_states = geometry_parameterization_solver.evaluate(optimization_inputs)

length_sectional_translations_b_spline_parameterization.coefficients = parameterization_solver_states['length_sectional_translations_b_spline_coefficients']
width_scaling_input = system_model.create_input(name='width_scaling', shape=(1,), val=-0.0, dv_flag=True, upper=0.01, lower=-0.015, scaler=1.e2)
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

displaced_mesh = structural_mesh + structural_displacements

# fishy_plot = fishy.plot(show=False, opacity=0.3)
# mesh_points = vedo.Points(displaced_mesh.value, r=5, c='gold').opacity(0.6)
# plotter = vedo.Plotter()
# plotter.show([fishy_plot, mesh_points], axes=1, viewup='y')

# endregion Structural Solver

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
objective = -((structural_displacements_flattened[int(flattened_index)]))/(fishy_length_input**2) + surface_area_penalty*surface_area**2
# objective = -(structural_displacements_flattened[int(flattened_index)]**2) + surface_area_penalty*surface_area**2

system_model.register_output(objective)
system_model.add_objective(objective, scaler=1.e2)
# endregion Objective Model

csdl_model = system_model.assemble()
sim = python_csdl_backend.Simulator(csdl_model)

# sim.run()
# sim.check_totals()
# exit()

# region Optimization
prob = CSDLProblem(problem_name='fishy_optimization', simulator=sim)
optimizer = SLSQP(prob, maxiter=100, ftol=1E-12)
optimizer.solve()
optimizer.print_results()

print('Initial Objective: ', objective.value)
print('Initial Length', fishy_length_input.value)
print('Initial Height', fishy_height_input.value)
print('Initial Surface Area', surface_area.value)

print('Optimized Objective: ', sim[objective.operation.name + '.' + objective.name])
print('Optimized Length', sim[fishy_length_input.name])
print('Optimized Width Scaling', sim[width_scaling_input.name])
print('Optimized Height', sim[fishy_height_input.name])
print('Optimized Surface Area', sim[surface_area.operation.name + '.' + surface_area.name])
# endregion Optimization
