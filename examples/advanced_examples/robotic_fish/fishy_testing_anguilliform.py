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
    with open("examples/advanced_examples/robotic_fish/pickle_files/fishy_volume_geometry.pickle", 'rb') as handle:
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
# structural_mesh_nodes = structural_mesh.points/1000 + np.array([0.08 + 0.04, 0, 0.])   # Shift the mesh to the right to make it the front module
structural_mesh_nodes = structural_mesh.points/1000 + np.array([0.04, 0., 0.])   # Shift the mesh to the right to make it the front module

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
    grid_search_density = 65
elif mesh_name == "module_v1_fine":
    grid_search_density = 100
else:
    raise Exception("Specify grid search for this mesh.")
structural_mesh_parametric = fishy.project(points=structural_mesh_nodes, grid_search_density=grid_search_density, max_iterations=500, plot=False)
# structural_mesh_parametric = fishy.project(points=structural_mesh_nodes, grid_search_density=100, max_iterations=500, plot=False)
# structural_module_front_parametric = fishy.project(points=np.array([[0.08 + 0.12, 0., 0.]]), grid_search_density=150, plot=False)
structural_module_front_parametric = fishy.project(points=np.array([[0.12, 0., 0.]]), grid_search_density=150, plot=False)
# structural_module_back_parametric = fishy.project(points=np.array([[0.12, 0., 0.]]), grid_search_density=150, plot=False)
structural_module_back_parametric = fishy.project(points=np.array([[0.04, 0., 0.]]), grid_search_density=150, plot=False)
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

# front_point_in_structural_solver = m3l.Variable(shape=(3,), value=np.array([0.08 + 0.12, 0., 0.]))
front_point_in_structural_solver = m3l.Variable(shape=(3,), value=np.array([0.08, 0., 0.]))
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
# module_1_min_u = np.min(mesh_parametric_coordinates_in_displacement_space[:,0])
# module_1_max_u = np.max(mesh_parametric_coordinates_in_displacement_space[:,0])
module_2_min_u = np.min(mesh_parametric_coordinates_in_displacement_space[:,0]) # Use the mesh projections to guarantee mesh is in module bounds
module_2_max_u = np.max(mesh_parametric_coordinates_in_displacement_space[:,0])

# module_1_min_u = fishy.project(points=np.array([[0. + 0.08 + 0.04, 0., 0.]]), grid_search_density=100, plot=True)[0,0]
module_1_min_u = module_2_max_u
module_1_max_u = fishy.project(points=np.array([[0.08 + 0.08 + 0.04, 0., 0.]]), grid_search_density=100, plot=False)[0,0]
# module_2_min_u = fishy.project(points=np.array([[0. + 0.04, 0., 0.]]), grid_search_density=100, plot=True)[0,0]
# module_2_max_u = fishy.project(points=np.array([[0.08 + 0.04, 0., 0.]]), grid_search_density=100, plot=True)[0,0]

module_3_min_u = fishy.project(points=np.array([[0. - 0.1 + 0.04, 0., 0.]]), grid_search_density=100, plot=False)[0,0]
# module_3_max_u = fishy.project(points=np.array([[0.08 - 0.1 + 0.04, 0., 0.]]), grid_search_density=100, plot=True)[0,0]
module_3_max_u = module_2_min_u
for i, old_u in enumerate(structural_mesh_parametric[:,0]):
    new_u = (old_u - module_2_min_u)/(module_2_max_u - module_2_min_u)
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

opposite_structural_displacements_b_spline = structural_displacements_b_spline.copy()
coefficients_shape = opposite_structural_displacements_b_spline.space.parametric_coefficients_shape + (3,)
indices = np.arange(np.prod(coefficients_shape)).reshape(coefficients_shape)
# want to flip the z component of the coefficients and invert the ordering of the x component along the 3rd axis of coefficients
z_indicies = indices[:,:,:,2].reshape((-1,))
for i in range(coefficients_shape[2]):
    # flip x component
    set_indices = indices[:,:,i,0].reshape((-1,))
    get_indices = indices[:,:,coefficients_shape[2]-i-1,0].reshape((-1,))
    opposite_structural_displacements_b_spline.coefficients[set_indices] = structural_displacements_b_spline.coefficients[get_indices]

    # flip z component
    set_indices = indices[:,:,i,2].reshape((-1,))
    get_indices = indices[:,:,coefficients_shape[2]-i-1,2].reshape((-1,))
    opposite_structural_displacements_b_spline.coefficients[set_indices] = -structural_displacements_b_spline.coefficients[get_indices]


# Interpolate between the two displacement fields
time_constant = 3
actuation_frequency = 1.
actuation_period = 1/actuation_frequency
# num_vast_time_steps = 7
# num_vast_time_steps = 41
# num_vast_time_steps = 81
num_cycles = 3
num_modules = 3
num_steps_per_cycle = 33
num_vast_time_steps = num_steps_per_cycle*num_cycles
time = np.linspace(0, actuation_period*num_cycles, num_vast_time_steps)
num_strokes = int(actuation_frequency*time[-1])*2+1

def compute_chamber_pressure_function(t, pressure_inputs, time_constant, p0, evaluation_t):
    if len(t) != len(pressure_inputs):
        raise ValueError('t and pressure_inputs must have the same length')
    
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

left_chamber_inputs = []
right_chamber_inputs = []
max_weight = 2.
for stroke_index in range(num_strokes):
    if stroke_index % 2 == 0:
        left_chamber_inputs.append(max_weight)
        right_chamber_inputs.append(0.)
    else:
        left_chamber_inputs.append(0.)
        right_chamber_inputs.append(max_weight)

t_pressure_inputs = np.linspace(0, time[-1], int(num_strokes))
t_pressure_inputs[1:] = t_pressure_inputs[1:] - (actuation_period/2)/2
initial_weight = max_weight/2
deformed_fishy_weights = compute_chamber_pressure_function(t_pressure_inputs, left_chamber_inputs, time_constant, initial_weight, time)

# import matplotlib.pyplot as plt
# plt.plot(time, deformed_fishy_weights)
# plt.plot(time[:-time.shape[0]//9] + time[time.shape[0]//9], deformed_fishy_weights[:-time.shape[0]//9])
# plt.plot(time[:-2*time.shape[0]//9] + time[2*time.shape[0]//9], deformed_fishy_weights[:-2*time.shape[0]//9])
# plt.show()

new_deformed_fishy_weights = []
new_opposite_deformed_fishy_weights = []
deformed_side_interpolation_indices = []
opposite_deformed_side_interpolation_indices = []
for i, weight in enumerate(deformed_fishy_weights):
    if weight >= 1:
        new_deformed_fishy_weights.append(weight - 1)     # - 1 rescales 1-2 to 0-1
        deformed_side_interpolation_indices.append(i)
    else:
        new_opposite_deformed_fishy_weights.append(1 - weight)  # 1- flips the sign so it's the weight on the deformed fishy
        opposite_deformed_side_interpolation_indices.append(i)
new_deformed_fishy_weights = np.array(new_deformed_fishy_weights)
new_opposite_deformed_fishy_weights = np.array(new_opposite_deformed_fishy_weights)

# opposite_deformed_fishy_weights = 1 - deformed_fishy_weights

# actuating_displacement_coefficients = m3l.linear_combination(structural_displacements_coefficients,
#                                                       opposite_structural_displacements_b_spline.coefficients,
#                                                    start_weights=deformed_fishy_weights, stop_weights=opposite_deformed_fishy_weights)
indices = np.arange(time.shape[0]*structural_displacements_coefficients.value.size).reshape((time.shape[0], structural_displacements_coefficients.value.size))
zero_displacements = m3l.Variable(shape=structural_displacements_coefficients.shape, value=np.zeros(structural_displacements_coefficients.shape))
actuating_displacement_coefficients_deformed_side = m3l.linear_combination(structural_displacements_coefficients,
                                                      zero_displacements,
                                                   start_weights=new_deformed_fishy_weights, stop_weights=1-new_deformed_fishy_weights)
actuating_displacement_coefficients_opposite_deformed_side = m3l.linear_combination(opposite_structural_displacements_b_spline.coefficients,
                                                      zero_displacements,
                                                   start_weights=new_opposite_deformed_fishy_weights, stop_weights=1-new_opposite_deformed_fishy_weights)

actuating_displacement_coefficients = m3l.Variable(name='actuating_displacement_coefficients', 
                                                   shape=(time.shape[0]*structural_displacements_coefficients.value.size,), 
                                                   value=np.zeros(time.shape[0]*structural_displacements_coefficients.value.size))
deformed_side_indices = indices[np.array(deformed_side_interpolation_indices),:].reshape((-1,))
opposite_side_indices = indices[np.array(opposite_deformed_side_interpolation_indices),:].reshape((-1,))

actuating_displacement_coefficients[deformed_side_indices] = actuating_displacement_coefficients_deformed_side.reshape((-1,))
actuating_displacement_coefficients[opposite_side_indices] = actuating_displacement_coefficients_opposite_deformed_side.reshape((-1,))

actuating_displacement_coefficients = \
    actuating_displacement_coefficients.reshape((-1,))

# region add displacements directly to coefficients (more efficient because less fitting but questionable accuracy)

# Project geometry coefficients onto geometry to get parametric coordinates (should be about evenly spaced)
# NOTE: This is a bit silly because we already have the parametric coordinates, but I don't want to figure it out.
deformed_fishy = fishy.copy()
coefficients_parametric = fishy.project(points=fishy.coefficients, grid_search_density=50, plot=False)
coefficient_indices = np.arange(fishy.coefficients.value.size).reshape((-1,3))
# points_in_front_indices = np.where(coefficients_parametric[:,0] > module_1_max_u)[0]
points_in_module_1_indices = np.where((coefficients_parametric[:,0] <= module_1_max_u) & (coefficients_parametric[:,0] >= module_1_min_u))[0]
points_behind_module_1_indices = np.where(coefficients_parametric[:,0] < module_1_min_u)[0]
points_in_module_2_indices = np.where((coefficients_parametric[:,0] <= module_2_max_u) & (coefficients_parametric[:,0] >= module_2_min_u))[0]
points_behind_module_2_indices = np.where(coefficients_parametric[:,0] < module_2_min_u)[0]
points_in_module_3_indices = np.where((coefficients_parametric[:,0] <= module_3_max_u) & (coefficients_parametric[:,0] >= module_3_min_u))[0]
points_behind_module_3_indices = np.where(coefficients_parametric[:,0] < module_3_min_u)[0]

module_1_displacements_b_spline_at_time_t = structural_displacements_b_spline.copy()
module_2_displacements_b_spline_at_time_t = structural_displacements_b_spline.copy()
module_3_displacements_b_spline_at_time_t = structural_displacements_b_spline.copy()
displacement_indices = np.arange(time.shape[0]*structural_displacements_coefficients.shape[0]).reshape(
                                (time.shape[0], structural_displacements_coefficients.shape[0]))
fishy_plot = fishy.plot(show=False, opacity=0.3)
video = vedo.Video('examples/advanced_examples/robotic_fish/temp/fishy_wiggle.mp4', fps=num_steps_per_cycle/2, backend='cv')
for i in range(time.shape[0]):
    # region Module 1
    module_displacement_coefficients = actuating_displacement_coefficients[displacement_indices[i,:].reshape((-1,))]
    module_1_displacements_b_spline_at_time_t.coefficients = module_displacement_coefficients

    # displacements for in from of module are 0, so don't have deformation

    # for coefficients in module, evaluate displacements at corresponding location and add them to the current coefficients
    coefficients_parametric_in_module = coefficients_parametric[points_in_module_1_indices,:]
    coefficients_parametric_in_module[:,0] = (coefficients_parametric_in_module[:,0] - module_1_min_u)/(module_1_max_u - module_1_min_u)
    deformations_in_module = module_1_displacements_b_spline_at_time_t.evaluate(coefficients_parametric_in_module)
    # NOTE: For line above, a temporary structural displacement B-spline will have to be created for each time step. (where it's coefficients change)
    indices = coefficient_indices[points_in_module_1_indices].reshape((-1))
    deformed_fishy.coefficients[indices] = fishy.coefficients[indices] + deformations_in_module

    # For coefficients behind the module, evaluate the derivative at the end of the module and apply the rigid link
    coefficients_parametric_behind_module = coefficients_parametric[points_behind_module_1_indices,:]
    # edge_parametric_coordinates = np.array([0., coefficients_parametric_behind_module[0,1], coefficients_parametric_behind_module[0,2]])
    edge_parametric_coordinates_u = np.zeros((coefficients_parametric_behind_module.shape[0],1))
    edge_parametric_coordinates_v = coefficients_parametric_behind_module[:,1].reshape((-1,1))
    edge_parametric_coordinates_w = coefficients_parametric_behind_module[:,2].reshape((-1,1))
    edge_parametric_coordinates = np.hstack((edge_parametric_coordinates_u, edge_parametric_coordinates_v, edge_parametric_coordinates_w))
    derivative_values_at_edge = module_1_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates, parametric_derivative_order=(1,0,0))
    displacement_values_at_edge = module_1_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates)
    delta_u = (coefficients_parametric_behind_module[:,0] - module_1_min_u)/(module_1_max_u - module_1_min_u)  # need to rescale delta_u or derivative due to parametric space mismatch
    delta_u = np.repeat(delta_u, 3)   # repeat each one 3 times to scale the x,y,z
    displacements_at_coefficients = displacement_values_at_edge + (derivative_values_at_edge*delta_u).reshape((-1,))
    indices = coefficient_indices[points_behind_module_1_indices].reshape((-1))
    deformed_fishy.coefficients[indices] = fishy.coefficients[indices] + displacements_at_coefficients

    # endregion Module 1

    # region Module 2
    time_step_offset = 1*num_vast_time_steps//(num_cycles*num_modules)   # (divide by 2 to get one period, then phase offset by 1/3 of period)
    if i >= time_step_offset:
        module_displacement_coefficients = actuating_displacement_coefficients[displacement_indices[i - time_step_offset,:].reshape((-1,))]
        module_2_displacements_b_spline_at_time_t.coefficients = module_displacement_coefficients

        # displacements for in from of module are 0, so don't have deformation

        # for coefficients in module, evaluate displacements at corresponding location and add them to the current coefficients
        coefficients_parametric_in_module = coefficients_parametric[points_in_module_2_indices,:]
        coefficients_parametric_in_module[:,0] = (coefficients_parametric_in_module[:,0] - module_2_min_u)/(module_2_max_u - module_2_min_u)
        deformations_in_module = module_2_displacements_b_spline_at_time_t.evaluate(coefficients_parametric_in_module)
        # NOTE: For line above, a temporary structural displacement B-spline will have to be created for each time step. (where it's coefficients change)
        indices = coefficient_indices[points_in_module_2_indices].reshape((-1))
        deformed_fishy.coefficients[indices] = deformed_fishy.coefficients[indices] + deformations_in_module

        # For coefficients behind the module, evaluate the derivative at the end of the module and apply the rigid link
        coefficients_parametric_behind_module = coefficients_parametric[points_behind_module_2_indices,:]
        # edge_parametric_coordinates = np.array([0., coefficients_parametric_behind_module[0,1], coefficients_parametric_behind_module[0,2]])
        edge_parametric_coordinates_u = np.zeros((coefficients_parametric_behind_module.shape[0],1))
        edge_parametric_coordinates_v = coefficients_parametric_behind_module[:,1].reshape((-1,1))
        edge_parametric_coordinates_w = coefficients_parametric_behind_module[:,2].reshape((-1,1))
        edge_parametric_coordinates = np.hstack((edge_parametric_coordinates_u, edge_parametric_coordinates_v, edge_parametric_coordinates_w))
        derivative_values_at_edge = module_2_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates, parametric_derivative_order=(1,0,0))
        displacement_values_at_edge = module_2_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates)
        delta_u = (coefficients_parametric_behind_module[:,0] - module_2_min_u)/(module_2_max_u - module_2_min_u)
        delta_u = np.repeat(delta_u, 3)   # repeat each one 3 times to scale the x,y,z
        displacements_at_coefficients = displacement_values_at_edge + (derivative_values_at_edge*delta_u).reshape((-1,))
        indices = coefficient_indices[points_behind_module_2_indices].reshape((-1))
        deformed_fishy.coefficients[indices] = deformed_fishy.coefficients[indices] + displacements_at_coefficients
    # endregion Module 2

    # region Module 3
    time_step_offset = 2*(num_cycles*num_modules)   # (divide by 2 to get one period, then phase offset by 2/3 of period)
    if i >= time_step_offset:
        module_displacement_coefficients = actuating_displacement_coefficients[displacement_indices[i - time_step_offset,:].reshape((-1,))]
        module_3_displacements_b_spline_at_time_t.coefficients = module_displacement_coefficients

        # displacements for in from of module are 0, so don't have deformation

        # for coefficients in module, evaluate displacements at corresponding location and add them to the current coefficients
        coefficients_parametric_in_module = coefficients_parametric[points_in_module_3_indices,:]
        coefficients_parametric_in_module[:,0] = (coefficients_parametric_in_module[:,0] - module_3_min_u)/(module_3_max_u - module_3_min_u)
        deformations_in_module = module_3_displacements_b_spline_at_time_t.evaluate(coefficients_parametric_in_module)
        # NOTE: For line above, a temporary structural displacement B-spline will have to be created for each time step. (where it's coefficients change)
        indices = coefficient_indices[points_in_module_3_indices].reshape((-1))
        deformed_fishy.coefficients[indices] = deformed_fishy.coefficients[indices] + deformations_in_module

        # For coefficients behind the module, evaluate the derivative at the end of the module and apply the rigid link
        coefficients_parametric_behind_module = coefficients_parametric[points_behind_module_3_indices,:]
        # edge_parametric_coordinates = np.array([0., coefficients_parametric_behind_module[0,1], coefficients_parametric_behind_module[0,2]])
        edge_parametric_coordinates_u = np.zeros((coefficients_parametric_behind_module.shape[0],1))
        edge_parametric_coordinates_v = coefficients_parametric_behind_module[:,1].reshape((-1,1))
        edge_parametric_coordinates_w = coefficients_parametric_behind_module[:,2].reshape((-1,1))
        edge_parametric_coordinates = np.hstack((edge_parametric_coordinates_u, edge_parametric_coordinates_v, edge_parametric_coordinates_w))
        derivative_values_at_edge = module_3_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates, parametric_derivative_order=(1,0,0))
        displacement_values_at_edge = module_3_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates)
        delta_u = (coefficients_parametric_behind_module[:,0] - module_3_min_u)/(module_3_max_u - module_3_min_u)
        delta_u = np.repeat(delta_u, 3)   # repeat each one 3 times to scale the x,y,z
        displacements_at_coefficients = displacement_values_at_edge + (derivative_values_at_edge*delta_u).reshape((-1,))
        indices = coefficient_indices[points_behind_module_3_indices].reshape((-1))
        deformed_fishy.coefficients[indices] = deformed_fishy.coefficients[indices] + displacements_at_coefficients
    # endregion Module 3

    deformed_fishy_plot = deformed_fishy.plot(opacity=0.8, show=False)
    plotter = vedo.Plotter(offscreen=True)
    plotter.show([fishy_plot, deformed_fishy_plot], axes=1, viewup='y')
    video.add_frame()
video.close()
# endregion
exit()



for i in range(coefficients_parametric.shape[0]):
    u = coefficients_parametric[i,0]
    if u > module_1_max_u:   # If in front of module, don't do anything
        continue    
    elif u > module_1_min_u: # If in module, evaluate displacement at that parametric coordinate
        coefficients_parametric_scaled_u = (coefficients_parametric[i,0] - module_1_min_u)/(module_1_max_u - module_1_min_u)
        coefficients_parametric_scaled = coefficients_parametric[i].copy()
        coefficients_parametric_scaled[0] = coefficients_parametric_scaled_u
        displacement_at_point = structural_displacements_b_spline.evaluate(coefficients_parametric_scaled.reshape((1,3)))
    else:       # If behind module, use derivative at the edge of the module and linearly interpolate to get displacement
        edge_parametric_coordinate = np.array([0., coefficients_parametric[i,1], coefficients_parametric[i,2]])
        derivative_values_at_edge = structural_displacements_b_spline.evaluate(edge_parametric_coordinate, parametric_derivative_order=(1,0,0))
        displacement_values_at_edge = structural_displacements_b_spline.evaluate(edge_parametric_coordinate)

        delta_u = (u - module_1_min_u)/(module_1_max_u - module_1_min_u)
        displacement_at_point = displacement_values_at_edge + (derivative_values_at_edge*delta_u).reshape((-1,))

    ind = coefficient_indices[i,:].reshape((-1,))
    deformed_fishy.coefficients[ind] = fishy.coefficients[ind] + displacement_at_point

deformed_fishy_plot = deformed_fishy.plot(opacity=0.8, show=True)
exit()




















# num_u_stuff = 50
# num_v_stuff = 25
# num_w_stuff = 25
num_u_stuff = 40
num_v_stuff = 15
num_w_stuff = 15

displacements_b_spline_at_time_t = structural_displacements_b_spline.copy()

displacement_indices = np.arange(time.shape[0]*structural_displacements_coefficients.shape[0]).reshape(
                                (time.shape[0], structural_displacements_coefficients.shape[0]))
fishy_plot = fishy.plot(show=False, opacity=0.3)
video = vedo.Video('examples/advanced_examples/robotic_fish/temp/fishy_wiggle.mp4', fps=21, backend='cv')
for i in range(time.shape[0]):
    get_indices = displacement_indices[i,:].reshape((-1,))
    module_displacement_coefficients = actuating_displacement_coefficients[get_indices]

    displacements_b_spline_at_time_t.coefficients = module_displacement_coefficients

    edge_u_coordinate = module_1_min_u
    w_coordinates, v_coordinates = np.meshgrid(np.linspace(0., 1., num_v_stuff), np.linspace(0., 1., num_w_stuff))
    u_coordinates = np.ones_like(v_coordinates)*0.
    derivative_parametric_coordinates = np.stack((u_coordinates.flatten(), v_coordinates.flatten(), w_coordinates.flatten()), axis=1)
    derivative_values_at_edge = displacements_b_spline_at_time_t.evaluate(derivative_parametric_coordinates, parametric_derivative_order=(1,0,0))

    displacement_values_at_edge = displacements_b_spline_at_time_t.evaluate(derivative_parametric_coordinates)

    u_stuff = np.linspace(0., edge_u_coordinate, num_u_stuff, endpoint=False)
    stuffing_points = m3l.Variable(shape=(num_u_stuff*num_v_stuff*num_w_stuff*3,), value=np.zeros(num_u_stuff*num_v_stuff*num_w_stuff*3))
    stuffing_parametric_coordinates = np.zeros((num_u_stuff*num_v_stuff*num_w_stuff, 3))

    # This is slow because of lots of small-ish indexing (It takes a little bit of time but not significant (~0.5 seconds I think))
    for i, u in enumerate(u_stuff):
        delta_u = (u - edge_u_coordinate)/(module_1_max_u - module_1_min_u)
        flattened_indices = np.arange(i*num_v_stuff*num_w_stuff*3, (i+1)*num_v_stuff*num_w_stuff*3)
        stuffing_points[flattened_indices] = displacement_values_at_edge + (derivative_values_at_edge*delta_u).reshape((-1,))
        stuffing_parametric_coordinates_at_this_u = derivative_parametric_coordinates.copy()
        stuffing_parametric_coordinates_at_this_u[:,0] = u
        indices = np.arange(i*num_v_stuff*num_w_stuff, (i+1)*num_v_stuff*num_w_stuff)
        stuffing_parametric_coordinates[indices] = stuffing_parametric_coordinates_at_this_u

    # Evaluate displacements B-spline instead of using raw data for fitting (easier to flip and smoother)
    parametric_grid = np.meshgrid(np.linspace(0., 1., num_u_stuff), np.linspace(0., 1., num_v_stuff), np.linspace(0., 1., num_w_stuff))
    parametric_grid = np.stack((parametric_grid[0].flatten(), parametric_grid[1].flatten(), parametric_grid[2].flatten()), axis=1)
    displacements_grid = displacements_b_spline_at_time_t.evaluate(parametric_grid)
    # - Scale u coordinates to module coordinates
    parametric_grid[:,0] = parametric_grid[:,0]*(module_1_max_u - module_1_min_u) + module_1_min_u

    # Fit displacements B-spline to structural displcaements + stuffing points
    combined_parametric_coordinates = np.concatenate((parametric_grid, stuffing_parametric_coordinates))
    combined_values = m3l.vstack((displacements_grid, stuffing_points))

    regularization_parameter = 1.e-3
    combined_evaluation_matrix = fishy.compute_evaluation_map(combined_parametric_coordinates, expand_map_for_physical=True)    # This takes time
    fitting_matrix = (combined_evaluation_matrix.T).dot(combined_evaluation_matrix) \
        + regularization_parameter*sps.identity(combined_evaluation_matrix.shape[1])
    fitting_rhs = m3l.matvec(combined_evaluation_matrix.T.tocsc(), combined_values)
    combined_structural_displacements_coefficients = m3l.solve(fitting_matrix, fitting_rhs)   # This takes time

    structural_displacements_b_spline = \
        fishy.space.create_function(name='structural_displacements_b_spline', coefficients=combined_structural_displacements_coefficients)


    # structural_displacements_magnitude_b_spline = BSpline(
    #     name='structural_displacements_magnitude_b_spline',
    #     space=fishy.space,
    #     coefficients=m3l.norm(structural_displacements_b_spline.coefficients.reshape((-1,3)), axes=(1,)),
    #     num_physical_dimensions=1,
    # )
    # fishy.plot(opacity=0.8, color=structural_displacements_magnitude_b_spline)

    deformed_fishy = fishy.copy()
    deformed_fishy.coefficients = deformed_fishy.coefficients + structural_displacements_b_spline.coefficients
    # deformed_fishy.plot(color=structural_displacements_magnitude_b_spline)

    deformed_fishy_plot = deformed_fishy.plot(opacity=0.8, show=True)
    plotter = vedo.Plotter(offscreen=True)
    plotter.show([fishy_plot, deformed_fishy_plot], axes=1, viewup='y')
    video.add_frame()
    # exit()
video.close()
exit()
# Stuffing below



# Use derivative to stuff data with rigid body displacement
num_u_stuff = 50
num_v_stuff = 25
num_w_stuff = 25

# edge_u_coordinate = 0.66
# edge_u_coordinate = structural_module_back_parametric[0][0]
edge_u_coordinate = module_1_min_u
w_coordinates, v_coordinates = np.meshgrid(np.linspace(0., 1., num_v_stuff), np.linspace(0., 1., num_w_stuff))
u_coordinates = np.ones_like(v_coordinates)*0.
derivative_parametric_coordinates = np.stack((u_coordinates.flatten(), v_coordinates.flatten(), w_coordinates.flatten()), axis=1)
derivative_values_at_edge = opposite_structural_displacements_b_spline.evaluate(derivative_parametric_coordinates, parametric_derivative_order=(1,0,0))

displacement_values_at_edge = opposite_structural_displacements_b_spline.evaluate(derivative_parametric_coordinates)


u_stuff = np.linspace(0., edge_u_coordinate, num_u_stuff, endpoint=False)
stuffing_points = m3l.Variable(shape=(num_u_stuff*num_v_stuff*num_w_stuff*3,), value=np.zeros(num_u_stuff*num_v_stuff*num_w_stuff*3))
stuffing_parametric_coordinates = np.zeros((num_u_stuff*num_v_stuff*num_w_stuff, 3))

# This is slow because of lots of small-ish indexing (It takes a little bit of time but not significant (~0.5 seconds I think))
for i, u in enumerate(u_stuff):
    delta_u = (u - edge_u_coordinate)/(module_1_max_u - module_1_min_u)
    flattened_indices = np.arange(i*num_v_stuff*num_w_stuff*3, (i+1)*num_v_stuff*num_w_stuff*3)
    stuffing_points[flattened_indices] = displacement_values_at_edge + (derivative_values_at_edge*delta_u).reshape((-1,))
    stuffing_parametric_coordinates_at_this_u = derivative_parametric_coordinates.copy()
    stuffing_parametric_coordinates_at_this_u[:,0] = u
    indices = np.arange(i*num_v_stuff*num_w_stuff, (i+1)*num_v_stuff*num_w_stuff)
    stuffing_parametric_coordinates[indices] = stuffing_parametric_coordinates_at_this_u

# Evaluate displacements B-spline instead of using raw data for fitting (easier to flip and smoother)
parametric_grid = np.meshgrid(np.linspace(0., 1., num_u_stuff), np.linspace(0., 1., num_v_stuff), np.linspace(0., 1., num_w_stuff))
parametric_grid = np.stack((parametric_grid[0].flatten(), parametric_grid[1].flatten(), parametric_grid[2].flatten()), axis=1)
opposite_displacements_grid = opposite_structural_displacements_b_spline.evaluate(parametric_grid)
# - Scale u coordinates to module coordinates
parametric_grid[:,0] = parametric_grid[:,0]*(module_1_max_u - module_1_min_u) + module_1_min_u

# Fit displacements B-spline to structural displcaements + stuffing points
# combined_parametric_coordinates = np.concatenate((structural_mesh_parametric, stuffing_parametric_coordinates))
# combined_values = m3l.vstack((structural_displacements_flattened, stuffing_points))
combined_parametric_coordinates = np.concatenate((parametric_grid, stuffing_parametric_coordinates))
combined_values = m3l.vstack((opposite_displacements_grid, stuffing_points))

regularization_parameter = 1.e-3
combined_evaluation_matrix = fishy.compute_evaluation_map(combined_parametric_coordinates, expand_map_for_physical=True)    # This takes time
fitting_matrix = (combined_evaluation_matrix.T).dot(combined_evaluation_matrix) \
    + regularization_parameter*sps.identity(combined_evaluation_matrix.shape[1])
fitting_rhs = m3l.matvec(combined_evaluation_matrix.T.tocsc(), combined_values)
combined_structural_displacements_coefficients = m3l.solve(fitting_matrix, fitting_rhs)   # This takes time

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
exit()
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
    delta_u = (u - edge_u_coordinate)/(module_1_max_u - module_1_min_u)
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
# deformed_fishy.plot(color=structural_displacements_magnitude_b_spline)
# deformed_fishy.plot()

opposite_deformed_fishy = deformed_fishy.copy()
coefficients_shape = opposite_deformed_fishy.space.parametric_coefficients_shape + (3,)
indices = np.arange(np.prod(coefficients_shape)).reshape(coefficients_shape)
# want to flip the z component of the coefficients and invert the ordering of the x component along the 3rd axis of coefficients
z_indicies = indices[:,:,:,2].reshape((-1,))
# opposite_deformed_fishy.coefficients[z_indicies] = -deformed_fishy.coefficients[z_indicies]   # Don't want to do this because it flips ordering.
# NOTE: Flipping the ordering makes the geometry flat when we interpolate.
for i in range(coefficients_shape[2]):
    # flip x component
    # set_indices = [:,:,i,0]
    # get_indices = [:,:,coefficients_shape[2]-i-1,0]
    set_indices = indices[:,:,i,0].reshape((-1,))
    get_indices = indices[:,:,coefficients_shape[2]-i-1,0].reshape((-1,))
    # set_indices = np.arange(i*coefficients_shape[0]*coefficients_shape[1], (i+1)*coefficients_shape[0]*coefficients_shape[1])
    # get_indices = np.arange((coefficients_shape[2]-i-1)*coefficients_shape[0]*coefficients_shape[1], (coefficients_shape[2]-i)*coefficients_shape[0]*coefficients_shape[1])
    opposite_deformed_fishy.coefficients[set_indices] = deformed_fishy.coefficients[get_indices]

    # flip z component
    set_indices = indices[:,:,i,2].reshape((-1,))
    get_indices = indices[:,:,coefficients_shape[2]-i-1,2].reshape((-1,))
    opposite_deformed_fishy.coefficients[set_indices] = -deformed_fishy.coefficients[get_indices]
# opposite_deformed_fishy.plot(color=structural_displacements_magnitude_b_spline)
# exit()

# endregion -Map Displacements To Geometry

# region -Perform interpolation to get dynamic profile
# NOTE: interpolation is a linear combination between displaced geometry and the displacement flipped around z. Interpolation is exponential in time.
# NOTE NOTE: Going to start with interpolation between just the deformed and opposite deformed geometry, then include the undeformed state
# - Then perhaps include an arbitrary number of intermediate states to make the profile look more and more accurate.

time_constant = 3
actuation_frequency = 1.
actuation_period = 1/actuation_frequency
# num_vast_time_steps = 41
num_vast_time_steps = 81
# time_segment_1 = np.linspace(0, actuation_period/2, num_vast_time_steps//2, endpoint=False)
# time_segment_2 = np.linspace(actuation_period/2, actuation_period, num_vast_time_steps//2 + 1, endpoint=True)
# time = np.concatenate((time_segment_1, time_segment_2))
time = np.linspace(0, actuation_period*2, num_vast_time_steps)
num_strokes = int(actuation_frequency*time[-1])*2+1

# time_offset_to_center_interpolation = 0.075
# weights_deformed_segment_1 = np.exp(-time_constant*(time_segment_1 + time_offset_to_center_interpolation))
# weights_opposite_deformed_segment_1 = 1 - weights_deformed_segment_1
# weights_deformed_segment_2 = 1 - np.exp(-time_constant*(time_segment_2 - actuation_period/2 + time_offset_to_center_interpolation)) \
#                             + np.exp(-time_constant*(time_segment_2[0] + time_offset_to_center_interpolation)) \
#                             - np.exp(-time_constant*(time_segment_2[-1] + time_offset_to_center_interpolation))
# weights_opposite_deformed_segment_2 = 1 - weights_deformed_segment_2
# weights_deformed_segment_2 = weights_opposite_deformed_segment_1  # Doesn't work because wrong shape
# weights_opposite_deformed_segment_2 = weights_deformed_segment_1  # Doesn't work because wrong shape


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

left_chamber_inputs = []
right_chamber_inputs = []
for stroke_index in range(num_strokes):
    if stroke_index % 2 == 0:
        left_chamber_inputs.append(1.)
        right_chamber_inputs.append(0.)
    else:
        left_chamber_inputs.append(0.)
        right_chamber_inputs.append(1.)

t_pressure_inputs = np.linspace(0, time[-1], int(num_strokes))
t_pressure_inputs[1:] = t_pressure_inputs[1:] - (actuation_period/2)/2
deformed_fishy_weights = compute_chamber_pressure_function(t_pressure_inputs, left_chamber_inputs, time_constant, 0.5, time)
opposite_deformed_fishy_weights = 1 - deformed_fishy_weights


# actuating_fishy_coefficients_first_half_period = m3l.linear_combination(deformed_fishy.coefficients, opposite_deformed_fishy.coefficients, 
#                                                    start_weights=weights_deformed_segment_1, stop_weights=weights_opposite_deformed_segment_1)
# actuating_fishy_coefficients_second_half_period = m3l.linear_combination(opposite_deformed_fishy.coefficients, deformed_fishy.coefficients,
#                                                     start_weights=weights_opposite_deformed_segment_2, stop_weights=weights_deformed_segment_2)

actuating_fishy_coefficients = m3l.linear_combination(deformed_fishy.coefficients, opposite_deformed_fishy.coefficients, 
                                                   start_weights=deformed_fishy_weights, stop_weights=opposite_deformed_fishy_weights)

actuating_fishy_coefficients = \
    actuating_fishy_coefficients.reshape((-1,))
# actuating_fishy_coefficients_second_half_period = \
#     actuating_fishy_coefficients_second_half_period.reshape((-1,))

# first_segment_indices = np.arange(time_segment_1.shape[0]*fishy.coefficients.shape[0]).reshape((time_segment_1.shape[0], fishy.coefficients.shape[0]))
# second_segment_indices = np.arange(time_segment_2.shape[0]*fishy.coefficients.shape[0]).reshape((time_segment_2.shape[0], fishy.coefficients.shape[0]))
indices = np.arange(time.shape[0]*fishy.coefficients.shape[0]).reshape((time.shape[0], fishy.coefficients.shape[0]))
fishy_plot = fishy.plot(show=False, opacity=0.3)
video = vedo.Video('examples/advanced_examples/robotic_fish/temp/fishy_wiggle.mp4', fps=21, backend='cv')
for i in range(time.shape[0]):

    # if i < num_vast_time_steps//2:
    #     get_indices = first_segment_indices[i,:].reshape((-1,))
    #     actuating_fishy_coefficients = actuating_fishy_coefficients_first_half_period[get_indices]
    # else:
    #     get_indices = second_segment_indices[i - time_segment_1.shape[0],:].reshape((-1,))
    #     actuating_fishy_coefficients = actuating_fishy_coefficients_second_half_period[get_indices]

    get_indices = indices[i,:].reshape((-1,))
    deformed_fishy.coefficients = actuating_fishy_coefficients[get_indices]
    deformed_fishy_plot = deformed_fishy.plot(opacity=0.8, show=False)
    plotter = vedo.Plotter(offscreen=True)
    plotter.show([fishy_plot, deformed_fishy_plot], axes=1, viewup='y')
    video.add_frame()
    # exit()
video.close()

exit()
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
