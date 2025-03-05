import csdl_alpha as csdl
import lsdo_geo
import lsdo_function_spaces as lfs
import lsdo_soft_csm
import pickle
import meshio
import numpy as np

from modopt import CSDLAlphaProblem
from modopt import PySLSQP
import vedo
import pandas
import matplotlib.pyplot as plt




'''
Objective: Maximize 'actuator' angle or average curvature
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

# region Read Experimental Data From Baseline Design
df = pandas.read_excel('examples/advanced_examples/robotic_fish/experimental_data/pressure_test.xlsx')
df2 = pandas.read_excel('examples/advanced_examples/robotic_fish/experimental_data/angle_vs_p.xlsx')

#get a data frame with selected columns
column_names = ['pressure',
                'act_1_left_1', 'act_1_left_2', 'act_1_left_3',
                'act_2_left_1', 'act_2_left_2', 'act_2_left_3',
                'act_3_left_1', 'act_3_left_2', 'act_3_left_3',
                'act_4_left_1', 'act_4_left_2', 'act_4_left_3',
                'act_1_right_1', 'act_1_right_2', 'act_1_right_3',
                'act_2_right_1', 'act_2_right_2', 'act_2_right_3',
                'act_3_right_1', 'act_3_right_2', 'act_3_right_3',
                'act_4_right_1', 'act_4_right_2', 'act_4_right_3']
bending_angle_column_names = column_names[1:]
pressure_column_name = ['pressure']
pressures = df[pressure_column_name].values
bending_angles = df[bending_angle_column_names].values
# Filter out columns of data where the final angle is greater than 80 degrees
bending_angles = bending_angles[:,np.where(bending_angles[-1] < 70)[0]]
bending_angles_data = bending_angles

# bending_angles_standard_deviation = df[bending_angle_column_names].std(axis=1)
bending_angles_standard_deviation = np.std(bending_angles, axis=1)
data = df[column_names].values
average_bending_angles = np.mean(bending_angles, axis=1)
# endregion Read Experimental Data From Baseline Design

# # region Read Data from Optimized Design Experiment
# file_names = ['left_chamber_output_1.csv', 'left_chamber_output_2.csv', 'left_chamber_output_3.csv',
#               'right_chamber_output_1.csv', 'right_chamber_output_2.csv', 'right_chamber_output_3.csv']
# file_path = 'examples/advanced_examples/robotic_fish/experimental_data/optimized_actuator_angle_vs_pressure_data/'
# pressures = []
# bending_angles = []
# for file_name in file_names:
#     pressure = 0
#     df = pandas.read_csv(file_path + file_name)
#     bending_angles_data_set = df.values
#     if 'right' in file_name:
#         bending_angles_data_set = -bending_angles_data_set
#     # There's a random data point that says 0 angle so let's remove that
#     bending_angles_data_set = bending_angles_data_set[bending_angles_data_set != 0.]
    
#     pressures_data_set = []
#     for i, angle in enumerate(bending_angles_data_set):
#         if i == 0:
#             pressures_data_set.append(pressure)
#         else:
#             if angle - bending_angles_data_set[i-1] > 1.:
#                 pressure += 0.5
#             pressures_data_set.append(pressure)

#     averaged_bending_angles = []
#     for pressure in np.unique(pressures_data_set):
#         indices = np.where(pressures_data_set == pressure)[0]
#         averaged_bending_angles.append(np.mean(bending_angles_data_set[indices]))
#     averaged_bending_angles = np.array(averaged_bending_angles)
#     averaged_bending_angles -= averaged_bending_angles[0]
#     pressures.append(np.unique(pressures_data_set))
#     bending_angles.append(averaged_bending_angles)

# pressures = np.vstack(pressures).T
# pressures = pressures[:,0].reshape((-1,1))
# bending_angles = np.vstack(bending_angles).T
# bending_angles_data = bending_angles

# # pressures = pressures[0,:]
# average_bending_angles = np.mean(bending_angles, axis=1)
# bending_angles_standard_deviation = np.std(bending_angles, axis=1)

# # endregion Read Data from Optimized Design Experiment

plt.plot(pressures[:,0]*6894.76/1e3, bending_angles[:,0], 'gx', label='Cervera-Torralba et al. (Experimental Data)')
plt.plot(pressures*6894.76/1e3, bending_angles[:,1:], 'gx')
plt.plot(pressures*6894.76/1e3, average_bending_angles, '-', linewidth=7., label='Cervera-Torralba et al. (Mean of Experimental Data)')
plt.fill_between(pressures[:,0]*6894.76/1e3,
                 average_bending_angles-bending_angles_standard_deviation, average_bending_angles+bending_angles_standard_deviation,
                 alpha=0.3, label='Cervera-Torralba et al. (1 Standard Deviation of Experimental Data)')
plt.title('Bending Angle vs Pressure for Optimized Design')
plt.xlabel('Pressure (kPa)')
plt.ylabel('Bending Angle (degrees)')
# plt.legend()
# plt.show()
# exit()

recorder = csdl.Recorder(inline=True)
recorder.start()

# region Import and Setup
def import_geometry() -> lfs.Function:
    # with open("examples/advanced_examples/robotic_fish/pickle_files/fishy_volume_geometry_very_fine_quadratic.pickle", 'rb') as handle:
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
# mesh_name = "module_v1_refined_2"
mesh_name = "module_v1_refinement_study_20mm"
# mesh_name = "module_v1_refinement_study_2point5mm"
structural_mesh = meshio.read(mesh_file_path + mesh_name + ".msh")
structural_mesh_nodes = structural_mesh.points/1000 + np.array([0.04, 0, 0.])   # Shift the mesh to the right to make it the middle module
structural_elements = structural_mesh.cells_dict['tetra']

# vedo_mesh = vedo.Mesh([structural_mesh_nodes, structural_elements]).wireframe()
# plotter = vedo.Plotter()
# plotter.show(vedo_mesh, axes=1, viewup='y')

# Reorder the nodes to match the FEniCS mesh
try:
    fenics_mesh_indices = pickle.load(open(mesh_file_path + mesh_name + "_fenics_mesh_indices.pickle", "rb"))
    left_chamber_indices = pickle.load(open(mesh_file_path + mesh_name + "_left_chamber_fenics_mesh_indices.pickle", "rb"))
    right_chamber_indices = pickle.load(open(mesh_file_path + mesh_name + "_right_chamber_fenics_mesh_indices.pickle", "rb"))
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


    # Get left and right chamber indices
    from femo.fea.fea_dolfinx import XDMFFile, MPI, FunctionSpace, locate_dofs_topological
    file_path = 'examples/advanced_examples/robotic_fish/meshes/'

    with XDMFFile(MPI.COMM_WORLD, file_path + mesh_name + ".xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    with XDMFFile(MPI.COMM_WORLD, file_path + mesh_name + "_left_chamber_inner_surfaces.xdmf", "r") as xdmf:
        left_chamber_facet_tags = xdmf.read_meshtags(mesh, name="Grid")

    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    with XDMFFile(MPI.COMM_WORLD, file_path + mesh_name + "_right_chamber_inner_surfaces.xdmf", "r") as xdmf:
        right_chamber_facet_tags = xdmf.read_meshtags(mesh, name="Grid")

    input_function_space = FunctionSpace(mesh, ('CG', 2))   # For some reason, finding the dofs only works if the input function space is CG2
    left_chamber_facets = left_chamber_facet_tags.find(508)     # module_v1_refined_2: 508 is what GMSH GUI assigned it (see in tools --> visibility)
    left_chamber_facet_dofs = list(locate_dofs_topological(input_function_space, mesh.topology.dim - 1, left_chamber_facets))

    right_chamber_facets = right_chamber_facet_tags.find(509)     # module_v1_refined_2: 509 is what GMSH GUI assigned it (see in tools --> visibility)
    right_chamber_facet_dofs = list(locate_dofs_topological(input_function_space, mesh.topology.dim - 1, right_chamber_facets))

    file_name = mesh_file_path + mesh_name + "_left_chamber_fenics_mesh_indices.pickle"
    with open(file_name, 'wb+') as handle:
        pickle.dump(left_chamber_facet_dofs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    left_chamber_indices = pickle.load(open(mesh_file_path + mesh_name + "_left_chamber_fenics_mesh_indices.pickle", "rb"))
    file_name = mesh_file_path + mesh_name + "_right_chamber_fenics_mesh_indices.pickle"
    with open(file_name, 'wb+') as handle:
        pickle.dump(right_chamber_facet_dofs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    right_chamber_indices = pickle.load(open(mesh_file_path + mesh_name + "_right_chamber_fenics_mesh_indices.pickle", "rb"))

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

structural_mesh_parametric = fishy.project(points=structural_mesh_nodes, grid_search_density_parameter=1., plot=False, force_reproject=False,
                                           projection_tolerance=1.e-3)
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
# num_ffd_sections = 2
# ffd_block = lsdo_geo.construct_ffd_block_around_entities(entities=fishy, num_coefficients=(num_ffd_sections,2,2), degree=(1,1,1))
ffd_min_x = np.min(fishy.coefficients.value[:,:,:,0])
ffd_max_x = np.max(fishy.coefficients.value[:,:,:,0])
ffd_min_y = np.min(fishy.coefficients.value[:,:,:,1])
ffd_max_y = np.max(fishy.coefficients.value[:,:,:,1])
ffd_min_z = np.min(fishy.coefficients.value[:,:,:,2])
ffd_max_z = np.max(fishy.coefficients.value[:,:,:,2])
ffd_x_values = np.array([ffd_min_x, ffd_max_x])
ffd_y_values = np.array([ffd_min_y, ffd_max_y])
# ffd_z_values = np.array([ffd_min_z, -0.0015, 0.0015, ffd_max_z])
ffd_z_values = np.array([ffd_min_z, -0.003, 0.003, ffd_max_z])

x,y,z = np.meshgrid(ffd_x_values, ffd_y_values, ffd_z_values, indexing='ij')
ffd_corners = np.array([x.flatten(), y.flatten(), z.flatten()]).T.reshape(
    (len(ffd_x_values), len(ffd_y_values), len(ffd_z_values), 3))
# ffd_block = lsdo_geo.construct_ffd_block_from_corners(entities=fishy, corners=ffd_corners, num_coefficients=(2,3,2))
ffd_block = lsdo_geo.construct_ffd_block_from_corners(entities=fishy, corners=ffd_corners, num_coefficients=(2,7,2))
# ffd_block.plot()

ffd_sectional_parameterization = lsdo_geo.VolumeSectionalParameterization(
    parameterized_points=ffd_block.coefficients,
    principal_parametric_dimension=2,
    parameterized_points_shape=ffd_block.coefficients.shape,
    name='ffd_sectional_parameterization',
)
num_ffd_sections = len(ffd_z_values)
# plotting_elements = fishy.plot(show=False, opacity=0.3, color='#FFCD00')

length_stretch = csdl.Variable(value=0., name='length_stretch')
width_stretch = csdl.Variable(value=0., name='width_stretch')
height_stretch = csdl.Variable(value=0., name='height_stretch')

width_shape_variables = csdl.Variable(shape=(ffd_block.coefficients.shape[1]//2 + 1,), value=0., name='width_shape_deltas')
# width_shape_variables.set_value(np.array([-6.928424636601911678e-03, -3.742655120055620593e-02, -6.632995188892187866e-02, 1.106849277260799624e-01])/100)

width_shape_deltas = csdl.Variable(shape=(ffd_block.coefficients.shape[1],), value=0.)
width_shape_deltas = width_shape_deltas.set(csdl.slice[0:width_shape_variables.size], width_shape_variables)
width_shape_deltas = width_shape_deltas.set(csdl.slice[width_shape_variables.size:], width_shape_variables[-2::-1])
# deltas_sum = csdl.sum(width_shape_deltas)

# endregion -Create Parameterization Objects

# region -Define Parameterization For Solver
length_stretches = csdl.expand(length_stretch, (num_ffd_sections,))
width_stretches = csdl.expand(width_stretch, (num_ffd_sections,))
width_stretches = width_stretches.set(csdl.slice[1:3], 0.)
width_stretches = width_stretches.set(csdl.slice[0], -width_stretch)
height_stretches = csdl.expand(height_stretch, (num_ffd_sections,))

ffd_sectional_parameterization_inputs = lsdo_geo.VolumeSectionalParameterizationInputs()
ffd_sectional_parameterization_inputs.add_sectional_stretch(axis=0, stretch=length_stretches)
ffd_sectional_parameterization_inputs.add_sectional_translation(axis=2, translation=width_stretches)
ffd_sectional_parameterization_inputs.add_sectional_stretch(axis=1, stretch=height_stretches)
ffd_block_coefficients = ffd_sectional_parameterization.evaluate(ffd_sectional_parameterization_inputs, plot=False)

width_shape_deltas_expanded = csdl.expand(width_shape_deltas, ffd_block_coefficients.shape[:2], 'i->ji')
ffd_block_coefficients = ffd_block_coefficients.set(csdl.slice[:,:,-1,2], ffd_block_coefficients[:,:,-1,2] + width_shape_deltas_expanded)
ffd_block_coefficients = ffd_block_coefficients.set(csdl.slice[:,:,0,2], ffd_block_coefficients[:,:,0,2] - width_shape_deltas_expanded)

fishy_coefficients = ffd_block.evaluate(coefficients=ffd_block_coefficients, plot=False)
# fishy.coefficients = fishy_coefficients.reshape(fishy.coefficients.shape)

fishy.coefficients = fishy_coefficients

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
# width = csdl.Variable(value=computed_fishy_width.value, name='width')
height = csdl.Variable(value=computed_fishy_height.value, name='height')

# length = csdl.Variable(value=1.1, name='length')
# width = csdl.Variable(value=0.02, name='width')
# height = csdl.Variable(value=0.07, name='height')

# endregion Geometric Design Variables

geometry_parameterization_solver = lsdo_geo.ParameterizationSolver()

geometry_parameterization_solver.add_parameter(length_stretch)
# geometry_parameterization_solver.add_parameter(width_stretch)
geometry_parameterization_solver.add_parameter(height_stretch)

geometric_parameterization_variables = lsdo_geo.GeometricVariables()
geometric_parameterization_variables.add_variable(computed_fishy_length, length)
# geometric_parameterization_variables.add_variable(computed_fishy_width, width)
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
pump_pressure = csdl.Variable(value=0.e4, name='pump_pressure')
E_dragon = csdl.Variable(value=3.e5, name='E_dragon')
nu_dragon = csdl.Variable(value=0.45, name='nu_dragon')
E_fr4 = csdl.Variable(value=0.9e8, name='E_fr4')
nu_fr4 = csdl.Variable(value=0.12, name='nu_fr4')
# pressure_input_coefficients = csdl.Variable(value=np.zeros((327437,)), name='pressure_input_coefficients')  # 20mm mesh
# pressure_input_coefficients = csdl.Variable(value=np.zeros((333115,)), name='pressure_input_coefficients')  # 10mm mesh
# pressure_input_coefficients = csdl.Variable(value=np.zeros((497409,)), name='pressure_input_coefficients')  # 2.5mm mesh
# pressure_input_coefficients = pressure_input_coefficients.set(csdl.slice[left_chamber_indices], -pump_pressure)
# # pressure_input_coefficients = pressure_input_coefficients.set(csdl.slice[right_chamber_indices], pump_pressure)

# structural_displacements_flattened, _ = lsdo_soft_csm.robotic_fish_static_structural_model(structural_mesh_displacements, pressure_input_coefficients)
# structural_displacements = structural_displacements_flattened.reshape((structural_displacements_flattened.size//3,3))

pump_pressures = pressures*6894.76
pump_pressures_csdl = csdl.Variable(value=pump_pressures, name='pump_pressures')
bending_angles = csdl.Variable(shape=(len(pump_pressures),), value=0., name='bending_angles')
# for j, pressure in enumerate(pump_pressures):
for j in csdl.frange(pump_pressures_csdl.size):
    pump_pressure = pump_pressures_csdl[j]

    structural_displacements, _, new_mesh_cooordinates = lsdo_soft_csm.robotic_fish_static_structural_model(structural_mesh_displacements, pump_pressure,
                                                                                                            E_dragon, nu_dragon, E_fr4, nu_fr4)
    # structural_displacements, _, new_mesh_cooordinates = lsdo_soft_csm.robotic_fish_static_structural_model(structural_mesh_displacements, pressure_input_coefficients)
    if new_mesh_cooordinates.shape != structural_mesh.shape:
        structural_mesh = new_mesh_cooordinates
        structural_mesh_parametric = fishy.project(points=new_mesh_cooordinates, grid_search_density_parameter=1., plot=False, projection_tolerance=1.e-3)



    # displaced_mesh = structural_mesh + structural_displacements
    # initial_displaced_mesh = displaced_mesh.value

    # Plot structural solver result
    # fishy_plot = fishy.plot(show=False, opacity=0.3)
    # vedo_mesh = vedo.Mesh([displaced_mesh.value, structural_elements]).wireframe()
    # plotter = vedo.Plotter()
    # plotter.show([fishy_plot, vedo_mesh], axes=1, viewup='y')

    # endregion Structural Solver

    # region Fit B-Spline to Displacements and Construct Deformed Geometry
    displacement_space = lfs.BSplineSpace(
        num_parametric_dimensions=3,
        degree=(2,2,2),
        coefficients_shape=(7,3,3))

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

    coefficients_parametric = fishy.project(points=fishy.coefficients, grid_search_density_parameter=1., projection_tolerance=2.e-3, plot=False, force_reproject=False)
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

    bending_angles = bending_angles.set(csdl.slice[j], angle)


bending_angles = bending_angles - bending_angles[0]
bending_angles = bending_angles*180/np.pi*2   # convert to degrees and multiply by 2 for symmetry

differences = bending_angles_data - csdl.expand(bending_angles, bending_angles_data.shape, 'i->ij')
rmse = csdl.sqrt(csdl.average(differences**2))


# parametric_coordinate_at_module_edge = np.array([[module_2_min_u, 0.5, 0.5]])
# derivative_at_module_edge_old = deformed_fishy.evaluate(parametric_coordinate_at_module_edge, parametric_derivative_orders=(1,0,0))
# derivative_at_module_edge_old = derivative_at_module_edge_old/csdl.norm(derivative_at_module_edge_old)    # normalize
# old_angle = csdl.arccos(csdl.vdot(initial_angle, derivative_at_module_edge_old))
# endregion Objective Model

# region Optimization
E_dragon.set_as_design_variable(lower=1.e4, upper=1.e7, scaler=1.e-5)
nu_dragon.set_as_design_variable(lower=0.3, upper=0.499, scaler=1.e1)
E_fr4.set_as_design_variable(lower=1.e6, upper=24.e9, scaler=1.e-8)
# nu_fr4.set_as_design_variable(lower=0.11, upper=0.13, scaler=1.e1)

rmse.set_as_objective(scaler=1.e0)

# width_shape_variables.set_value(np.array([5.731138044206841364e-3, -6.657115938444321479e-3, -7.965134789907922119e-3, -8.002576986651789070e-3]))
# height.set_value(7.931587973291041926e-2)

# # Optimized parameters for semi-optimized design
# E_dragon.set_value(3.661597153363377544e+5)
# nu_dragon.set_value(4.183650217403863003e-1)
# E_fr4.set_value(4.947969608814554165e+7)

# Manual guess for baseline design
E_dragon.set_value(2.361597153363377544e+5)
nu_dragon.set_value(4.183650217403863003e-1)
E_fr4.set_value(4.947969608814554165e+7)

# sim = csdl.experimental.PySimulator(recorder=recorder)
additional_inputs = [width_shape_variables, height, E_dragon, nu_dragon, E_fr4, nu_fr4]
additional_outputs = [objective, angle, bending_angles]
sim = csdl.experimental.JaxSimulator(
    recorder = recorder,
    additional_inputs=additional_inputs,
    additional_outputs=additional_outputs,
    gpu=False
)

# sim.run()

# print('bending_angles: ', bending_angles.value)
# print('bending_angles_expanded: ', csdl.expand(bending_angles, bending_angles_data.shape, 'i->ij').value)
# print('bending_angles_data: ', bending_angles_data)
# print('differences: ', differences.value)
# print('RMSE: ', rmse.value)

# import matplotlib.pyplot as plt
# plt.plot(np.array(pump_pressures)/1e3, bending_angles*180/np.pi*2, '-o', label='Simulated', color='magenta')
# plt.legend()
# plt.show()
# exit()


optimization_problem = CSDLAlphaProblem(problem_name='fishy_optimization', simulator=sim)
optimizer = PySLSQP(optimization_problem, solver_options={'maxiter':100, 'acc':1.e-8}, readable_outputs=['x'])
# recorder.execute()

optimizer.solve()
optimizer.print_results()

plt.plot(np.array(pump_pressures)/1e3, bending_angles.value, '-o', label='Simulated', color='magenta')
plt.legend()
plt.show()
exit()


bending_angles = []
# pump_pressures_psi = np.linspace(0., 6., 18)
# pump_pressures_psi = np.linspace(0., 6., 7)
# pump_pressures_psi = np.hstack((pump_pressures_psi, np.array([6.5, 7.])))
# pump_pressures = pump_pressures_psi*6894.76
pump_pressures = pressures*6894.76
pump_pressures_for_plot = []
for i, pressure in enumerate(pump_pressures):
    sim[pump_pressure] = pressure
    try:
        sim.run()
    except:
        break
    print('Pressure in psi: ', pressure/6894.76)
    print('Angle: ', angle.value*180/np.pi*2)
    # pump_pressures_for_plot.append(pressure/6894.76)
    pump_pressures_for_plot.append(pressure)
    bending_angles.append(angle.value)

bending_angles = np.array(bending_angles)
bending_angles = bending_angles - bending_angles[0]


import matplotlib.pyplot as plt
plt.plot(np.array(pump_pressures_for_plot)/1e3, bending_angles*180/np.pi*2, '-o', label='Simulated', color='magenta')
plt.legend()
plt.show()

# compute RMSE against experimental data for bending angles
differences = bending_angles_data - bending_angles.reshape((-1,1))
rmse = np.sqrt(np.mean(differences**2))
print('RMSE: ', rmse)

exit()



# exit()
optimization_problem = CSDLAlphaProblem(problem_name='fishy_optimization', simulator=sim)
# optimizer = SLSQP(optimization_problem, maxiter=100, ftol=1.e-7)
# optimizer = PySLSQP(optimization_problem, solver_options={'maxiter':100, 'acc':1.e-9}, readable_outputs=['x'])
optimizer = PySLSQP(optimization_problem, solver_options={'maxiter':100, 'acc':1.e-8}, readable_outputs=['x'])

initial_objective_value = objective.value
initial_length = length.value
# initial_width = width.value
initial_height = height.value
# initial_surface_area = surface_area.value
initial_angle = angle.value


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
# print('Initial Width', initial_width)
print('Initial Height', initial_height)
# print('Initial Surface Area', initial_surface_area)
print('Initial Angle: ', initial_angle)

print('Optimized Objective: ', objective.value)
print('Optimized Length', length.value)
# print('Optimized Width', width.value)
print('Optimized Height', height.value)
# print('Optimized Surface Area', surface_area.value)
print('Optimized Angle: ', angle.value)

print('Percent Change in Objective', (objective.value - initial_objective_value)/initial_objective_value*100)
print("Percent Change in Length: ", (length.value - initial_length)/initial_length*100)
# print("Percent Change in Width: ", (width.value - initial_width)/initial_width*100)
print("Percent Change in Height: ", (height.value - initial_height)/initial_height*100)

# from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
# verify_derivatives_inline([csdl.norm(fishy.coefficients)], [length, width, height], 1.e-6, raise_on_error=False)
# verify_derivatives_inline([csdl.norm(structural_displacements_flattened)], [length, width, height], 1.e-6, raise_on_error=False)
# verify_derivatives_inline([csdl.norm(structural_displacements_b_spline.coefficients)], [length, width, height], 1.e-6, raise_on_error=False)
# verify_derivatives_inline([csdl.norm(deformations_in_module)], [length, width, height], 1.e-6, raise_on_error=False)
# verify_derivatives_inline([csdl.norm(deformed_fishy.coefficients)], [length, width, height], 1.e-6, raise_on_error=False)
# optimizer.check_first_derivatives(optimization_problem.x0)

## Plot structural solver result
# fishy_plot = fishy.plot(show=False, opacity=0.3)
# vedo_mesh_initial = vedo.Mesh([initial_displaced_mesh, structural_elements]).wireframe().color('yellow').opacity(0.4)
# vedo_mesh = vedo.Mesh([displaced_mesh.value, structural_elements]).wireframe().color('green').opacity(0.8)
# plotter = vedo.Plotter()
# plotter.show([fishy_plot, vedo_mesh_initial, vedo_mesh], axes=1, viewup='y')

# endregion Optimization