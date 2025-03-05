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
bending_angles_data_baseline = bending_angles
pressures_baseline = pressures

# bending_angles_standard_deviation = df[bending_angle_column_names].std(axis=1)
bending_angles_standard_deviation_baseline = np.std(bending_angles, axis=1)
data = df[column_names].values
average_bending_angles_baseline = np.mean(bending_angles, axis=1)


plt.plot(pressures_baseline[:,0]*6894.76/1e3, bending_angles_data_baseline[:,0], 'gx', label='Cervera-Torralba et al. (Baseline Design Experimental Data)')
plt.plot(pressures_baseline*6894.76/1e3, bending_angles_data_baseline[:,1:], 'gx')
plt.plot(pressures_baseline*6894.76/1e3, average_bending_angles_baseline, '-g', linewidth=7., label='Cervera-Torralba et al. (Baseline Design Mean of Experimental Data)')
plt.fill_between(pressures_baseline[:,0]*6894.76/1e3,
                 average_bending_angles_baseline-bending_angles_standard_deviation_baseline, average_bending_angles_baseline+bending_angles_standard_deviation_baseline,
                 alpha=0.3, color='green', label='Cervera-Torralba et al. (Baseline Design 1 Standard Deviation of Experimental Data)')

# endregion Read Experimental Data From Baseline Design

# region Read Data from Optimized Design Experiment
file_names = ['left_chamber_output_1.csv', 'left_chamber_output_2.csv', 'left_chamber_output_3.csv',
              'right_chamber_output_1.csv', 'right_chamber_output_2.csv', 'right_chamber_output_3.csv']
file_path = 'examples/advanced_examples/robotic_fish/experimental_data/optimized_actuator_angle_vs_pressure_data/'
pressures = []
bending_angles = []
for file_name in file_names:
    pressure = 0
    df = pandas.read_csv(file_path + file_name)
    bending_angles_data_set = df.values
    if 'right' in file_name:
        bending_angles_data_set = -bending_angles_data_set
    # There's a random data point that says 0 angle so let's remove that
    bending_angles_data_set = bending_angles_data_set[bending_angles_data_set != 0.]
    
    pressures_data_set = []
    for i, angle in enumerate(bending_angles_data_set):
        if i == 0:
            pressures_data_set.append(pressure)
        else:
            if angle - bending_angles_data_set[i-1] > 1.:
                pressure += 0.5
            pressures_data_set.append(pressure)

    averaged_bending_angles = []
    for pressure in np.unique(pressures_data_set):
        indices = np.where(pressures_data_set == pressure)[0]
        averaged_bending_angles.append(np.mean(bending_angles_data_set[indices]))
    averaged_bending_angles = np.array(averaged_bending_angles)
    averaged_bending_angles -= averaged_bending_angles[0]
    pressures.append(np.unique(pressures_data_set))
    bending_angles.append(averaged_bending_angles)

pressures = np.vstack(pressures).T
pressures = pressures[:,0].reshape((-1,1))
bending_angles = np.vstack(bending_angles).T
bending_angles_data_optimized = bending_angles
pressures_optimized = pressures

# pressures = pressures[0,:]
average_bending_angles_optimized = np.mean(bending_angles, axis=1)
bending_angles_standard_deviation_optimized = np.std(bending_angles, axis=1)

plt.plot(pressures[:,0]*6894.76/1e3, bending_angles[:,0], 'bx', label='Optimized Design Experimental Data')
plt.plot(pressures*6894.76/1e3, bending_angles[:,1:], 'bx')
plt.plot(pressures*6894.76/1e3, average_bending_angles_optimized, '-b', linewidth=7., label='Optimized Design Mean of Experimental Data')
plt.fill_between(pressures[:,0]*6894.76/1e3,
                 average_bending_angles_optimized-bending_angles_standard_deviation_optimized, average_bending_angles_optimized+bending_angles_standard_deviation_optimized,
                 alpha=0.3, label='Optimized Design 1 Standard Deviation of Experimental Data')
# endregion Read Data from Optimized Design Experiment

plt.title('Bending Angle vs Pressure for Baseline and Optimized Designs')
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
# mesh_name = "module_v1_refinement_study_20mm"
# mesh_name = "module_v1_refinement_study_2point5mm"
mesh_name = "module_v1_very_refined_centerline_2point5mm"
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

structural_mesh = fishy.evaluate(structural_mesh_parametric).reshape((-1,3))
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

# width_shape_variables = csdl.Variable(shape=(ffd_block.coefficients.shape[1]//2 + 1,), value=0., name='width_shape_deltas')
# # width_shape_variables.set_value(np.array([-6.928424636601911678e-03, -3.742655120055620593e-02, -6.632995188892187866e-02, 1.106849277260799624e-01])/100)

# width_shape_deltas = csdl.Variable(shape=(ffd_block.coefficients.shape[1],), value=0.)
# width_shape_deltas = width_shape_deltas.set(csdl.slice[0:width_shape_variables.size], width_shape_variables)
# width_shape_deltas = width_shape_deltas.set(csdl.slice[width_shape_variables.size:], width_shape_variables[-2::-1])
# deltas_sum = csdl.sum(width_shape_deltas)

# endregion -Create Parameterization Objects

# region -Define Parameterization For Solver
# length_stretches = csdl.expand(length_stretch, (num_ffd_sections,))
width_stretches = csdl.expand(width_stretch, (num_ffd_sections,))
# width_stretches = width_stretches.set(csdl.slice[1:3], 0.)
# width_stretches = width_stretches.set(csdl.slice[0], -width_stretch)
height_stretches = csdl.expand(height_stretch, (num_ffd_sections,))

ffd_sectional_parameterization_inputs = lsdo_geo.VolumeSectionalParameterizationInputs()
# ffd_sectional_parameterization_inputs.add_sectional_stretch(axis=0, stretch=length_stretches)
# ffd_sectional_parameterization_inputs.add_sectional_translation(axis=2, translation=width_stretches)
ffd_sectional_parameterization_inputs.add_sectional_stretch(axis=1, stretch=height_stretches)
ffd_block_coefficients = ffd_sectional_parameterization.evaluate(ffd_sectional_parameterization_inputs, plot=False)

# width_shape_deltas_expanded = csdl.expand(width_shape_deltas, ffd_block_coefficients.shape[:2], 'i->ji')
# ffd_block_coefficients = ffd_block_coefficients.set(csdl.slice[:,:,-1,2], ffd_block_coefficients[:,:,-1,2] + width_shape_deltas_expanded)
# ffd_block_coefficients = ffd_block_coefficients.set(csdl.slice[:,:,0,2], ffd_block_coefficients[:,:,0,2] - width_shape_deltas_expanded)

fishy_coefficients = ffd_block.evaluate(coefficients=ffd_block_coefficients, plot=False)
fishy.coefficients = fishy_coefficients.reshape(fishy.coefficients.shape)

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
# length = csdl.Variable(value=computed_fishy_length.value, name='length')
# width = csdl.Variable(value=computed_fishy_width.value, name='width')
height = csdl.Variable(value=computed_fishy_height.value, name='height')

# length = csdl.Variable(value=1.1, name='length')
# width = csdl.Variable(value=0.02, name='width')
# height = csdl.Variable(value=0.07, name='height')

# endregion Geometric Design Variables

geometry_parameterization_solver = lsdo_geo.ParameterizationSolver()

# geometry_parameterization_solver.add_parameter(length_stretch)
# geometry_parameterization_solver.add_parameter(width_stretch)
geometry_parameterization_solver.add_parameter(height_stretch)

# geometric_parameterization_variables = lsdo_geo.GeometricVariables()
# # geometric_parameterization_variables.add_variable(computed_fishy_length, length)
# # geometric_parameterization_variables.add_variable(computed_fishy_width, width)
# geometric_parameterization_variables.add_variable(computed_fishy_height, height)

# geometry_parameterization_solver.evaluate(geometric_variables=geometric_parameterization_variables)

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

structural_mesh_displacements_baseline = csdl.Variable(value=np.zeros(structural_mesh_nodes.shape).flatten(), name='structural_mesh_displacements_baseline')
# endregion Evaluate Meshes
# endregion Geoemetry Parameterization

# pump_pressure = csdl.Variable(value=0.e4, name='pump_pressure')
E_dragon = csdl.Variable(value=3.e5, name='E_dragon')
nu_dragon = csdl.Variable(value=0.45, name='nu_dragon')
E_fr4 = csdl.Variable(value=0.9e8, name='E_fr4')
nu_fr4 = csdl.Variable(value=0.12, name='nu_fr4')

# region Run Structural Solver for Baseline Design
recorder.inline = False
pressures_baseline = pressures_baseline*6894.76
pump_pressures = pressures_baseline
pump_pressures_csdl = csdl.Variable(value=pump_pressures, name='pump_pressures')
bending_angles = csdl.Variable(shape=(len(pump_pressures),), value=0., name='bending_angles_baseline')
for j in csdl.frange(pump_pressures_csdl.size):
    pump_pressure = pump_pressures_csdl[j]
    # print('------------------------------------')
    # print('Structural Mesh Displacements Norm For Baseline Design')
    # print(csdl.norm(structural_mesh_displacements).value)
    # print('------------------------------------')

    structural_displacements, _, new_mesh_cooordinates = lsdo_soft_csm.robotic_fish_static_structural_model(structural_mesh_displacements_baseline, pump_pressure,
                                                                                                            E_dragon, nu_dragon, E_fr4, nu_fr4)
    # structural_displacements, _, new_mesh_cooordinates = lsdo_soft_csm.robotic_fish_static_structural_model(structural_mesh_displacements, pressure_input_coefficients)
    if new_mesh_cooordinates.shape != structural_mesh.shape:
        structural_mesh = new_mesh_cooordinates
        structural_mesh_parametric = fishy.project(points=new_mesh_cooordinates, grid_search_density_parameter=1., plot=False, projection_tolerance=1.e-3)


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


    derivative_at_module_edge1 = deformed_module.evaluate(np.array([[0., 0.5, 0.5]]), parametric_derivative_orders=(1,0,0))
    derivative_at_module_edge = derivative_at_module_edge1/csdl.norm(derivative_at_module_edge1)    # normalize
    module_edge = deformed_module.evaluate(np.array([[0., 0.5, 0.5]]))
    # endregion Fit B-Spline to Displacements and Construct Deformed Geometry


    initial_angle = csdl.Variable(shape=(3,), value=np.array([1., 0., 0.]))
    angle = csdl.arccos(csdl.vdot(initial_angle, derivative_at_module_edge))

    objective = -angle

    bending_angles = bending_angles.set(csdl.slice[j], angle)

# endregion Run Structural Solver for Baseline Design

bending_angles = bending_angles - bending_angles[0]
bending_angles_baseline = bending_angles*180/np.pi*2   # convert to degrees and multiply by 2 for symmetry

differences_baseline = bending_angles_data_baseline - csdl.expand(bending_angles_baseline, bending_angles_data_baseline.shape, 'i->ij')
# rmse = csdl.sqrt(csdl.average(differences**2))


# region Update Design To Optimized Design
height = csdl.Variable(value=7.931587973291041926e-2, name='height_optimized')

geometric_parameterization_variables = lsdo_geo.GeometricVariables()
geometric_parameterization_variables.add_variable(computed_fishy_height, height)

geometry_parameterization_solver.evaluate(geometric_variables=geometric_parameterization_variables)

width_shape_variables = csdl.Variable(shape=(ffd_block.coefficients.shape[1]//2 + 1,),
                                       value=np.array([5.731138044206841364e-3, -6.657115938444321479e-3, -7.965134789907922119e-3, -8.002576986651789070e-3]),
                                        name='width_shape_deltas')

width_shape_deltas = csdl.Variable(shape=(ffd_block.coefficients.shape[1],), value=0.)
width_shape_deltas = width_shape_deltas.set(csdl.slice[0:width_shape_variables.size], width_shape_variables)
width_shape_deltas = width_shape_deltas.set(csdl.slice[width_shape_variables.size:], width_shape_variables[-2::-1])
width_shape_deltas_expanded = csdl.expand(width_shape_deltas, ffd_block_coefficients.shape[:2], 'i->ji')
ffd_block_coefficients = ffd_block_coefficients.set(csdl.slice[:,:,-1,2], ffd_block_coefficients[:,:,-1,2] + width_shape_deltas_expanded)
ffd_block_coefficients = ffd_block_coefficients.set(csdl.slice[:,:,0,2], ffd_block_coefficients[:,:,0,2] - width_shape_deltas_expanded)

fishy_coefficients = ffd_block.evaluate(coefficients=ffd_block_coefficients, plot=False)

fishy.coefficients = fishy_coefficients

structural_mesh = fishy.evaluate(structural_mesh_parametric).reshape((-1,3))

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
# endregion Update Design To Optimized Design


# region Run Structural Solver for Optimized Design
pressures_optimized = pressures_optimized*6894.76
pump_pressures = pressures_optimized
pump_pressures_csdl = csdl.Variable(value=pump_pressures, name='pump_pressures')
bending_angles = csdl.Variable(shape=(len(pump_pressures),), value=0., name='bending_angles_optimized')
for j in csdl.frange(pump_pressures_csdl.size):
    pump_pressure = pump_pressures_csdl[j]
    # print('------------------------------------')
    # print('Structural Mesh Displacements Norm For Optimized Design')
    # print(csdl.norm(structural_mesh_displacements).value)
    # print('------------------------------------')

    structural_displacements, _, new_mesh_cooordinates = lsdo_soft_csm.robotic_fish_static_structural_model(structural_mesh_displacements, pump_pressure,
                                                                                                            E_dragon, nu_dragon, E_fr4, nu_fr4)
    # structural_displacements, _, new_mesh_cooordinates = lsdo_soft_csm.robotic_fish_static_structural_model(structural_mesh_displacements, pressure_input_coefficients)
    if new_mesh_cooordinates.shape != structural_mesh.shape:
        structural_mesh = new_mesh_cooordinates
        structural_mesh_parametric = fishy.project(points=new_mesh_cooordinates, grid_search_density_parameter=1., plot=False, projection_tolerance=1.e-3)


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


    derivative_at_module_edge1 = deformed_module.evaluate(np.array([[0., 0.5, 0.5]]), parametric_derivative_orders=(1,0,0))
    derivative_at_module_edge = derivative_at_module_edge1/csdl.norm(derivative_at_module_edge1)    # normalize
    module_edge = deformed_module.evaluate(np.array([[0., 0.5, 0.5]]))
    # endregion Fit B-Spline to Displacements and Construct Deformed Geometry


    initial_angle = csdl.Variable(shape=(3,), value=np.array([1., 0., 0.]))
    angle = csdl.arccos(csdl.vdot(initial_angle, derivative_at_module_edge))

    objective = -angle

    bending_angles = bending_angles.set(csdl.slice[j], angle)

# endregion Run Structural Solver for Baseline Design

bending_angles = bending_angles - bending_angles[0]
bending_angles_optimized = bending_angles*180/np.pi*2   # convert to degrees and multiply by 2 for symmetry

differences_optimized = bending_angles_data_optimized - csdl.expand(bending_angles_optimized, bending_angles_data_optimized.shape, 'i->ij')
all_differences = csdl.concatenate([differences_baseline.flatten(), differences_optimized.flatten()])
rmse = csdl.sqrt(csdl.average(all_differences**2))

# region Optimization
E_dragon.set_as_design_variable(lower=1.75e5, upper=1.e7, scaler=1.e-5)
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

# Optimized parameters for baseline design
# E_dragon.set_value(2.294259270093112590e+5)
# nu_dragon.set_value(3.000000000000000000e-1)
# E_fr4.set_value(5.410652348884978924e+7)

# # For combined optimization, guess halfway between baseline and optimized
# E_dragon.set_value((2.294259270093112590e+5 + 3.661597153363377544e+5)/2)
# nu_dragon.set_value((3.000000000000000000e-1 + 4.183650217403863003e-1)/2)
# E_fr4.set_value((5.410652348884978924e+7 + 4.947969608814554165e+7)/2)

# Optimized results (SKETCHY)
# E_dragon.set_value(2.608265819840872801e+5)
# nu_dragon.set_value(3.000000000000000000e-1)
# E_fr4.set_value(5.833226413830866486e+7)

# Best result during optimization
E_dragon.set_value(2.120794898134622741e+5)
nu_dragon.set_value(3.000000000000000444e-1)
E_fr4.set_value(1.133121251595327195e+9)

# # Manual guess based on result from above parameters
# E_dragon.set_value(1.520794898134622741e+5)
# nu_dragon.set_value(3.0e-1)
# E_fr4.set_value(2.133121251595327195e+9)

# sim = csdl.experimental.PySimulator(recorder=recorder)
additional_inputs = [E_dragon, nu_dragon, E_fr4]
additional_outputs = [objective, bending_angles_baseline, bending_angles_optimized]
sim = csdl.experimental.JaxSimulator(
    recorder = recorder,
    additional_inputs=additional_inputs,
    additional_outputs=additional_outputs,
    gpu=False
)

# sim.run()

# print('bending_angles_baseline: ', bending_angles_baseline.value)
# print('bending_angles_optimized: ', bending_angles_optimized.value)
# # print('bending_angles_expanded: ', csdl.expand(bending_angles, bending_angles_data.shape, 'i->ij').value)
# # print('bending_angles_data: ', bending_angles_data)
# print('differences: ', all_differences.value)
# print('RMSE: ', rmse.value)

# import matplotlib.pyplot as plt
# plt.plot(pressures_baseline/1e3, bending_angles_baseline.value, '-o', label='Simulated Baseline', color='red')
# plt.plot(pressures_optimized/1e3, bending_angles_optimized.value, '-o', label='Simulated Optimized', color='orange')
# plt.legend()
# plt.show()
# exit()


optimization_problem = CSDLAlphaProblem(problem_name='fishy_optimization', simulator=sim)
optimizer = PySLSQP(optimization_problem, solver_options={'maxiter':100, 'acc':1.e-8}, readable_outputs=['x'])
# recorder.execute()

optimizer.solve()
optimizer.print_results()

plt.plot(pressures_baseline/1e3, bending_angles_baseline.value, '--o', label='Simulated Baseline Design', color='red')
plt.plot(pressures_optimized/1e3, bending_angles_optimized.value, '-o', label='Simulated Optimized Design', color='orange')
plt.legend()
plt.show()
exit()
# endregion Optimization