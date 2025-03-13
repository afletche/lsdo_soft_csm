import csdl_alpha as csdl
import lsdo_soft_csm
import numpy as np
import csdml
import pickle
import lsdo_function_spaces as lfs
import lsdo_geo
import meshio
import time
import vedo
import jax
# enable 64
# jax.config.update("jax_enable_x64", True)

recorder = csdl.Recorder(inline=True)
recorder.start()


# region Import and Setup
def import_geometry() -> lfs.Function:
    with open("examples/advanced_examples/robotic_fish/pickle_files/fishy_volume_geometry_fine.pickle", 'rb') as handle:
    # with open("examples/advanced_examples/robotic_fish/pickle_files/fishy_mini_v1_volume_geometry_2.pickle", 'rb') as handle:
        fishy = pickle.load(handle)
        # fishy.coefficients = csdl.Variable(value=fishy.coefficients.value, name='fishy_coefficients')   # Remake variables because can't pickle variables
        fishy.coefficients = csdl.Variable(value=fishy.coefficients, name='fishy_coefficients')   # Remake variables because can't pickle variables
        return fishy

fishy = import_geometry()
fishy.name = 'fishy'
fishy.coefficients.reshape((51,15,15,3))
# fishy.plot()

# endregion Import and Setup

# region -Structural Mesh Projection
mesh_file_path = "examples/advanced_examples/robotic_fish/meshes/"
# mesh_name = "module_v1_fine"
# mesh_name = "module_v1"
mesh_name = "module_v1_1point5mm"
# mesh_name = "module_v1_refined_2"
# mesh_name = "module_v1_refined_3"
# mesh_name = "module_v1_refinement_study_2point5mm"
structural_mesh = meshio.read(mesh_file_path + mesh_name + ".msh")
structural_mesh_nodes = structural_mesh.points/1000 + np.array([0.04, 0, 0.])   # Shift the mesh to the right to make it the middle module
structural_elements = structural_mesh.cells_dict['tetra']

# vedo_mesh = vedo.Mesh([structural_mesh_nodes, structural_elements]).wireframe()
# plotter = vedo.Plotter()
# plotter.show(vedo_mesh, axes=1, viewup='y')

# region Extract fenicsx information from pickle files
# Reorder the nodes to match the FEniCS mesh
# Also get the facet dofs for the pump input
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
# endregion Extract fenicsx information from pickle files

structural_mesh_parametric = fishy.project(points=structural_mesh_nodes, grid_search_density_parameter=1., plot=False,
                                           projection_tolerance=1e-3, newton_tolerance=1e-12)
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

fishy_right_channel_edge_parametric = fishy.project(points=np.array([[0., 0., -0.02]]), plot=False)
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
# ffd_block = lsdo_geo.construct_ffd_block_from_corners(entities=fishy, corners=ffd_corners, num_coefficients=(2,5,2), degree=(1,2,1))
ffd_block = lsdo_geo.construct_ffd_block_from_corners(entities=fishy, corners=ffd_corners, num_coefficients=(2,7,2), degree=(1,2,1))
# ffd_block = lsdo_geo.construct_ffd_block_from_corners(entities=fishy, corners=ffd_corners, num_coefficients=(2,9,2), degree=(1,2,1))
# ffd_block = lsdo_geo.construct_ffd_block_from_corners(entities=fishy, corners=ffd_corners, num_coefficients=(2,11,2))
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

# width_shape_variable = csdl.Variable(shape=(1,), value=-0., name='width_shape_deltas')
width_shape_variables = csdl.Variable(shape=(ffd_block.coefficients.shape[1]//2 + 1,), value=0., name='width_shape_deltas')
# width_shape_variables = width_shape_variables.set(csdl.slice[:], width_shape_variable)

# width_shape_variables.set_value(np.array([5.731138044206841364e-3, -6.657115938444321479e-3, -7.965134789907922119e-3, -8.002576986651789070e-3]))

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
# height.set_value(7.931587973291041926e-2)

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

# ffd_block_plot = ffd_block.plot(plot_embedded_points=False, show=False)
# fishy.plot(opacity=1., additional_plotting_elements=[ffd_block_plot])
# exit()

fishy_left = fishy.evaluate(fishy_left_parametric)
fishy_right = fishy.evaluate(fishy_right_parametric)
computed_fishy_width = csdl.norm(fishy_left - fishy_right)
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

# region Evaluate Meshes
# endregion Geoemetry Parameterization

# region Define Control
actuation_frequency = csdl.Variable(value=0.5, name='actuation_frequency')
actuation_period = 1/actuation_frequency
# initial_volume = length.value*computed_fishy_width.value*height.value
# average_width_change = 1 + csdl.sum(width_shape_deltas)/width_shape_deltas.size/(computed_fishy_width.value)
# new_volume = length*computed_fishy_width.value*(average_width_change)*height
# volume_ratio = new_volume/initial_volume
# frequency_of_max_pressure = 0.5/(volume_ratio)

# pump_pressure = csdl.Variable(value=4.5e4, name='base_max_pressure')
# pump_pressure = csdl.Variable(value=3.75e4, name='base_max_pressure')
pump_pressure = csdl.Variable(value=2.5e4, name='base_max_pressure')
# pump_pressure = csdl.Variable(value=2.472724806605190562e+4, name='base_max_pressure')
# pump_pressure = csdl.Variable(value=1.e4, name='base_max_pressure')
# pump_pressure = csdl.Variable(value=0.e4, name='base_max_pressure')
one_hertz_max_pressure = 3.5e4    # Pa
half_hertz_max_pressure = 5e4    # Pa
delta_frequency = 0.5
slope = (one_hertz_max_pressure - half_hertz_max_pressure)/delta_frequency
intercept = half_hertz_max_pressure
max_pressure = slope*(actuation_frequency - 0.5) + intercept
frequency_pressure_bound = pump_pressure - max_pressure    # Must be less than 0 to be feasible
# NOTE: This doesn't currently account for changes in chamber volume
# endregion Define Control

# region Structural Solver
# E_dragon = csdl.Variable(value=3.e5, name='E_dragon')
# nu_dragon = csdl.Variable(value=0.45, name='nu_dragon')
# E_fr4 = csdl.Variable(value=0.9e8, name='E_fr4')
# nu_fr4 = csdl.Variable(value=0.12, name='nu_fr4')
E_dragon = csdl.Variable(value=2.3794898134622741e+5, name='E_dragon')
nu_dragon = csdl.Variable(value=0.3, name='nu_dragon')
E_fr4 = csdl.Variable(value=0.223121251595327195e+9, name='E_fr4')
nu_fr4 = csdl.Variable(value=0.12, name='nu_fr4')
# E_dragon.set_value(2.4794898134622741e+5)
# nu_dragon.set_value(3.0e-1)
# E_fr4.set_value(0.223121251595327195e+9)

# structural_displacements, structural_applied_work, new_mesh_cooordinates = lsdo_soft_csm.robotic_fish_static_structural_model(structural_mesh_displacements, pump_pressure,
#                                                                                                             E_dragon, nu_dragon, E_fr4, nu_fr4)
# if new_mesh_cooordinates.shape != structural_mesh.shape:
#     structural_mesh = new_mesh_cooordinates
#     structural_mesh_parametric = fishy.project(points=new_mesh_cooordinates, grid_search_density_parameter=1., plot=False, projection_tolerance=1.e-3)


def import_training_data(filename):
    training_data_inputs_1 = []
    training_data_inputs_2 = []
    training_data_inputs_3 = []
    training_data_outputs_1 = []
    training_data_outputs_2 = []
    import h5py
    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        # print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        # a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        # print(type(f[a_group_key])) 

        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        # data = list(f[a_group_key])

        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        # data = list(f[a_group_key])
        # preferred methods to get dataset values:
        # ds_obj = f[a_group_key]      # returns as a h5py dataset object
        for sample_name in f.keys():
            sample_group = f[sample_name]
            sample_width_shape_variables = sample_group['width_shape_deltas'][()].reshape((-1,1))
            training_data_inputs_1.append(sample_width_shape_variables)
            sample_pump_pressure = sample_group['base_max_pressure'][()].reshape((-1,1))
            training_data_inputs_2.append(sample_pump_pressure)
            sample_displacement = sample_group['variable_0'][()].reshape((-1,1))
            training_data_outputs_1.append(sample_displacement)
            sample_applied_work = sample_group['applied_work'][()].reshape((-1,1))
            training_data_outputs_2.append(sample_applied_work)

            if filename == 'saved_data_6_dims.hdf5':
                sample_height = sample_group['height'][()].reshape((-1,1))
                training_data_inputs_3.append(sample_height)
        training_data_inputs_1 = np.hstack(training_data_inputs_1)
        training_data_inputs_2 = np.hstack(training_data_inputs_2)
        training_data_outputs_1 = np.hstack(training_data_outputs_1)
        training_data_outputs_2 = np.hstack(training_data_outputs_2)
        return training_data_inputs_1, training_data_inputs_2, training_data_inputs_3, \
            training_data_outputs_1, training_data_outputs_2

# region train ML model

# region load training/test data
training_data_width_shape_variables_5_dims, training_data_pump_pressure_5_dims,_, \
    training_data_displacements_5_dims, training_data_applied_work_5_dims = \
    import_training_data('examples/advanced_examples/robotic_fish/training_data/saved_data_5_dims.hdf5')
training_data_width_shape_variables_6_dims, training_data_pump_pressure_6_dims, training_data_height_6_dims, \
    training_data_displacements_6_dims, training_data_applied_work_6_dims = \
    import_training_data('examples/advanced_examples/robotic_fish/training_data/saved_data_6_dims.hdf5')
training_data_width_shape_variables = np.hstack([training_data_width_shape_variables_5_dims, training_data_width_shape_variables_6_dims])
training_data_pump_pressure = np.hstack([training_data_pump_pressure_5_dims, training_data_pump_pressure_6_dims])
training_data_displacements = np.hstack([training_data_displacements_5_dims, training_data_displacements_6_dims])
training_data_applied_work = np.hstack([training_data_applied_work_5_dims, training_data_applied_work_6_dims])


training_data_geometry_coefficients = np.load('examples/advanced_examples/robotic_fish/training_data/training_data_geometry_coefficients.npy')
# training_data_mesh_displacements = np.load('examples/advanced_examples/robotic_fish/training_data/training_data_mesh_displacements.npy')
num_test_data = 50
test_data_geometry_coefficients = training_data_geometry_coefficients[:, -num_test_data:]
# test_data_mesh_displacements = training_data_mesh_displacements[:, -num_test_data:]
test_data_pump_pressure = training_data_pump_pressure[:, -num_test_data:]
test_data_displacements = training_data_displacements[:, -num_test_data:]
test_data_applied_work = training_data_applied_work[:, -num_test_data:]

training_data_geometry_coefficients = training_data_geometry_coefficients[:, :-num_test_data]
# training_data_mesh_displacements = training_data_mesh_displacements[:, :-num_test_data]
training_data_pump_pressure = training_data_pump_pressure[:, :-num_test_data]
training_data_displacements = training_data_displacements[:, :-num_test_data]
training_data_applied_work = training_data_applied_work[:, :-num_test_data]

training_data_inputs = np.vstack([training_data_geometry_coefficients, training_data_pump_pressure]).T
test_data_inputs = np.vstack([test_data_geometry_coefficients, test_data_pump_pressure]).T
# training_data_inputs = np.vstack([training_data_mesh_displacements, training_data_pump_pressure]).T
# test_data_inputs = np.vstack([test_data_mesh_displacements, test_data_pump_pressure]).T
training_data_outputs = np.vstack([training_data_displacements, training_data_applied_work]).T
test_data_outputs = np.vstack([test_data_displacements, test_data_applied_work]).T

training_data_input_mins = training_data_inputs.min(axis=0)
training_data_input_maxs = training_data_inputs.max(axis=0)
training_data_output_mins = training_data_outputs.min(axis=0)
training_data_output_maxs = training_data_outputs.max(axis=0)
training_inputs_max_minus_min = training_data_input_maxs - training_data_input_mins
training_outputs_max_minus_min = training_data_output_maxs - training_data_output_mins
training_inputs_max_minus_min = np.where(training_inputs_max_minus_min == 0, 1, training_inputs_max_minus_min)
training_outputs_max_minus_min = np.where(training_outputs_max_minus_min == 0, 1, training_outputs_max_minus_min)
training_data_inputs = (training_data_inputs - training_data_input_mins) / training_inputs_max_minus_min
test_data_inputs = (test_data_inputs - training_data_input_mins) / training_inputs_max_minus_min
training_data_outputs = (training_data_outputs - training_data_output_mins) / training_outputs_max_minus_min
test_data_outputs = (test_data_outputs - training_data_output_mins) / training_outputs_max_minus_min

input_size = fishy.coefficients.size + 1
output_size = structural_mesh_displacements.size + 1
model = csdml.FCNN(input_dim=input_size, hidden_dims=[80, 80, 80], output_dim=output_size, activation=['tanh', 'tanh', 'tanh', None])

# best_param_vals = pickle.load(open('best_param_vals_80_80_80_tanh_new_loss.pkl', 'rb'))
best_param_vals = pickle.load(open('best_param_vals.pkl', 'rb'))
model.set_param_values(best_param_vals)


fishy_coefficients = csdl.Variable(value=fishy.coefficients.flatten().value, name='fishy_coefficients')
# inputs = csdl.concatenate([fishy.coefficients.flatten(), pump_pressure]).reshape((1,-1))
inputs = csdl.concatenate([fishy_coefficients, pump_pressure]).reshape((1,-1))
t1 = time.time()
outputs = model.forward(inputs)
t2 = time.time()
print('Time to evaluate model INLINE:', t2-t1)

outputs = outputs[0,:]

# Get unscaled outputs
outputs = outputs * training_outputs_max_minus_min + training_data_output_mins
displacements = outputs[:-1]
applied_work = outputs[-1]

print('Displacements:', displacements.value.reshape((-1,3)))
print('Applied Work:', applied_work.value)

# print('True Displacements: ', structural_displacements.value.reshape((-1,3)))
# print('True Applied Work: ', structural_applied_work.value)
displaced_mesh = structural_mesh + displacements.reshape((-1,3))

initial_displaced_mesh = displaced_mesh.value

# # Plot structural solver result
# # fishy_plot = fishy.plot(show=False, opacity=0.3)
# vedo_mesh = vedo.Mesh([displaced_mesh.value, structural_elements]).wireframe().color('lightgreen').opacity(0.5)
# plotter = vedo.Plotter()
# # plotter.show([fishy_plot, vedo_mesh], axes=1, viewup='y')
# plotter.show([vedo_mesh], axes=0, viewup='y')

# print('Time to evaluate model:', t2-t1)


additional_inputs = [pump_pressure, fishy_coefficients]
additional_outputs = [displacements]

sim = csdl.experimental.JaxSimulator(
    recorder = recorder,
    additional_inputs=additional_inputs,
    additional_outputs=additional_outputs,
    gpu=False
)

sim.run()

num_times = 100
t1 = time.time()
for i in range(num_times):
    sim.run()
t2 = time.time()
print('Time to run simulation:', (t2-t1)/num_times)