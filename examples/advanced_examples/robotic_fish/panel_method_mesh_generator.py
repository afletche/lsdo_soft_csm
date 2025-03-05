import csdl_alpha as csdl
import lsdo_geo
import lsdo_function_spaces as lfs
import lsdo_soft_csm
from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver
from lsdo_serpent.core.fish_post_processor import fish_post_processor

import pickle
import meshio
import numpy as np
from modopt import PySLSQP
from modopt import CSDLAlphaProblem
import matplotlib.pyplot as plt
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
# mesh_name = "module_v1_refined_2"
# mesh_name = "module_v1_refined_3"
mesh_name = "module_v1_refinement_study_2point5mm"
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

# region Fluid Mesh Parametric Coordinates Definition
# num_panels_per_dimension = 5    # NOTE: Probably want to individually manipulate this for each direction.
# side_num_chordwise = 7  # not including front/back contribution
# side_num_chordwise = 13  # not including front/back contribution    NOTE: Been running with this
# side_num_chordwise = 15  # not including front/back contribution
# side_num_chordwise = 17  # not including front/back contribution
side_num_chordwise = 21  # not including front/back contribution
# side_num_chordwise = 25  # not including front/back contribution
# side_num_chordwise = 34  # not including front/back contribution
# side_num_spanwise = 5  # not including top/bottom contribution      NOTE: Been running with this
# side_num_spanwise = 7  # not including top/bottom contribution
# side_num_spanwise = 9  # not including top/bottom contribution
side_num_spanwise = 11  # not including top/bottom contribution
# side_num_spanwise = 17  # not including top/bottom contribution
num_chordwise = side_num_chordwise + 3 + 5
num_spanwise = side_num_spanwise + side_num_spanwise//2
parametric_grid_1 = np.zeros((3, side_num_spanwise, 3))  # First dimension can be arbitrarily set
parametric_grid_2 = np.zeros((5, side_num_spanwise, 3))  # First dimension can be arbitrarily set
parametric_grid_3 = np.zeros((side_num_chordwise, side_num_spanwise//2, 3))  # Second dimension can be arbitrarily set
parametric_grid_4 = np.zeros((side_num_chordwise, side_num_spanwise//2, 3))  # Second dimension can be arbitrarily set
parametric_grid_5 = np.zeros((side_num_chordwise, side_num_spanwise, 3))
parametric_grid_6 = np.zeros((side_num_chordwise, side_num_spanwise, 3))
# parametric_mesh_2, parametric_mesh_1 = \
#     np.meshgrid(np.linspace(0., 1., num_panels_per_dimension), np.linspace(0., 1., num_panels_per_dimension))
# parametric_grid_1 = np.zeros((num_panels_per_dimension**2, 3))
# parametric_grid_2 = np.zeros((num_panels_per_dimension**2, 3))
# parametric_grid_3 = np.zeros((num_panels_per_dimension**2, 3))
# parametric_grid_4 = np.zeros((num_panels_per_dimension**2, 3))
# parametric_grid_5 = np.zeros((num_panels_per_dimension**2, 3))
# parametric_grid_6 = np.zeros((num_panels_per_dimension**2, 3))
# parametric_grid_5 = np.zeros((num_panels_per_dimension, num_panels_per_dimension, 3))
# parametric_grid_6 = np.zeros((num_panels_per_dimension, num_panels_per_dimension, 3))

parametric_mesh_2, parametric_mesh_1 = \
    np.meshgrid(np.linspace(0., 1., side_num_spanwise), np.linspace(1., 0., 3))
parametric_grid_1[:,:,1] = parametric_mesh_2
parametric_grid_1[:,:,2] = parametric_mesh_1
parametric_mesh_2, parametric_mesh_1 = \
    np.meshgrid(np.linspace(0., 1., side_num_spanwise), np.linspace(0., 1., 5))
parametric_grid_2[:,:,0] = 1.
parametric_grid_2[:,:,1] = parametric_mesh_2
parametric_grid_2[:,:,2] = parametric_mesh_1
parametric_mesh_2, parametric_mesh_1 = \
    np.meshgrid(np.linspace(0., 1., side_num_spanwise//2), np.linspace(0., 1., side_num_chordwise))
parametric_grid_3[:,:,0] = parametric_mesh_1
# parametric_grid_3[:,1] = np.zeros(parametric_mesh_1.shape)
parametric_grid_3[:,:,2] = parametric_mesh_2
parametric_mesh_2, parametric_mesh_1 = \
    np.meshgrid(np.linspace(0., 1., side_num_spanwise//2), np.linspace(0., 1., side_num_chordwise))
parametric_grid_4[:,:,0] = parametric_mesh_1
parametric_grid_4[:,:,1] = 1.
parametric_grid_4[:,:,2] = parametric_mesh_2
parametric_mesh_2, parametric_mesh_1 = \
    np.meshgrid(np.linspace(0., 1., side_num_spanwise), np.linspace(0., 1., side_num_chordwise + 2)[1:-1])
parametric_grid_5[:,:,0] = parametric_mesh_1
parametric_grid_5[:,:,1] = parametric_mesh_2
# parametric_grid_5[:,:,2] = np.zeros(parametric_mesh_1.shape)
parametric_mesh_2, parametric_mesh_1 = \
    np.meshgrid(np.linspace(0., 1., side_num_spanwise), np.linspace(1., 0., side_num_chordwise + 2)[1:-1])
parametric_grid_6[:,:,0] = parametric_mesh_1
parametric_grid_6[:,:,1] = parametric_mesh_2
parametric_grid_6[:,:,2] = np.ones_like(parametric_mesh_1)


# panel_method_parametric_grids = [parametric_grid_1, parametric_grid_2, parametric_grid_3, parametric_grid_4, parametric_grid_5, parametric_grid_6]

# region Stich together parametric mesh

# num_chordwise = side_num_chordwise + side_num_chordwise + 5 + 4   # left, right, front, back (middle of back (TE) gets added twice)
# num_spanwise = side_num_spanwise  # top, bottom, left, right
# panel_method_parametric_mesh = np.zeros((num_chordwise, side_num_spanwise, 3))

# panel_method_parametric_mesh[:2] = parametric_grid_1[1:]  # Only later half chordwise, All spanwise
# panel_method_parametric_mesh[2:2 + side_num_chordwise] = parametric_grid_5
# panel_method_parametric_mesh[2 + side_num_chordwise: 2 + side_num_chordwise + 5] = parametric_grid_2
# panel_method_parametric_mesh[2 + side_num_chordwise + 5 : 2 + side_num_chordwise + 5 + side_num_chordwise] = \
#     parametric_grid_6 
# panel_method_parametric_mesh[2 + side_num_chordwise + 5 + side_num_chordwise:] = parametric_grid_1[:2]

# NOTE: Try just evaluating a line on the top (and bottom). For the front faces, use the same endpoint (front top point).
# -- This will create a series of triangular panels for the front face, but I think it might be necessary.
num_chordwise = side_num_chordwise + side_num_chordwise + 5 + 4   # left, right, front, back (middle of back (TE) gets added twice)
num_spanwise = side_num_spanwise + 2
panel_method_parametric_mesh = np.zeros((num_chordwise, num_spanwise, 3))

panel_method_parametric_mesh[:2,1:-1] = parametric_grid_1[1:]  # Only later half chordwise, All spanwise
panel_method_parametric_mesh[2:2 + side_num_chordwise,1:-1] = parametric_grid_5
panel_method_parametric_mesh[2 + side_num_chordwise: 2 + side_num_chordwise + 5,1:-1] = parametric_grid_2
panel_method_parametric_mesh[2 + side_num_chordwise + 5 : 2 + side_num_chordwise + 5 + side_num_chordwise,1:-1] = \
    parametric_grid_6 
panel_method_parametric_mesh[2 + side_num_chordwise + 5 + side_num_chordwise:,1:-1] = parametric_grid_1[:2]

# # - Insert top and bottom
top_line = np.linspace(np.array([0., 1., 0.5]), np.array([1., 1., 0.5]), side_num_chordwise + 2)[1:-1]
bottom_line = np.linspace(np.array([0., 0., 0.5]), np.array([1., 0., 0.5]), side_num_chordwise + 2)[1:-1]
top_line_reversed = top_line[::-1]
bottom_line_reversed = bottom_line[::-1]

panel_method_parametric_mesh[:2,0] = bottom_line[0]
panel_method_parametric_mesh[:2,-1] = top_line[0]
panel_method_parametric_mesh[2:2 + side_num_chordwise, 0] = bottom_line
panel_method_parametric_mesh[2:2 + side_num_chordwise, -1] = top_line
panel_method_parametric_mesh[2 + side_num_chordwise : 2 + side_num_chordwise + 5,0] = bottom_line[-1]
panel_method_parametric_mesh[2 + side_num_chordwise : 2 + side_num_chordwise + 5,-1] = top_line[-1]
panel_method_parametric_mesh[2 + side_num_chordwise + 5 : 2 + side_num_chordwise + 5 + side_num_chordwise,0] = bottom_line_reversed
panel_method_parametric_mesh[2 + side_num_chordwise + 5 : 2 + side_num_chordwise + 5 + side_num_chordwise,-1] = top_line_reversed
panel_method_parametric_mesh[2 + side_num_chordwise + 5 + side_num_chordwise:,0] = bottom_line[0]
panel_method_parametric_mesh[2 + side_num_chordwise + 5 + side_num_chordwise:,-1] = top_line[0]

panel_mesh_this_timestep = fishy.evaluate(panel_method_parametric_mesh, plot=False)
# endregion Stich together parametric mesh
# endregion Fluid Mesh Parametric Coordinates Definition


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
# ffd_block = lsdo_geo.construct_ffd_block_from_corners(entities=fishy, corners=ffd_corners, num_coefficients=(2,7,2), degree=(1,2,1))
ffd_block = lsdo_geo.construct_ffd_block_from_corners(entities=fishy, corners=ffd_corners, num_coefficients=(2,9,2), degree=(1,2,1))
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

# endregion Evaluate Meshes
# endregion Geoemetry Parameterization

# region Define Control
actuation_frequency = csdl.Variable(value=1., name='actuation_frequency')
actuation_period = 1/actuation_frequency
initial_volume = length.value*computed_fishy_width.value*height.value
average_width_change = 1 + csdl.sum(width_shape_deltas)/width_shape_deltas.size/(computed_fishy_width.value)
new_volume = length*computed_fishy_width.value*(average_width_change)*height
volume_ratio = new_volume/initial_volume
frequency_of_max_pressure = 0.5/(volume_ratio)

base_max_pressure = 4.e4
# base_max_pressure = 1.
d_pressure_d_frequency = 0.5*base_max_pressure
pump_pressure = base_max_pressure# - d_pressure_d_frequency*(actuation_frequency - frequency_of_max_pressure)/frequency_of_max_pressure
pump_pressure = csdl.Variable(value=1.e4, name='base_max_pressure')
# pump_pressure_constraint = (0.5, 7psi), (1, 5.25 psi)
one_hertz_max_pressure = 3.5e4    # Pa
half_hertz_max_pressure = 5e4    # Pa
delta_frequency = 0.5
slope = (one_hertz_max_pressure - half_hertz_max_pressure)/delta_frequency
intercept = half_hertz_max_pressure
max_pressure = slope*(actuation_frequency - 0.5) + intercept
frequency_pressure_bound = pump_pressure - max_pressure    # Must be less than 0 to be feasible
# pump_pressure = csdl.Variable(value=0.e4, name='base_max_pressure')
# endregion Define Control

# region Structural Solver
# pressure_input_coefficients = csdl.Variable(value=np.zeros((327437,)), name='pressure_input_coefficients')  # 20mm mesh
# pressure_input_coefficients = csdl.Variable(value=np.zeros((333115,)), name='pressure_input_coefficients')  # 10mm mesh
pressure_input_coefficients = csdl.Variable(value=np.zeros((497409,)), name='pressure_input_coefficients')  # 2point5mm mesh
pressure_input_coefficients = pressure_input_coefficients.set(csdl.slice[left_chamber_indices], -pump_pressure)
pressure_input_coefficients = pressure_input_coefficients.set(csdl.slice[right_chamber_indices], pump_pressure)
structural_displacements_flattened, applied_work = lsdo_soft_csm.robotic_fish_static_structural_model_front_fixed(structural_mesh_displacements, pressure_input_coefficients)
structural_displacements = structural_displacements_flattened.reshape((structural_displacements_flattened.size//3,3))

# sim = csdl.experimental.PySimulator(recorder)
# exit()

displaced_mesh = structural_mesh + structural_displacements

initial_displaced_mesh = displaced_mesh.value

# Plot structural solver result
# fishy_plot = fishy.plot(show=False, opacity=0.3)
# vedo_mesh = vedo.Mesh([displaced_mesh.value, structural_elements]).wireframe()
# plotter = vedo.Plotter()
# plotter.show([fishy_plot, vedo_mesh], axes=1, viewup='y')
# exit()
# endregion Structural Solver

# region Fit B-Spline to Displacements
displacement_space = lfs.BSplineSpace(
    num_parametric_dimensions=3,
    # degree=(2,2,2),
    # degree=(2,1,1),
    degree=(2,0,0),
    # coefficients_shape=(7,3,3))
    # coefficients_shape=(5,3,3))
    coefficients_shape=(5,1,1))

mesh_parametric_coordinates_in_displacement_space = structural_mesh_parametric.copy()
module_2_min_u = np.min(mesh_parametric_coordinates_in_displacement_space[:,0]) # Use the mesh projections to guarantee mesh is in module bounds
module_2_max_u = np.max(mesh_parametric_coordinates_in_displacement_space[:,0])
for i, old_u in enumerate(structural_mesh_parametric[:,0]):
    new_u = (old_u - module_2_min_u)/(module_2_max_u - module_2_min_u)
    mesh_parametric_coordinates_in_displacement_space[i,0] = new_u

structural_displacements_b_spline = displacement_space.fit_function(structural_displacements,
                                                                    parametric_coordinates=mesh_parametric_coordinates_in_displacement_space)

# deformed_module = displacement_space.fit_function(structural_mesh + structural_displacements, 
#                                                   parametric_coordinates=mesh_parametric_coordinates_in_displacement_space)

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

# endregion Fit B-Spline to Displacements

# region Construct Dynamic Geometry and Dynamic Panel Method Mesh

module_1_min_u = module_2_max_u
# module_1_max_u = fishy.project(points=np.array([0.08 + 0.1 + 0.04, 0., 0.]), plot=False)[0,0]
module_1_max_u = fishy.project(points=np.array([0.08 + 0.08 + 0.04, 0., 0.]), plot=False)[0,0]

module_3_min_u = fishy.project(points=np.array([0. - 0.1 + 0.04, 0., 0.]), plot=False)[0,0]
module_3_max_u = module_2_min_u

opposite_structural_displacements_b_spline = structural_displacements_b_spline.copy()
coefficients_shape = opposite_structural_displacements_b_spline.space.coefficients_shape + (3,)
# indices = np.arange(np.prod(coefficients_shape)).reshape(coefficients_shape)
# want to flip the z component of the coefficients and invert the ordering of the x component along the 3rd axis of coefficients
# z_indicies = indices[:,:,:,2].reshape((-1,))
for i in range(coefficients_shape[2]):
    # flip x component
    # set_indices = indices[:,:,i,0].reshape((-1,))
    # get_indices = indices[:,:,coefficients_shape[2]-i-1,0].reshape((-1,))
    # opposite_structural_displacements_b_spline.coefficients[set_indices] = structural_displacements_b_spline.coefficients[get_indices]

    opposite_structural_displacements_b_spline.coefficients = \
        opposite_structural_displacements_b_spline.coefficients.set(csdl.slice[:,:,i,0],
                                opposite_structural_displacements_b_spline.coefficients[:,:,coefficients_shape[2]-i-1,0])

    # flip z component
    # set_indices = indices[:,:,i,2].reshape((-1,))
    # get_indices = indices[:,:,coefficients_shape[2]-i-1,2].reshape((-1,))
    # opposite_structural_displacements_b_spline.coefficients[set_indices] = -structural_displacements_b_spline.coefficients[get_indices]

    opposite_structural_displacements_b_spline.coefficients = \
        opposite_structural_displacements_b_spline.coefficients.set(csdl.slice[:,:,i,2],
                                -opposite_structural_displacements_b_spline.coefficients[:,:,coefficients_shape[2]-i-1,2])


# Interpolate between the two displacement fields
time_constant = 3
# num_panel_method_time_steps = 7
# num_panel_method_time_steps = 41
# num_panel_method_time_steps = 81
# num_cycles = 3
# num_cycles = 2
num_cycles = 1
final_time = actuation_period*num_cycles
num_modules = 3
# num_steps_per_cycle = 21
# num_steps_per_cycle = 22
# num_steps_per_cycle = 31
# num_steps_per_cycle = 33
num_steps_per_cycle = 41
# num_steps_per_cycle = 17
# num_steps_per_cycle = 13
num_panel_method_time_steps = num_steps_per_cycle*num_cycles
panel_method_dt = final_time/num_panel_method_time_steps

# time = np.linspace(0, actuation_period*num_cycles, num_panel_method_time_steps)
time = csdl.Variable(value=np.linspace(0, final_time.value[0], num_panel_method_time_steps), name='time')
time_expanded = csdl.expand(time, (num_panel_method_time_steps,num_chordwise,num_spanwise,3), 'i->ijkl')   #TODO: Don't hardcode this shape
# time = csdl.average(time_expanded, axes=(1,2,3))

# num_strokes = int(actuation_frequency*time[-1])*2+1

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

# left_chamber_inputs = []
# right_chamber_inputs = []
# max_weight = 2.
# for stroke_index in range(num_strokes):
#     if stroke_index % 2 == 0:
#         left_chamber_inputs.append(max_weight)
#         right_chamber_inputs.append(0.)
#     else:
#         left_chamber_inputs.append(0.)
#         right_chamber_inputs.append(max_weight)

# t_pressure_inputs = np.linspace(0, time[-1], int(num_strokes))
# t_pressure_inputs[1:] = t_pressure_inputs[1:] - (actuation_period/2)/2
# initial_weight = max_weight/2
# deformed_fishy_weights = compute_chamber_pressure_function(t_pressure_inputs, left_chamber_inputs, time_constant, initial_weight, time)
# deformed_fishy_weights = np.sin(2*np.pi*actuation_frequency*time) + 1
# decay_factor = 0.1
# pressure_scaling_factor = 0.8
# initial_volume = length.value*computed_fishy_width.value*height.value
# average_width_change = 1 + csdl.sum(width_shape_deltas)/width_shape_deltas.size/(computed_fishy_width.value)
# new_volume = length*average_width_change*height
# volume_ratio = new_volume/initial_volume

# frequency_of_max_pressure = 0.5/(volume_ratio*pressure_scaling_factor)
# max_pressure_scaling = 1 - decay_factor*(actuation_frequency - frequency_of_max_pressure)/frequency_of_max_pressure
max_pressure_scaling = 1.
deformed_fishy_weights = max_pressure_scaling*csdl.sin(2*np.pi*actuation_frequency*time) + 1
deformed_fishy_weights_module_2 = max_pressure_scaling*csdl.sin(2*np.pi*actuation_frequency*time - 2*np.pi/3) + 1
deformed_fishy_weights_module_3 = max_pressure_scaling*csdl.sin(2*np.pi*actuation_frequency*time - 2*2*np.pi/3) + 1

# d_deformed_fishy_weights_d_t = np.cos(2*np.pi*actuation_frequency*time)*2*np.pi*actuation_frequency

# plt.plot(time, deformed_fishy_weights, label='module 1')
# plt.plot(time[:-time.shape[0]//9] + time[time.shape[0]//9], deformed_fishy_weights[:-time.shape[0]//9], label='module 2')
# plt.plot(time[:-2*time.shape[0]//9] + time[2*time.shape[0]//9], deformed_fishy_weights[:-2*time.shape[0]//9], label='module 3')
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel('Interpolation Weight')
# plt.title('Interpolation Weights vs. Time')
# plt.show()
# exit()

# new_deformed_fishy_weights = []
# new_opposite_deformed_fishy_weights = []
# # new_d_deformed_fishy_weights_d_t = []
# # new_d_opposite_deformed_fishy_weights_d_t = []
# deformed_side_interpolation_indices = []
# opposite_deformed_side_interpolation_indices = []
# for i, weight in enumerate(deformed_fishy_weights):
#     if weight >= 1:
#         new_deformed_fishy_weights.append(weight - 1)     # - 1 rescales 1-2 to 0-1
#         deformed_side_interpolation_indices.append(i)
#         # new_d_deformed_fishy_weights_d_t.append(d_deformed_fishy_weights_d_t[i])
#     else:
#         new_opposite_deformed_fishy_weights.append(1 - weight)  # 1- flips the sign so it's the weight on the deformed fishy
#         opposite_deformed_side_interpolation_indices.append(i)
#         # new_d_opposite_deformed_fishy_weights_d_t.append(-d_deformed_fishy_weights_d_t[i])
# new_deformed_fishy_weights = np.array(new_deformed_fishy_weights)
# new_opposite_deformed_fishy_weights = np.array(new_opposite_deformed_fishy_weights)
# # new_d_deformed_fishy_weights_d_t = np.array(new_d_deformed_fishy_weights_d_t)
# # new_d_opposite_deformed_fishy_weights_d_t = np.array(new_d_opposite_deformed_fishy_weights_d_t)

# new_deformed_fishy_weights = []
# new_opposite_deformed_fishy_weights = []
deformed_side_interpolation_indices = []
opposite_deformed_side_interpolation_indices = []
deformed_side_interpolation_indices_module_2 = []
opposite_deformed_side_interpolation_indices_module_2 = []
deformed_side_interpolation_indices_module_3 = []
opposite_deformed_side_interpolation_indices_module_3 = []
for i in range(deformed_fishy_weights.size):
    if deformed_fishy_weights[i].value >= 1:
        # new_deformed_fishy_weights.append(deformed_fishy_weights[i] - 1)     # - 1 rescales 1-2 to 0-1
        deformed_side_interpolation_indices.append(i)
        # new_d_deformed_fishy_weights_d_t.append(d_deformed_fishy_weights_d_t[i])
        # deformed_side_interpolation_indices_module_2.append(i)
        # deformed_side_interpolation_indices_module_3.append(i)
    else:
        # new_opposite_deformed_fishy_weights.append(1 - deformed_fishy_weights[i])  # 1- flips the sign so it's the weight on the deformed fishy
        opposite_deformed_side_interpolation_indices.append(i)
        # new_d_opposite_deformed_fishy_weights_d_t.append(-d_deformed_fishy_weights_d_t[i])
        # opposite_deformed_side_interpolation_indices_module_2.append(i)
        # opposite_deformed_side_interpolation_indices_module_3.append(i)

for i in range(deformed_fishy_weights.size):
    if deformed_fishy_weights_module_2[i].value >= 1:
        deformed_side_interpolation_indices_module_2.append(i)
    else:
        opposite_deformed_side_interpolation_indices_module_2.append(i)

for i in range(deformed_fishy_weights.size):
    if deformed_fishy_weights_module_3[i].value >= 1:
        deformed_side_interpolation_indices_module_3.append(i)
    else:
        opposite_deformed_side_interpolation_indices_module_3.append(i)
# new_deformed_fishy_weights = np.array(new_deformed_fishy_weights)
# new_opposite_deformed_fishy_weights = np.array(new_opposite_deformed_fishy_weights)

############################
recorder.inline = True
############################

new_deformed_fishy_weights = deformed_fishy_weights[deformed_side_interpolation_indices] - 1
new_opposite_deformed_fishy_weights = 1 - deformed_fishy_weights[opposite_deformed_side_interpolation_indices]
new_deformed_fishy_weights_module_2 = deformed_fishy_weights_module_2[deformed_side_interpolation_indices_module_2] - 1
new_opposite_deformed_fishy_weights_module_2 = 1 - deformed_fishy_weights_module_2[opposite_deformed_side_interpolation_indices_module_2]
new_deformed_fishy_weights_module_3 = deformed_fishy_weights_module_3[deformed_side_interpolation_indices_module_3] - 1
new_opposite_deformed_fishy_weights_module_3 = 1 - deformed_fishy_weights_module_3[opposite_deformed_side_interpolation_indices_module_3]


def linear_combination(x:tuple[csdl.Variable], weights:tuple[csdl.Variable]):
    '''
    Performs a linear combination of CSDL variables x with weights w.

    Parameters
    ----------
    x : tuple[csdl.Variable]
        The variables to be combined.
    weights : tuple[csdl.Variable]
        The weights to be used in the combination. The tuple length corresponds to the number of variables in x.
        Each CSDL variable of weights may have a shape corresponding to the number of interpolations to perform.
        If this shape is greater than 1, the additional axis (or axes) will be added to the first axis of the variable.
    '''
    num_steps = weights[0].shape
    for weight in weights:
        if weight.shape != num_steps:
            raise ValueError('All weights must have the same shape.')
        
    x_shape = x[0].shape
    for part in x:
        if part.shape != x_shape:
            raise ValueError('All variables must have the same number of steps.')
        
    num_steps_flattened = np.prod(num_steps)
    output_flattened = csdl.Variable(value=np.zeros((num_steps_flattened,) + x_shape))

    for i in range(len(x)):
        for j in csdl.frange(num_steps_flattened):
            output_flattened = output_flattened.set(csdl.slice[j], output_flattened[j] + x[i]*weights[i][j])

    output = output_flattened.reshape(num_steps + x_shape)
    return output

def linspace(start, stop, num_steps):
    start_weights = np.linspace(0, 1, num_steps)
    stop_weights = 1 - start_weights
    output = linear_combination((start, stop), (start_weights, stop_weights))
    return output

zero_displacements = csdl.Variable(value=np.zeros(structural_displacements_b_spline.coefficients.shape))
# actuating_displacement_coefficients_deformed_side = csdl.linear_combination(structural_displacements_b_spline.coefficients,
#                                                       zero_displacements,
#                                                     num_steps=len(new_deformed_fishy_weights),
#                                                    start_weights=new_deformed_fishy_weights, 
#                                                    stop_weights=1-new_deformed_fishy_weights)
# actuating_displacement_coefficients_opposite_deformed_side = csdl.linear_combination(opposite_structural_displacements_b_spline.coefficients,
#                                                       zero_displacements,
#                                                     num_steps=len(new_opposite_deformed_fishy_weights),
#                                                    start_weights=new_opposite_deformed_fishy_weights, 
#                                                    stop_weights=1-new_opposite_deformed_fishy_weights)


actuating_displacement_coefficients_deformed_side = linear_combination((structural_displacements_b_spline.coefficients, zero_displacements),
                                                    (new_deformed_fishy_weights, 1-new_deformed_fishy_weights))
actuating_displacement_coefficients_opposite_deformed_side = linear_combination((opposite_structural_displacements_b_spline.coefficients,
                                                                                 zero_displacements),
                                                    (new_opposite_deformed_fishy_weights, 1-new_opposite_deformed_fishy_weights))
actuating_displacement_coefficients_deformed_side_module_2 = linear_combination((structural_displacements_b_spline.coefficients, zero_displacements),
                                                (new_deformed_fishy_weights_module_2, 1-new_deformed_fishy_weights_module_2))
actuating_displacement_coefficients_opposite_deformed_side_module_2 = linear_combination((opposite_structural_displacements_b_spline.coefficients,
                                                                                 zero_displacements),
                                                (new_opposite_deformed_fishy_weights_module_2, 1-new_opposite_deformed_fishy_weights_module_2))
actuating_displacement_coefficients_deformed_side_module_3 = linear_combination((structural_displacements_b_spline.coefficients, zero_displacements),
                                                (new_deformed_fishy_weights_module_3, 1-new_deformed_fishy_weights_module_3))
actuating_displacement_coefficients_opposite_deformed_side_module_3 = linear_combination((opposite_structural_displacements_b_spline.coefficients,
                                                                                 zero_displacements),
                                                (new_opposite_deformed_fishy_weights_module_3, 1-new_opposite_deformed_fishy_weights_module_3))


actuating_displacement_coefficients = csdl.Variable(name='actuating_displacement_coefficients', 
                                                   value=np.zeros((time.shape[0],) + structural_displacements_b_spline.coefficients.shape))

actuating_displacement_coefficients = actuating_displacement_coefficients.set(csdl.slice[deformed_side_interpolation_indices],
                                                                                actuating_displacement_coefficients_deformed_side)
actuating_displacement_coefficients = actuating_displacement_coefficients.set(csdl.slice[opposite_deformed_side_interpolation_indices],
                                                                                actuating_displacement_coefficients_opposite_deformed_side)

actuating_displacement_coefficients_module_2 = csdl.Variable(name='actuating_displacement_coefficients_module_2', 
                                                   value=np.zeros((time.shape[0],) + structural_displacements_b_spline.coefficients.shape))

actuating_displacement_coefficients_module_2 = actuating_displacement_coefficients_module_2.set(csdl.slice[deformed_side_interpolation_indices_module_2],
                                                                                actuating_displacement_coefficients_deformed_side_module_2)
actuating_displacement_coefficients_module_2 = actuating_displacement_coefficients_module_2.set(csdl.slice[opposite_deformed_side_interpolation_indices_module_2],
                                                                                actuating_displacement_coefficients_opposite_deformed_side_module_2)

actuating_displacement_coefficients_module_3 = csdl.Variable(name='actuating_displacement_coefficients_module_3', 
                                                   value=np.zeros((time.shape[0],) + structural_displacements_b_spline.coefficients.shape))

actuating_displacement_coefficients_module_3 = actuating_displacement_coefficients_module_3.set(csdl.slice[deformed_side_interpolation_indices_module_3],
                                                                                actuating_displacement_coefficients_deformed_side_module_3)
actuating_displacement_coefficients_module_3 = actuating_displacement_coefficients_module_3.set(csdl.slice[opposite_deformed_side_interpolation_indices_module_3],
                                                                                actuating_displacement_coefficients_opposite_deformed_side_module_3)

# add displacements directly to coefficients (more efficient because less fitting but questionable accuracy)

# Project geometry coefficients onto geometry to get parametric coordinates (should be about evenly spaced)
# NOTE: This is a bit silly because we already have the parametric coordinates, but I don't want to figure it out.
coefficients_parametric = fishy.project(points=fishy.coefficients, plot=False, grid_search_density_parameter=1.1, newton_tolerance=1.e-12, projection_tolerance=1.e-1)
points_in_module_1_indices = np.where((coefficients_parametric[:,0] <= module_1_max_u) & (coefficients_parametric[:,0] >= module_1_min_u))[0]
points_in_front_of_module_1_indices = np.where(coefficients_parametric[:,0] > module_1_max_u)[0]
points_behind_module_1_indices = np.where(coefficients_parametric[:,0] < module_1_min_u)[0]
points_in_module_2_indices = np.where((coefficients_parametric[:,0] <= module_2_max_u) & (coefficients_parametric[:,0] >= module_2_min_u))[0]
points_in_front_of_module_2_indices = np.where(coefficients_parametric[:,0] > module_2_max_u)[0]
points_behind_module_2_indices = np.where(coefficients_parametric[:,0] < module_2_min_u)[0]
points_in_module_3_indices = np.where((coefficients_parametric[:,0] <= module_3_max_u) & (coefficients_parametric[:,0] >= module_3_min_u))[0]
points_in_front_of_module_3_indices = np.where(coefficients_parametric[:,0] > module_3_max_u)[0]
points_behind_module_3_indices = np.where(coefficients_parametric[:,0] < module_3_min_u)[0]

module_1_displacements_b_spline_at_time_t = structural_displacements_b_spline.copy()
module_2_displacements_b_spline_at_time_t = structural_displacements_b_spline.copy()
module_3_displacements_b_spline_at_time_t = structural_displacements_b_spline.copy()

# region Construct dynamic geometry
fishy_configurations = []
for i in range(time.shape[0]):
# for i in csdl.frange(time.shape[0]):
    fishy_configuration = fishy.copy()

    # region Evaluate Displacements at Time t
    # region Module 1
    module_1_displacements_b_spline_at_time_t.coefficients = actuating_displacement_coefficients[i]

    # # displacements for in front of module are 0, so don't have deformation (REPLACING THIS SO THIS IS NOT TRUE)
    # coefficients_parametric_in_front_of_module_1 = coefficients_parametric[points_in_front_of_module_1_indices,:]
    # # edge_parametric_coordinates = np.array([0., coefficients_parametric_in_front_of_module_1[0,1], coefficients_parametric_in_front_of_module_1[0,2]])
    # edge_parametric_coordinates_u = np.ones((coefficients_parametric_in_front_of_module_1.shape[0],1))
    # edge_parametric_coordinates_v = coefficients_parametric_in_front_of_module_1[:,1].reshape((-1,1))
    # edge_parametric_coordinates_w = coefficients_parametric_in_front_of_module_1[:,2].reshape((-1,1))
    # edge_parametric_coordinates = np.hstack((edge_parametric_coordinates_u, edge_parametric_coordinates_v, edge_parametric_coordinates_w))
    # derivative_values_at_edge = module_1_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates, parametric_derivative_orders=(1,0,0))
    # # derivative_values_at_edge = module_1_displacements_b_spline_at_time_t.evaluate(np.array([1., 0.5, 0.5]), parametric_derivative_orders=(1,0,0))
    # displacement_values_at_edge = module_1_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates)
    # delta_u = (coefficients_parametric_in_front_of_module_1[:,0] - module_1_max_u)/(module_1_max_u - module_1_min_u)  # need to rescale delta_u or derivative due to parametric space mismatch
    # delta_u = np.repeat(delta_u, 3).reshape((-1,3))   # repeat each one 3 times to scale the x,y,z
    # # displacements_at_coefficients = displacement_values_at_edge + csdl.expand(derivative_values_at_edge, delta_u.shape, 'j->ij')*delta_u
    # displacements_at_coefficients = displacement_values_at_edge + derivative_values_at_edge*delta_u
    # # indices = coefficient_indices[points_in_front_of_module_1_indices].reshape((-1))
    # # deformed_fishy.coefficients[indices] = fishy.coefficients[indices] + displacements_at_coefficients
    # fishy_configuration.coefficients = fishy_configuration.coefficients.set(csdl.slice[list(points_in_front_of_module_1_indices)],
    #                                                                 fishy_configuration.coefficients[list(points_in_front_of_module_1_indices)] \
    #                                                                     + displacements_at_coefficients)
    

    # for coefficients in module, evaluate displacements at corresponding location and add them to the current coefficients
    coefficients_parametric_in_module_1 = coefficients_parametric[points_in_module_1_indices,:]
    coefficients_parametric_in_module_1[:,0] = (coefficients_parametric_in_module_1[:,0] - module_1_min_u)/(module_1_max_u - module_1_min_u)
    deformations_in_module = module_1_displacements_b_spline_at_time_t.evaluate(coefficients_parametric_in_module_1)
    # NOTE: For line above, a temporary structural displacement B-spline will have to be created for each time step. (where it's coefficients change)
    # indices = coefficient_indices[points_in_module_1_indices].reshape((-1))
    # fishy_configuration.coefficients[indices] = fishy.coefficients[indices] + deformations_in_module
    fishy_configuration.coefficients = fishy_configuration.coefficients.set(csdl.slice[list(points_in_module_1_indices)],
                                                                    fishy_configuration.coefficients[list(points_in_module_1_indices)] + deformations_in_module)

    # For coefficients behind the module, evaluate the derivative at the end of the module and apply the rigid link
    coefficients_parametric_behind_module_1 = coefficients_parametric[points_behind_module_1_indices,:]
    # edge_parametric_coordinates = np.array([0., coefficients_parametric_behind_module_1[0,1], coefficients_parametric_behind_module_1[0,2]])
    edge_parametric_coordinates_u = np.zeros((coefficients_parametric_behind_module_1.shape[0],1))
    edge_parametric_coordinates_v = coefficients_parametric_behind_module_1[:,1].reshape((-1,1))
    edge_parametric_coordinates_w = coefficients_parametric_behind_module_1[:,2].reshape((-1,1))
    edge_parametric_coordinates = np.hstack((edge_parametric_coordinates_u, edge_parametric_coordinates_v, edge_parametric_coordinates_w))
    derivative_values_at_edge = module_1_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates, parametric_derivative_orders=(1,0,0))
    # derivative_values_at_edge = module_1_displacements_b_spline_at_time_t.evaluate(np.array([0., 0.5, 0.5]), parametric_derivative_orders=(1,0,0))
    displacement_values_at_edge = module_1_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates)
    delta_u = (coefficients_parametric_behind_module_1[:,0] - module_1_min_u)/(module_1_max_u - module_1_min_u)  # need to rescale delta_u or derivative due to parametric space mismatch
    delta_u = np.repeat(delta_u, 3).reshape((-1,3))   # repeat each one 3 times to scale the x,y,z
    displacements_at_coefficients = displacement_values_at_edge + derivative_values_at_edge*delta_u
    # indices = coefficient_indices[points_behind_module_1_indices].reshape((-1))
    # fishy_configuration.coefficients[indices] = fishy.coefficients[indices] + displacements_at_coefficients
    fishy_configuration.coefficients = fishy_configuration.coefficients.set(csdl.slice[list(points_behind_module_1_indices)],
                                                                    fishy_configuration.coefficients[list(points_behind_module_1_indices)] \
                                                                        + displacements_at_coefficients)

    # endregion Module 1

    # region Module 2
    time_step_offset = 1*num_panel_method_time_steps//(num_cycles*num_modules)   # (divide by 2 to get one period, then phase offset by 1/3 of period)
    # if i >= time_step_offset:
    # module_displacement_coefficients = actuating_displacement_coefficients[i - time_step_offset,:]
    module_displacement_coefficients = actuating_displacement_coefficients_module_2[i,:]
    module_2_displacements_b_spline_at_time_t.coefficients = module_displacement_coefficients

    # # displacements for in front of module are 0, so don't have deformation (REPLACING THIS SO THIS IS NOT TRUE)
    # coefficients_parametric_in_front_of_module = coefficients_parametric[points_in_front_of_module_2_indices,:]
    # # edge_parametric_coordinates = np.array([0., coefficients_parametric_in_front_of_module[0,1], coefficients_parametric_in_front_of_module[0,2]])
    # edge_parametric_coordinates_u = np.ones((coefficients_parametric_in_front_of_module.shape[0],1))
    # edge_parametric_coordinates_v = coefficients_parametric_in_front_of_module[:,1].reshape((-1,1))
    # edge_parametric_coordinates_w = coefficients_parametric_in_front_of_module[:,2].reshape((-1,1))
    # edge_parametric_coordinates = np.hstack((edge_parametric_coordinates_u, edge_parametric_coordinates_v, edge_parametric_coordinates_w))
    # derivative_values_at_edge = module_2_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates, parametric_derivative_orders=(1,0,0))
    # displacement_values_at_edge = module_2_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates)
    # delta_u = (coefficients_parametric_in_front_of_module[:,0] - module_2_max_u)/(module_2_max_u - module_2_min_u)  # need to rescale delta_u or derivative due to parametric space mismatch
    # delta_u = np.repeat(delta_u, 3).reshape((-1,3))   # repeat each one 3 times to scale the x,y,z
    # displacements_at_coefficients = displacement_values_at_edge + derivative_values_at_edge*delta_u
    # # indices = coefficient_indices[points_in_front_of_module_2_indices].reshape((-1))
    # # fishy_configuration.coefficients[indices] = fishy.coefficients[indices] + displacements_at_coefficients
    # fishy_configuration.coefficients = fishy_configuration.coefficients.set(csdl.slice[list(points_in_front_of_module_2_indices)],
    #                                                                 fishy_configuration.coefficients[list(points_in_front_of_module_2_indices)] \
    #                                                                     + displacements_at_coefficients)
    

    # for coefficients in module, evaluate displacements at corresponding location and add them to the current coefficients
    coefficients_parametric_in_module = coefficients_parametric[points_in_module_2_indices,:]
    coefficients_parametric_in_module[:,0] = (coefficients_parametric_in_module[:,0] - module_2_min_u)/(module_2_max_u - module_2_min_u)
    deformations_in_module = module_2_displacements_b_spline_at_time_t.evaluate(coefficients_parametric_in_module)
    # NOTE: For line above, a temporary structural displacement B-spline will have to be created for each time step. (where it's coefficients change)
    # indices = coefficient_indices[points_in_module_2_indices].reshape((-1))
    # fishy_configuration.coefficients[indices] = fishy.coefficients[indices] + deformations_in_module
    fishy_configuration.coefficients = fishy_configuration.coefficients.set(csdl.slice[list(points_in_module_2_indices)],
                                                                    fishy_configuration.coefficients[list(points_in_module_2_indices)] + deformations_in_module)

    # For coefficients behind the module, evaluate the derivative at the end of the module and apply the rigid link
    coefficients_parametric_behind_module = coefficients_parametric[points_behind_module_2_indices,:]
    # edge_parametric_coordinates = np.array([0., coefficients_parametric_behind_module[0,1], coefficients_parametric_behind_module[0,2]])
    edge_parametric_coordinates_u = np.zeros((coefficients_parametric_behind_module.shape[0],1))
    edge_parametric_coordinates_v = coefficients_parametric_behind_module[:,1].reshape((-1,1))
    edge_parametric_coordinates_w = coefficients_parametric_behind_module[:,2].reshape((-1,1))
    edge_parametric_coordinates = np.hstack((edge_parametric_coordinates_u, edge_parametric_coordinates_v, edge_parametric_coordinates_w))
    derivative_values_at_edge = module_2_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates, parametric_derivative_orders=(1,0,0))
    displacement_values_at_edge = module_2_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates)
    delta_u = (coefficients_parametric_behind_module[:,0] - module_2_min_u)/(module_2_max_u - module_2_min_u)  # need to rescale delta_u or derivative due to parametric space mismatch
    delta_u = np.repeat(delta_u, 3).reshape((-1,3))   # repeat each one 3 times to scale the x,y,z
    displacements_at_coefficients = displacement_values_at_edge + derivative_values_at_edge*delta_u
    # indices = coefficient_indices[points_behind_module_2_indices].reshape((-1))
    # fishy_configuration.coefficients[indices] = fishy.coefficients[indices] + displacements_at_coefficients
    fishy_configuration.coefficients = fishy_configuration.coefficients.set(csdl.slice[list(points_behind_module_2_indices)],
                                                                    fishy_configuration.coefficients[list(points_behind_module_2_indices)] \
                                                                        + displacements_at_coefficients)
    # endregion Module 2

    # region Module 3
    time_step_offset = 2*(num_cycles*num_modules)   # (divide by 2 to get one period, then phase offset by 2/3 of period)
    # if i >= time_step_offset:
    # module_displacement_coefficients = actuating_displacement_coefficients[i - time_step_offset,:]
    module_displacement_coefficients = actuating_displacement_coefficients_module_3[i,:]
    module_3_displacements_b_spline_at_time_t.coefficients = module_displacement_coefficients

    # # displacements for in front of module are 0, so don't have deformation (REPLACING THIS SO THIS IS NOT TRUE)
    # coefficients_parametric_in_front_of_module = coefficients_parametric[points_in_front_of_module_3_indices,:]
    # # edge_parametric_coordinates = np.array([0., coefficients_parametric_in_front_of_module[0,1], coefficients_parametric_in_front_of_module[0,2]])
    # edge_parametric_coordinates_u = np.ones((coefficients_parametric_in_front_of_module.shape[0],1))
    # edge_parametric_coordinates_v = coefficients_parametric_in_front_of_module[:,1].reshape((-1,1))
    # edge_parametric_coordinates_w = coefficients_parametric_in_front_of_module[:,2].reshape((-1,1))
    # edge_parametric_coordinates = np.hstack((edge_parametric_coordinates_u, edge_parametric_coordinates_v, edge_parametric_coordinates_w))
    # derivative_values_at_edge = module_3_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates, parametric_derivative_orders=(1,0,0))
    # displacement_values_at_edge = module_3_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates)
    # delta_u = (coefficients_parametric_in_front_of_module[:,0] - module_3_max_u)/(module_3_max_u - module_3_min_u)  # need to rescale delta_u or derivative due to parametric space mismatch
    # delta_u = np.repeat(delta_u, 3).reshape((-1,3))   # repeat each one 3 times to scale the x,y,z
    # displacements_at_coefficients = displacement_values_at_edge + derivative_values_at_edge*delta_u
    # # indices = coefficient_indices[points_in_front_of_module_3_indices].reshape((-1))
    # # fishy_configuration.coefficients[indices] = fishy.coefficients[indices] + displacements_at_coefficients
    # fishy_configuration.coefficients = fishy_configuration.coefficients.set(csdl.slice[list(points_in_front_of_module_3_indices)],
    #                                                                 fishy_configuration.coefficients[list(points_in_front_of_module_3_indices)] \
    #                                                                     + displacements_at_coefficients)
    

    # for coefficients in module, evaluate displacements at corresponding location and add them to the current coefficients
    coefficients_parametric_in_module = coefficients_parametric[points_in_module_3_indices,:]
    coefficients_parametric_in_module[:,0] = (coefficients_parametric_in_module[:,0] - module_3_min_u)/(module_3_max_u - module_3_min_u)
    deformations_in_module = module_3_displacements_b_spline_at_time_t.evaluate(coefficients_parametric_in_module)
    # NOTE: For line above, a temporary structural displacement B-spline will have to be created for each time step. (where it's coefficients change)
    # indices = coefficient_indices[points_in_module_3_indices].reshape((-1))
    # fishy_configuration.coefficients[indices] = fishy.coefficients[indices] + deformations_in_module
    fishy_configuration.coefficients = fishy_configuration.coefficients.set(csdl.slice[list(points_in_module_3_indices)],
                                                                    fishy_configuration.coefficients[list(points_in_module_3_indices)] \
                                                                        + deformations_in_module)

    # For coefficients behind the module, evaluate the derivative at the end of the module and apply the rigid link
    coefficients_parametric_behind_module = coefficients_parametric[points_behind_module_3_indices,:]
    # edge_parametric_coordinates = np.array([0., coefficients_parametric_behind_module[0,1], coefficients_parametric_behind_module[0,2]])
    edge_parametric_coordinates_u = np.zeros((coefficients_parametric_behind_module.shape[0],1))
    edge_parametric_coordinates_v = coefficients_parametric_behind_module[:,1].reshape((-1,1))
    edge_parametric_coordinates_w = coefficients_parametric_behind_module[:,2].reshape((-1,1))
    edge_parametric_coordinates = np.hstack((edge_parametric_coordinates_u, edge_parametric_coordinates_v, edge_parametric_coordinates_w))
    derivative_values_at_edge = module_3_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates, parametric_derivative_orders=(1,0,0))
    displacement_values_at_edge = module_3_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates)
    delta_u = (coefficients_parametric_behind_module[:,0] - module_3_min_u)/(module_3_max_u - module_3_min_u)  # need to rescale delta_u or derivative due to parametric space mismatch
    delta_u = np.repeat(delta_u, 3).reshape((-1,3))   # repeat each one 3 times to scale the x,y,z
    displacements_at_coefficients = displacement_values_at_edge + derivative_values_at_edge*delta_u
    # indices = coefficient_indices[points_behind_module_3_indices].reshape((-1))
    # fishy_configuration.coefficients[indices] = fishy.coefficients[indices] + displacements_at_coefficients
    fishy_configuration.coefficients = fishy_configuration.coefficients.set(csdl.slice[list(points_behind_module_3_indices)],
                                                                    fishy_configuration.coefficients[list(points_behind_module_3_indices)] \
                                                                        + displacements_at_coefficients)
    # endregion Module 3

    # endregion Evaluate Displacements at Time t

    fishy_configurations.append(fishy_configuration)        # Can't be doing this in a csdl for loop
# endregion Construct dynamic geometry

# region Evaluate Vortex Panel Method Mesh
# Preallocate panel mesh
panel_mesh = csdl.Variable(value=np.zeros((num_panel_method_time_steps, num_chordwise, num_spanwise, 3)), name='panel_mesh')

for i in range(time.shape[0]):
    fishy_configuration = fishy_configurations[i]
    panel_mesh_this_timestep = fishy_configuration.evaluate(panel_method_parametric_mesh, plot=False)
    panel_mesh = panel_mesh.set(csdl.slice[i], panel_mesh_this_timestep)

def diag_deriv(output, input):
    num = num_chordwise*num_spanwise*3
    jac_diags = csdl.Variable(value = np.zeros((num_panel_method_time_steps, num)))
    jac_zeros = csdl.Variable(value = np.zeros((num_panel_method_time_steps, num)))
    for i in csdl.frange(num):
        cot = csdl.src.operations.derivatives.reverse.vjp(
            [(output, jac_zeros.set(csdl.slice[:,i], np.ones((num_panel_method_time_steps))).reshape((num_panel_method_time_steps, num_chordwise, num_spanwise, 3)))],
            input,
            recorder.get_root_graph(),
        )[input]
        jac_diags = jac_diags.set(csdl.slice[:,i], cot)

    return jac_diags.reshape(output.shape)


# NOTE: MESH VELOCITIES HERE
panel_mesh_velocities = diag_deriv(panel_mesh, time)


pickle.dump(panel_mesh.value, open('examples/advanced_examples/robotic_fish/temp/panel_mesh.pickle', 'wb'))
# exit()
pickle.dump(panel_mesh_velocities.value, open('examples/advanced_examples/robotic_fish/temp/panel_mesh_velocities.pickle', 'wb'))
exit()
# endregion Evaluate Vortex Panel Method Mesh
# endregion Construct Dynamic Geometry and Dynamic Panel Method Mesh

# # region Plot actuating geometry and fluid mesh
# plotting_box = ffd_block.copy()
# plotting_box.coefficients = plotting_box.coefficients.set(csdl.slice[:,:,:,2], plotting_box.coefficients[:,:,:,2]*4.5)
# # plotting_box.coefficients = plotting_box.coefficients.set(csdl.slice[:,:,:,2], plotting_box.coefficients[:,:,:,2]*15.)
# # plotting_box.coefficients = plotting_box.coefficients.set(csdl.slice[:,:,:,0], plotting_box.coefficients[:,:,:,0]*2.5)
# # plotting_box.coefficients = plotting_box.coefficients.set(csdl.slice[:,:,:,1], plotting_box.coefficients[:,:,:,1]*10.)
# plotting_box = plotting_box.plot(show=False, opacity=0.)

# fishy_plot = fishy.plot(show=False, opacity=0.3)
# video = vedo.Video('examples/advanced_examples/robotic_fish/temp/fishy_wiggle.mp4', fps=num_steps_per_cycle/2, backend='cv')
# for i in range(time.shape[0]):
#     plotting_elements = []
#     plotting_elements.append(plotting_box)
#     fishy_configuration_plot = fishy_configurations[i].plot(opacity=0.5, show=False)
#     plotting_elements.append(fishy_configuration_plot)
#     plotter = vedo.Plotter(offscreen=True)
#     # plotter.show([plotting_box, fishy_plot, fishy_configuration_plot], axes=1, viewup='y')
#     vertices = []
#     faces = []
#     for u_index in range(panel_mesh.value[i].shape[0]):
#         for v_index in range(panel_mesh.value[i].shape[1]):
#             vertex = tuple(panel_mesh.value[i,u_index,v_index,:])
#             vertices.append(vertex)
#             if u_index != 0 and v_index != 0:
#                 face = tuple((
#                     (u_index-1)*panel_mesh.value[i].shape[1]+(v_index-1),
#                     (u_index-1)*panel_mesh.value[i].shape[1]+(v_index),
#                     (u_index)*panel_mesh.value[i].shape[1]+(v_index),
#                     (u_index)*panel_mesh.value[i].shape[1]+(v_index-1),
#                 ))
#                 faces.append(face)

#     vedo_mesh = vedo.Mesh([vertices, faces]).wireframe().linewidth(2).color('#F5F0E6')
#     plotting_elements.append(vedo_mesh)

#     # for j in range(panel_mesh.value[i].shape[0]):
#     #     for k in range(panel_mesh.value[i].shape[1]):
#     #         arrow = vedo.Arrow(tuple(panel_mesh.value[i,j,k,:]), tuple(panel_mesh.value[i,j,k,:] + panel_mesh_velocities.value[i,j,k,:]*0.1), s=0.0001).opacity(0.5)
#     #         # arrow = vedo.Arrow(tuple(panel_mesh.value[i,j,k,:]), tuple(panel_mesh.value[i,j,k,:] + panel_mesh_velocities.value[i,j,k,:]), s=0.0001).opacity(0.5)
#     #         plotting_elements.append(arrow)
#     # plotter.show([plotting_box, fishy_configuration_plot, vedo_mesh], axes=1, viewup='y')
#     plotter.show(plotting_elements, axes=1, viewup='y')
#     # plotter.show(plotting_elements, axes=0, viewup='y')
#     video.add_frame()
# video.close()
# # endregion Plot actuating geometry and fluid mesh

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

# region Fluid Solver
# swim_speed = csdl.Variable(value=0.25)
swim_speed = csdl.Variable(value=length.value*0.66)
# swim_speed = csdl.Variable(value=25.)
num_nodes = 1
dt = panel_method_dt
panel_mesh = panel_mesh.reshape((num_nodes,) + panel_mesh.shape) # (nn, nt, nc, ns, 3)
panel_mesh_velocities = (panel_mesh[:,1:] - panel_mesh[:,:-1])/dt
panel_mesh = (panel_mesh[:,1:] + panel_mesh[:,:-1])/2 # kind of like a midpoint rule I guess
nt = panel_mesh.shape[1]
# panel_mesh_velocities = panel_mesh_velocities.reshape((num_nodes,) + panel_mesh_velocities.shape)

# We use the computed fish velocities for the collocation point velocities
# NOTE: we want collocation velocities at the panel centers; we get this by averaging the velocities as such
coll_vel = (panel_mesh_velocities[:,:,:-1,:-1,:] + panel_mesh_velocities[:,:,1:,:-1,:] + panel_mesh_velocities[:,:,1:,1:,:] + panel_mesh_velocities[:,:,:-1,1:,:])/4.

# here we set up the free-stream velocity grid for each panel_MESH NODE
# panel_mesh_free_stream = np.zeros_like(panel_mesh_velocities)
# panel_mesh_free_stream[:,:,:,:,0] = cruise_velocity
panel_mesh_free_stream = csdl.Variable(shape=panel_mesh_velocities.shape, value=0.)
panel_mesh_free_stream = panel_mesh_free_stream.set(csdl.slice[:,:,:,:,0], swim_speed)

panel_mesh_list = [panel_mesh]
panel_mesh_velocities_list = [panel_mesh_free_stream]
coll_vel_list = [coll_vel]
output_dict, panel_mesh_dict, wake_panel_mesh_dict, mu, sigma, mu_wake = unsteady_panel_solver(
    panel_mesh_list, 
    panel_mesh_velocities_list, 
    coll_vel_list,
    dt=dt, 
    free_wake=False
)

num_boundary_layer_per_panel = 2
boundary_layer_mesh = csdl.Variable(value=np.zeros((num_nodes, nt, int(num_boundary_layer_per_panel*panel_mesh.shape[2]-1), panel_mesh.shape[3], 3)))
boundary_layer_mesh = boundary_layer_mesh.set(csdl.slice[:,:,::2,:,:], value=panel_mesh)
boundary_layer_mesh = boundary_layer_mesh.set(csdl.slice[:,:,1::2,:,:], value=(panel_mesh[:,:,1:,:,:] + panel_mesh[:,:,:-1,:,:])/2)

panel_forces = fish_post_processor(panel_mesh_dict, output_dict, boundary_layer_mesh, mu)
net_thrust_drag = csdl.sum(panel_forces[0,:-1,:,:,0])

# water_density = 1000
# panel_forces = output_dict['surface_0']['panel_forces']*water_density
# panel_forces = panel_forces[0]  # Get rid of the num_nodes dimension
# panel_forces_x = panel_forces[:-1,:,:,0]
# # net_thrust_drag = csdl.sum(panel_forces_x)
# # net_thrust_drag = csdl.sum(panel_forces_x, axes=(1,2))
# net_thrust = csdl.sum(panel_forces_x)
# # skin_friction_coefficient = csdl.Variable(value=0.02067545877401697)
# # skin_friction_coefficient = csdl.Variable(value=2.2135804402199244)
# skin_friction_coefficient = csdl.Variable(value=1.4972332801562531)

# skin_friction = 1/2*water_density*swim_speed**2*skin_friction_coefficient*height*length
# net_thrust_drag = net_thrust - skin_friction
print(net_thrust_drag.value)
# print(np.sum(net_thrust_drag.value))
# import matplotlib.pyplot as plt
# plt.plot(net_thrust_drag.value)
# plt.show()

# endregion Fluid Solver

# region Objective Model
actuation_power = applied_work/actuation_period
cost_of_transport = actuation_power/swim_speed
objective = -swim_speed
# objective = -0.2*swim_speed + 0.9*cost_of_transport
# objective = cost_of_transport
# objective = cost_of_transport# + net_thrust_drag**2
print('swim speed: ', swim_speed.value)
print('cost of transport: ', cost_of_transport.value)
print('net force: ', net_thrust_drag.value)
# exit()
# endregion Objective Model

# region Additional Cosntraints
# width_manufacturing_constraint = computed_fishy_width - 0.012*2   # this width is only being computed at the middle, which is where we want this constraint
# endregion Additional Constraints


# # region Optimization
# swim_speed.set_as_design_variable(lower=0.1, upper=100., scaler=1.e1)
actuation_frequency.set_as_design_variable(lower=0.2, upper=1., scaler=1.)
pump_pressure.set_as_design_variable(lower=1.e3, upper=4.5e4, scaler=1e-4)
# # length.set_as_design_variable(lower=0.3, upper=1., scaler=1.e1)
# length.set_as_design_variable(lower=length.value*0.8, upper=length.value*1.2, scaler=1.1e1)
# # width.set_as_design_variable(lower=0.02, upper=0.08, scaler=1.e2)
# width_shape_variable.set_as_design_variable(lower=-computed_fishy_width.value*0.4, upper=computed_fishy_width.value*0.4, scaler=1.e2)
# width_shape_variable.set_as_design_variable(lower=-(computed_fishy_width.value*0.4)/2, upper=-(computed_fishy_width.value*0.2)/2, scaler=1.e2)
width_shape_variables.set_as_design_variable(lower=-(computed_fishy_width.value*0.4)/2, upper=(computed_fishy_width.value*0.4)/2, scaler=1.e2)
# # height.set_as_design_variable(lower=0.03, upper=0.12, scaler=1.e2)
# height.set_as_design_variable(lower=height.value*0.8, upper=height.value*1.2, scaler=1.e2)
height.set_as_design_variable(lower=height.value*0.8, upper=height.value*2., scaler=1.e2)
net_thrust_drag.set_as_constraint(equals=0, scaler=1.e2)
frequency_pressure_bound.set_as_constraint(upper=0., scaler=1e-4)
# width_manufacturing_constraint.set_as_constraint(lower=0., scaler=1.e2)
# NOTE: For this constraint, likely want to use C_t to help with scaling

objective.set_as_objective(scaler=1.e1)
# objective.set_as_objective(scaler=1.e0)

# exit()
# sim = csdl.experimental.PySimulator(recorder=recorder)
# additional_inputs = [length, width_shape_variable, height, actuation_frequency, swim_speed]
additional_inputs = [width_shape_variables, height, actuation_frequency, swim_speed]
# additional_inputs = []
additional_outputs = [objective, net_thrust_drag, cost_of_transport]

sim = csdl.experimental.JaxSimulator(
    recorder = recorder,
    additional_inputs=additional_inputs,
    additional_outputs=additional_outputs,
    gpu=False
)
# sim.check_totals()
# import time
# t1 = time.time()
# sim.run()
# t2 = time.time()
# print('compile and run time: ', t2 - t1)
# print('constraint value', net_thrust_drag.value)
# print('objective value', objective.value)
# sim.run()
# t3 = time.time()
# print('run time: ', t3 - t2)
# # # exit()

# # widths = np.linspace(0.04, 0.02, 5)
# width_shape_variable_sweep = np.linspace(0., -0.01, 5)
# net_thrust_drags = []
# cost_of_transports = []
# for i in range(len(width_shape_variable_sweep)):
#     # width.value = width_shape_variables[i]
#     sim[width_shape_variables] = width_shape_variable_sweep[i]
#     sim.run()
#     print('width delta: ', width_shape_variable_sweep[i])
#     print('constraint value', net_thrust_drag.value)
#     print('objective value', objective.value)
#     print('swim speed: ', swim_speed.value)
#     print('cost of transport: ', cost_of_transport.value)
#     net_thrust_drags.append(net_thrust_drag.value)
#     cost_of_transports.append(cost_of_transport.value)
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot(width_shape_variable_sweep, np.array(net_thrust_drags))
# plt.figure(2)
# plt.plot(width_shape_variable_sweep, np.array(cost_of_transports))
# plt.show()
# exit()

import time
t1 = time.time()
optimization_problem = CSDLAlphaProblem(problem_name='fishy_optimization', simulator=sim)
t2 = time.time()
print('compile time?: ', t2 - t1)
optimizer = PySLSQP(optimization_problem, solver_options={'maxiter':100, 'acc':1.e-4}, readable_outputs=['x'])

t3 = time.time()
print('optimizer setup time: ', t3 - t2)
optimizer.solve()
t4 = time.time()
print('solve time: ', t4 - t3)
optimizer.print_results()

print('swim speed: ', swim_speed.value)
print('cost of transport: ', cost_of_transport.value)

# initial_objective_value = objective.value
# initial_length = length.value
# initial_width = width.value
# initial_height = height.value
# initial_surface_area = surface_area.value
# initial_angle = angle.value

# # from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
# # verify_derivatives_inline([csdl.norm(fishy.coefficients)], [length, width, height], 1.e-6, raise_on_error=False)
# # verify_derivatives_inline([structural_mesh_displacements[list(np.arange(0, structural_displacements_flattened.size, 100))]], [length, width, height], 1.e-6, raise_on_error=False)
# # verify_derivatives_inline([structural_displacements_flattened[list(np.arange(0, structural_displacements_flattened.size, 100))]], [length, width, height], 1.e-6, raise_on_error=False)
# # verify_derivatives_inline([structural_displacements_flattened[10000]], [length, width, height], 1.e-6, raise_on_error=False)
# # verify_derivatives_inline([deformed_module.coefficients], [length, width, height], 1.e-6, raise_on_error=False)
# # verify_derivatives_inline([derivative_at_module_edge1], [length, width, height], 1.e-6, raise_on_error=False)
# # verify_derivatives_inline([derivative_at_module_edge], [length, width, height], 1.e-6, raise_on_error=False)
# # verify_derivatives_inline([objective], [length, width, height], 1.e-6, raise_on_error=False)
# # optimizer.check_first_derivatives(optimization_problem.x0)

# # exit()

# # d_objective_d_length = csdl.derivative(objective, length)
# # d_objective_d_width = csdl.derivative(objective, width)
# # d_objective_d_height = csdl.derivative(objective, height)

# # print('d_objective_d_length', d_objective_d_length.value)
# # print('d_objective_d_width', d_objective_d_width.value)
# # print('d_objective_d_height', d_objective_d_height.value)
# # exit()

# # recorder.execute()


# # video = vedo.Video('fishy_width_sweep_fixed_midpoint.mp4', duration=5, backend='cv')
# # width_values = np.linspace(0.015, 0.05, 5)
# # direction_values = np.zeros((len(width_values), 3))
# # old_direction_values = np.zeros((len(width_values), 3))
# # objective_values = np.zeros_like(width_values)
# # old_objective_values = np.zeros_like(width_values)
# # for i, width_value in enumerate(width_values):
# #     print(i)
# #     width.value = width_value
# #     recorder.execute()
# #     objective_values[i] = objective.value
# #     old_objective_values[i] = -old_angle.value
# #     direction_values[i] = derivative_at_module_edge.value
# #     old_direction_values[i] = derivative_at_module_edge_old.value

# #     fishy_plot = fishy.plot(show=False, opacity=0.3)
# #     vedo_mesh = vedo.Mesh([displaced_mesh.value, structural_elements]).wireframe()
# #     arrow = vedo.Arrow(tuple(module_edge.value.reshape((-1,))), 
# #                                    tuple((module_edge.value - derivative_at_module_edge.value/10).reshape((-1,))), s=0.0005)
# #     plotter = vedo.Plotter(offscreen=True)
# #     plotter.show([fishy_plot, vedo_mesh, arrow], axes=1, viewup='y')
# #     video.add_frame()

# # video.close()

# # print(width_values)
# # print(objective_values)
# # print(direction_values)
# # print(old_direction_values)
# # import matplotlib.pyplot as plt
# # plt.plot(width_values, -objective_values, label='Angle')
# # plt.title('Angle vs Width')
# # plt.xlabel('Width')
# # plt.ylabel('Angle')
# # # plt.plot(width_values, old_objective_values, label='Old Objective')
# # plt.legend()
# # plt.show()
# # exit()

# optimizer.solve()
# optimizer.print_results()


# print('Initial Objective: ', initial_objective_value)
# print('Initial Length', initial_length)
# print('Initial Width', initial_width)
# print('Initial Height', initial_height)
# print('Initial Surface Area', initial_surface_area)
# print('Initial Angle: ', initial_angle)

# print('Optimized Objective: ', objective.value)
# print('Optimized Length', length.value)
# print('Optimized Width', width.value)
# print('Optimized Height', height.value)
# print('Optimized Surface Area', surface_area.value)
# print('Optimized Angle: ', angle.value)

# print('Percent Change in Objective', (objective.value - initial_objective_value)/initial_objective_value*100)
# print("Percent Change in Length: ", (length.value - initial_length)/initial_length*100)
# print("Percent Change in Width: ", (width.value - initial_width)/initial_width*100)
# print("Percent Change in Height: ", (height.value - initial_height)/initial_height*100)

# # from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
# # verify_derivatives_inline([csdl.norm(fishy.coefficients)], [length, width, height], 1.e-6, raise_on_error=False)
# # verify_derivatives_inline([csdl.norm(structural_displacements_flattened)], [length, width, height], 1.e-6, raise_on_error=False)
# # verify_derivatives_inline([csdl.norm(structural_displacements_b_spline.coefficients)], [length, width, height], 1.e-6, raise_on_error=False)
# # verify_derivatives_inline([csdl.norm(deformations_in_module)], [length, width, height], 1.e-6, raise_on_error=False)
# # verify_derivatives_inline([csdl.norm(deformed_fishy.coefficients)], [length, width, height], 1.e-6, raise_on_error=False)
# # optimizer.check_first_derivatives(optimization_problem.x0)

# # Plot structural solver result
# fishy_plot = fishy.plot(show=False, opacity=0.3)
# vedo_mesh_initial = vedo.Mesh([initial_displaced_mesh, structural_elements]).wireframe().color('yellow').opacity(0.4)
# vedo_mesh = vedo.Mesh([displaced_mesh.value, structural_elements]).wireframe().color('green').opacity(0.8)
# plotter = vedo.Plotter()
# plotter.show([fishy_plot, vedo_mesh_initial, vedo_mesh], axes=1, viewup='y')

# # endregion Optimization

