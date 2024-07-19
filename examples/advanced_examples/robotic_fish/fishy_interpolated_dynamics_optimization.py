import csdl_alpha as csdl
import lsdo_geo
import lsdo_function_spaces as lfs
import lsdo_soft_csm
import pickle
import meshio
import numpy as np
from modopt import SLSQP
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
        fishy = pickle.load(handle)
        fishy.coefficients = csdl.Variable(value=fishy.coefficients.value, name='fishy_coefficients')   # Remake variables because can't pickle variables
        return fishy

fishy = import_geometry()
fishy.name = 'fishy'
fishy.coefficients.reshape((51,15,15,3))

# fishy.plot(opacity=0.3)
# region -Structural Mesh Projection
mesh_file_path = "examples/advanced_examples/robotic_fish/meshes/"
# mesh_name = "module_v1_fine"
mesh_name = "module_v1"
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

structural_mesh_parametric = fishy.project(points=structural_mesh_nodes, grid_search_density_parameter=4., plot=False)
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
# side_num_chordwise = 21  # not including front/back contribution
side_num_chordwise = 25  # not including front/back contribution
side_num_spanwise = 7  # not including top/bottom contribution
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

# # endregion Evaluate Vortex Panel Method Mesh

# fishy_plot = fishy.plot(opacity=0.5, show=False)
# plotter = vedo.Plotter(offscreen=False)
# # plotter.show([plotting_box, fishy_plot, fishy_plot], axes=1, viewup='y')
# vertices = []
# faces = []
# for u_index in range(panel_mesh_this_timestep.shape[0]):
#     for v_index in range(panel_mesh_this_timestep.shape[1]):
#         vertex = tuple(panel_mesh_this_timestep.value[u_index,v_index,:])
#         vertices.append(vertex)
#         if u_index != 0 and v_index != 0:
#             face = tuple((
#                 (u_index-1)*num_spanwise+(v_index-1),
#                 (u_index-1)*num_spanwise+(v_index),
#                 (u_index)*num_spanwise+(v_index),
#                 (u_index)*num_spanwise+(v_index-1),
#             ))
#             faces.append(face)

# vedo_mesh = vedo.Mesh([vertices, faces]).wireframe().linewidth(2).color('#F5F0E6')
# plotter.show([fishy_plot, vedo_mesh], axes=1, viewup='y')

# test = fishy.evaluate(panel_method_parametric_mesh, plot=True)
# exit()

# endregion Stich together parametric mesh
# test = fishy.evaluate(parametric_grid_1, plot=True)


# # NOTE: Trying out a different approach of just projecting the mesh on
# # print(np.min(fishy.coefficients.value, axis=0))
# mins = np.min(fishy.coefficients.value, axis=0)
# maxs = np.max(fishy.coefficients.value, axis=0)
# num_spanwise = 7
# num_chordwise_head = 9
# back_edge = np.linspace(np.array([mins[0], -0.001, maxs[2]]), np.array([mins[0], 0.001, maxs[2]]), num_spanwise)
# modules_back_edge = np.linspace(np.array([-0.1 + 0.04, mins[1], maxs[2]]), np.array([-0.1 + 0.04, maxs[1], maxs[2]]), num_spanwise)
# modules_front_edge = np.linspace(np.array([0.08 +  0.08 + 0.04, mins[1], maxs[2]]), np.array([0.08 + 0.08 + 0.04, maxs[1], maxs[2]]), num_spanwise)
# line_over_head_top_edge = np.linspace(np.array([0.2, maxs[1], maxs[2]]), np.array([maxs[0], maxs[1], maxs[2]]), num_chordwise_head)
# top_edge_parametric = fishy.project(points=line_over_head_top_edge, direction=np.array([0., 0., 1.]), grid_search_density_parameter=10, plot=False)
# top_values = fishy.evaluate(top_edge_parametric).value
# line_in_front = np.linspace(np.array([maxs[0], top_values[-1,1], maxs[2]]), np.array([maxs[0], 0., maxs[2]]), num_chordwise_head//3)
# front_edge_parametric = fishy.project(points=line_in_front, direction=np.array([0., 0., 1.]), grid_search_density_parameter=10, plot=False)
# front_values = fishy.evaluate(front_edge_parametric).value
# head_top_values = np.concatenate((top_values, front_values), axis=0)
# head_top_values[:,2] = maxs[2]
# head_bottom_values = head_top_values.copy()
# head_bottom_values[:,1] = -head_bottom_values[:,1]

# num_chordwise_tail = 7
# num_chordwise_modules = 21
# tail_grid = np.linspace(back_edge, modules_back_edge, num_chordwise_tail)
# modules_grid = np.linspace(modules_back_edge, modules_front_edge, num_chordwise_modules)
# head_grid = np.linspace(head_bottom_values, head_top_values, num_spanwise)

# # - Project the tail grid
# tail_parametric_right = fishy.project(points=tail_grid, direction=np.array([0., 0., 1.]), grid_search_density_parameter=11, plot=True)
# # - Project the modules grid
# modules_parametric_right = fishy.project(points=modules_grid, direction=np.array([0., 0., 1.]), grid_search_density_parameter=11, plot=True)
# # - Project the head grid
# head_parametric_right = fishy.project(points=head_grid, direction=np.array([0., 0., 1.]), grid_search_density_parameter=11, plot=True)

# # - Project the left side
# tail_grid[:,:,2] = -tail_grid[:,:,2]
# modules_grid[:,:,2] = -modules_grid[:,:,2]
# head_grid[:,:,2] = -head_grid[:,:,2]
# tail_parametric_left = fishy.project(points=tail_grid, direction=np.array([0., 0., 1.]), grid_search_density_parameter=15, plot=True)
# modules_parametric_left = fishy.project(points=modules_grid, direction=np.array([0., 0., 1.]), grid_search_density_parameter=15, plot=True)
# head_parametric_left = fishy.project(points=head_grid, direction=np.array([0., 0., 1.]), grid_search_density_parameter=15, plot=True)

# # Stich together parametric mesh
# panel_mesh_parametric = np.zeros((2*(num_chordwise_tail + num_chordwise_modules + num_chordwise_head), num_spanwise, 3))
# panel_mesh_parametric[:num_chordwise_tail] = tail_parametric_right.reshape((num_chordwise_tail, num_spanwise, 3))
# panel_mesh_parametric[num_chordwise_tail:num_chordwise_tail + num_chordwise_modules] = \
#     modules_parametric_right.reshape((num_chordwise_modules, num_spanwise, 3))
# panel_mesh_parametric[num_chordwise_tail + num_chordwise_modules:num_chordwise_tail + num_chordwise_modules + num_chordwise_head] = \
#     head_parametric_right.reshape((num_chordwise_head, num_spanwise, 3))
# panel_mesh_parametric[num_chordwise_tail + num_chordwise_modules + num_chordwise_head:num_chordwise_tail + num_chordwise_modules + 2*num_chordwise_head] = \
#     head_parametric_left.reshape((num_chordwise_head, num_spanwise, 3))
# panel_mesh_parametric[num_chordwise_tail + num_chordwise_modules + 2*num_chordwise_head:
#                       num_chordwise_tail + 2*num_chordwise_modules + 2*num_chordwise_head] = \
#     modules_parametric_left.reshape((num_chordwise_modules, num_spanwise, 3))
# panel_mesh_parametric[num_chordwise_tail + 2*num_chordwise_modules + 2*num_chordwise_head:] = \
#     tail_parametric_left.reshape((num_chordwise_tail, num_spanwise, 3))

# # - Plot the parametric mesh
# test = fishy.evaluate(panel_mesh_parametric, plot=True)
# print(test.shape)
# exit()
# endregion Fluid Mesh Parametric Coordinates Definition

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
# fishy_plot = fishy.plot(show=False, opacity=0.3)
# vedo_mesh = vedo.Mesh([displaced_mesh.value, structural_elements]).wireframe()
# plotter = vedo.Plotter()
# plotter.show([fishy_plot, vedo_mesh], axes=1, viewup='y')

# endregion Structural Solver

# region Fit B-Spline to Displacements and Construct Deformed Geometry
displacement_space = lfs.BSplineSpace(
    num_parametric_dimensions=3,
    degree=(2,0,0),
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

# deformed_fishy = fishy.copy()
# # for coefficients in module, evaluate displacements at corresponding location and add them to the current coefficients
# coefficients_parametric_in_module = coefficients_parametric[points_in_module_2_indices,:]
# coefficients_parametric_in_module[:,0] = (coefficients_parametric_in_module[:,0] - module_2_min_u)/(module_2_max_u - module_2_min_u)
# deformations_in_module = structural_displacements_b_spline.evaluate(coefficients_parametric_in_module)
# # NOTE: For line above, a temporary structural displacement B-spline will have to be created for each time step. (where it's coefficients change)
# # indices = list(coefficient_indices[points_in_module_2_indices].reshape((-1)))
# deformed_fishy.coefficients = deformed_fishy.coefficients.set(csdl.slice[points_in_module_2_indices], 
#                                                               fishy.coefficients[points_in_module_2_indices] + deformations_in_module)

# deformed_fishy.plot()
# exit()

derivative_at_module_edge1 = deformed_module.evaluate(np.array([[0., 0.5, 0.5]]), parametric_derivative_orders=(1,0,0))
derivative_at_module_edge = derivative_at_module_edge1/csdl.norm(derivative_at_module_edge1)    # normalize
module_edge = deformed_module.evaluate(np.array([[0., 0.5, 0.5]]))
# endregion Fit B-Spline to Displacements and Construct Deformed Geometry

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
actuation_frequency = 1.
actuation_period = 1/actuation_frequency
# num_vast_time_steps = 7
# num_vast_time_steps = 41
# num_vast_time_steps = 81
# num_cycles = 3
num_cycles = 2
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
# deformed_fishy_weights = compute_chamber_pressure_function(t_pressure_inputs, left_chamber_inputs, time_constant, initial_weight, time)
deformed_fishy_weights = np.sin(2*np.pi*actuation_frequency*time) + 1

# plt.plot(time, deformed_fishy_weights, label='module 1')
# plt.plot(time[:-time.shape[0]//9] + time[time.shape[0]//9], deformed_fishy_weights[:-time.shape[0]//9], label='module 2')
# plt.plot(time[:-2*time.shape[0]//9] + time[2*time.shape[0]//9], deformed_fishy_weights[:-2*time.shape[0]//9], label='module 3')
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel('Interpolation Weight')
# plt.title('Interpolation Weights vs. Time')
# plt.show()
# exit()

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
# indices = np.arange(time.shape[0]*structural_displacements_b_spline.coefficients.value.size).reshape((time.shape[0], structural_displacements_b_spline.coefficients.value.size))
zero_displacements = csdl.Variable(value=np.zeros(structural_displacements_b_spline.coefficients.shape))
actuating_displacement_coefficients_deformed_side = csdl.linear_combination(structural_displacements_b_spline.coefficients,
                                                      zero_displacements,
                                                    num_steps=len(new_deformed_fishy_weights),
                                                   start_weights=new_deformed_fishy_weights, 
                                                   stop_weights=1-new_deformed_fishy_weights)
actuating_displacement_coefficients_opposite_deformed_side = csdl.linear_combination(opposite_structural_displacements_b_spline.coefficients,
                                                      zero_displacements,
                                                    num_steps=len(new_opposite_deformed_fishy_weights),
                                                   start_weights=new_opposite_deformed_fishy_weights, 
                                                   stop_weights=1-new_opposite_deformed_fishy_weights)

actuating_displacement_coefficients = csdl.Variable(name='actuating_displacement_coefficients', 
                                                   value=np.zeros((time.shape[0],) + structural_displacements_b_spline.coefficients.shape))

actuating_displacement_coefficients = actuating_displacement_coefficients.set(csdl.slice[deformed_side_interpolation_indices],
                                                                                actuating_displacement_coefficients_deformed_side)
actuating_displacement_coefficients = actuating_displacement_coefficients.set(csdl.slice[opposite_deformed_side_interpolation_indices],
                                                                                actuating_displacement_coefficients_opposite_deformed_side)

# add displacements directly to coefficients (more efficient because less fitting but questionable accuracy)

# Project geometry coefficients onto geometry to get parametric coordinates (should be about evenly spaced)
# NOTE: This is a bit silly because we already have the parametric coordinates, but I don't want to figure it out.
deformed_fishy = fishy.copy()
coefficients_parametric = fishy.project(points=fishy.coefficients, plot=False, grid_search_density_parameter=5)
# coefficient_indices = np.arange(fishy.coefficients.value.size).reshape((-1,3))
# points_in_front_indices = np.where(coefficients_parametric[:,0] > module_1_max_u)[0]
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
# displacement_indices = np.arange(time.shape[0]*structural_displacements_coefficients.shape[0]).reshape(
#                                 (time.shape[0], structural_displacements_coefficients.shape[0]))

# - Preallocate panel mesh
panel_mesh = csdl.Variable(value=np.zeros((num_vast_time_steps, num_chordwise, num_spanwise, 3)), name='panel_mesh')

plotting_box = ffd_block.copy()
# plotting_box.coefficients = plotting_box.coefficients.set(csdl.slice[:,:,:,2], plotting_box.coefficients[:,:,:,2]*4.5)
plotting_box.coefficients = plotting_box.coefficients.set(csdl.slice[:,:,:,2], plotting_box.coefficients[:,:,:,2]*6.5)
plotting_box = plotting_box.plot(show=False, opacity=0.)

fishy_plot = fishy.plot(show=False, opacity=0.3)
video = vedo.Video('examples/advanced_examples/robotic_fish/temp/fishy_wiggle.mp4', fps=num_steps_per_cycle/2, backend='cv')
for i in range(time.shape[0]):
    # region Module 1
    module_1_displacements_b_spline_at_time_t.coefficients = actuating_displacement_coefficients[i]

    # displacements for in front of module are 0, so don't have deformation (REPLACING THIS SO THIS IS NOT TRUE)
    coefficients_parametric_in_front_of_module = coefficients_parametric[points_in_front_of_module_1_indices,:]
    # edge_parametric_coordinates = np.array([0., coefficients_parametric_in_front_of_module[0,1], coefficients_parametric_in_front_of_module[0,2]])
    edge_parametric_coordinates_u = np.ones((coefficients_parametric_in_front_of_module.shape[0],1))
    edge_parametric_coordinates_v = coefficients_parametric_in_front_of_module[:,1].reshape((-1,1))
    edge_parametric_coordinates_w = coefficients_parametric_in_front_of_module[:,2].reshape((-1,1))
    edge_parametric_coordinates = np.hstack((edge_parametric_coordinates_u, edge_parametric_coordinates_v, edge_parametric_coordinates_w))
    derivative_values_at_edge = module_1_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates, parametric_derivative_orders=(1,0,0))
    # derivative_values_at_edge = module_1_displacements_b_spline_at_time_t.evaluate(np.array([1., 0.5, 0.5]), parametric_derivative_orders=(1,0,0))
    displacement_values_at_edge = module_1_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates)
    delta_u = (coefficients_parametric_in_front_of_module[:,0] - module_1_max_u)/(module_1_max_u - module_1_min_u)  # need to rescale delta_u or derivative due to parametric space mismatch
    delta_u = np.repeat(delta_u, 3).reshape((-1,3))   # repeat each one 3 times to scale the x,y,z
    # displacements_at_coefficients = displacement_values_at_edge + csdl.expand(derivative_values_at_edge, delta_u.shape, 'j->ij')*delta_u
    displacements_at_coefficients = displacement_values_at_edge + derivative_values_at_edge*delta_u
    # indices = coefficient_indices[points_in_front_of_module_1_indices].reshape((-1))
    # deformed_fishy.coefficients[indices] = fishy.coefficients[indices] + displacements_at_coefficients
    deformed_fishy.coefficients = deformed_fishy.coefficients.set(csdl.slice[list(points_in_front_of_module_1_indices)],
                                                                    fishy.coefficients[list(points_in_front_of_module_1_indices)] \
                                                                        + displacements_at_coefficients)
    

    # for coefficients in module, evaluate displacements at corresponding location and add them to the current coefficients
    coefficients_parametric_in_module = coefficients_parametric[points_in_module_1_indices,:]
    coefficients_parametric_in_module[:,0] = (coefficients_parametric_in_module[:,0] - module_1_min_u)/(module_1_max_u - module_1_min_u)
    deformations_in_module = module_1_displacements_b_spline_at_time_t.evaluate(coefficients_parametric_in_module)
    # NOTE: For line above, a temporary structural displacement B-spline will have to be created for each time step. (where it's coefficients change)
    # indices = coefficient_indices[points_in_module_1_indices].reshape((-1))
    # deformed_fishy.coefficients[indices] = fishy.coefficients[indices] + deformations_in_module
    deformed_fishy.coefficients = deformed_fishy.coefficients.set(csdl.slice[list(points_in_module_1_indices)],
                                                                    fishy.coefficients[list(points_in_module_1_indices)] + deformations_in_module)

    # For coefficients behind the module, evaluate the derivative at the end of the module and apply the rigid link
    coefficients_parametric_behind_module = coefficients_parametric[points_behind_module_1_indices,:]
    # edge_parametric_coordinates = np.array([0., coefficients_parametric_behind_module[0,1], coefficients_parametric_behind_module[0,2]])
    edge_parametric_coordinates_u = np.zeros((coefficients_parametric_behind_module.shape[0],1))
    edge_parametric_coordinates_v = coefficients_parametric_behind_module[:,1].reshape((-1,1))
    edge_parametric_coordinates_w = coefficients_parametric_behind_module[:,2].reshape((-1,1))
    edge_parametric_coordinates = np.hstack((edge_parametric_coordinates_u, edge_parametric_coordinates_v, edge_parametric_coordinates_w))
    derivative_values_at_edge = module_1_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates, parametric_derivative_orders=(1,0,0))
    # derivative_values_at_edge = module_1_displacements_b_spline_at_time_t.evaluate(np.array([0., 0.5, 0.5]), parametric_derivative_orders=(1,0,0))
    displacement_values_at_edge = module_1_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates)
    delta_u = (coefficients_parametric_behind_module[:,0] - module_1_min_u)/(module_1_max_u - module_1_min_u)  # need to rescale delta_u or derivative due to parametric space mismatch
    delta_u = np.repeat(delta_u, 3).reshape((-1,3))   # repeat each one 3 times to scale the x,y,z
    displacements_at_coefficients = displacement_values_at_edge + derivative_values_at_edge*delta_u
    # indices = coefficient_indices[points_behind_module_1_indices].reshape((-1))
    # deformed_fishy.coefficients[indices] = fishy.coefficients[indices] + displacements_at_coefficients
    deformed_fishy.coefficients = deformed_fishy.coefficients.set(csdl.slice[list(points_behind_module_1_indices)],
                                                                    fishy.coefficients[list(points_behind_module_1_indices)] \
                                                                        + displacements_at_coefficients)

    # endregion Module 1

    # region Module 2
    time_step_offset = 1*num_vast_time_steps//(num_cycles*num_modules)   # (divide by 2 to get one period, then phase offset by 1/3 of period)
    # if i >= time_step_offset:
    module_displacement_coefficients = actuating_displacement_coefficients[i - time_step_offset,:]
    module_2_displacements_b_spline_at_time_t.coefficients = module_displacement_coefficients

    # displacements for in front of module are 0, so don't have deformation (REPLACING THIS SO THIS IS NOT TRUE)
    coefficients_parametric_in_front_of_module = coefficients_parametric[points_in_front_of_module_2_indices,:]
    # edge_parametric_coordinates = np.array([0., coefficients_parametric_in_front_of_module[0,1], coefficients_parametric_in_front_of_module[0,2]])
    edge_parametric_coordinates_u = np.ones((coefficients_parametric_in_front_of_module.shape[0],1))
    edge_parametric_coordinates_v = coefficients_parametric_in_front_of_module[:,1].reshape((-1,1))
    edge_parametric_coordinates_w = coefficients_parametric_in_front_of_module[:,2].reshape((-1,1))
    edge_parametric_coordinates = np.hstack((edge_parametric_coordinates_u, edge_parametric_coordinates_v, edge_parametric_coordinates_w))
    derivative_values_at_edge = module_2_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates, parametric_derivative_orders=(1,0,0))
    displacement_values_at_edge = module_2_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates)
    delta_u = (coefficients_parametric_in_front_of_module[:,0] - module_2_max_u)/(module_2_max_u - module_2_min_u)  # need to rescale delta_u or derivative due to parametric space mismatch
    delta_u = np.repeat(delta_u, 3).reshape((-1,3))   # repeat each one 3 times to scale the x,y,z
    displacements_at_coefficients = displacement_values_at_edge + derivative_values_at_edge*delta_u
    # indices = coefficient_indices[points_in_front_of_module_2_indices].reshape((-1))
    # deformed_fishy.coefficients[indices] = fishy.coefficients[indices] + displacements_at_coefficients
    deformed_fishy.coefficients = deformed_fishy.coefficients.set(csdl.slice[list(points_in_front_of_module_2_indices)],
                                                                    deformed_fishy.coefficients[list(points_in_front_of_module_2_indices)] \
                                                                        + displacements_at_coefficients)
    

    # for coefficients in module, evaluate displacements at corresponding location and add them to the current coefficients
    coefficients_parametric_in_module = coefficients_parametric[points_in_module_2_indices,:]
    coefficients_parametric_in_module[:,0] = (coefficients_parametric_in_module[:,0] - module_2_min_u)/(module_2_max_u - module_2_min_u)
    deformations_in_module = module_2_displacements_b_spline_at_time_t.evaluate(coefficients_parametric_in_module)
    # NOTE: For line above, a temporary structural displacement B-spline will have to be created for each time step. (where it's coefficients change)
    # indices = coefficient_indices[points_in_module_2_indices].reshape((-1))
    # deformed_fishy.coefficients[indices] = fishy.coefficients[indices] + deformations_in_module
    deformed_fishy.coefficients = deformed_fishy.coefficients.set(csdl.slice[list(points_in_module_2_indices)],
                                                                    deformed_fishy.coefficients[list(points_in_module_2_indices)] + deformations_in_module)

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
    # deformed_fishy.coefficients[indices] = fishy.coefficients[indices] + displacements_at_coefficients
    deformed_fishy.coefficients = deformed_fishy.coefficients.set(csdl.slice[list(points_behind_module_2_indices)],
                                                                    deformed_fishy.coefficients[list(points_behind_module_2_indices)] \
                                                                        + displacements_at_coefficients)
    # endregion Module 2

    # region Module 3
    time_step_offset = 2*(num_cycles*num_modules)   # (divide by 2 to get one period, then phase offset by 2/3 of period)
    # if i >= time_step_offset:
    module_displacement_coefficients = actuating_displacement_coefficients[i - time_step_offset,:]
    module_3_displacements_b_spline_at_time_t.coefficients = module_displacement_coefficients

    # displacements for in front of module are 0, so don't have deformation (REPLACING THIS SO THIS IS NOT TRUE)
    coefficients_parametric_in_front_of_module = coefficients_parametric[points_in_front_of_module_3_indices,:]
    # edge_parametric_coordinates = np.array([0., coefficients_parametric_in_front_of_module[0,1], coefficients_parametric_in_front_of_module[0,2]])
    edge_parametric_coordinates_u = np.ones((coefficients_parametric_in_front_of_module.shape[0],1))
    edge_parametric_coordinates_v = coefficients_parametric_in_front_of_module[:,1].reshape((-1,1))
    edge_parametric_coordinates_w = coefficients_parametric_in_front_of_module[:,2].reshape((-1,1))
    edge_parametric_coordinates = np.hstack((edge_parametric_coordinates_u, edge_parametric_coordinates_v, edge_parametric_coordinates_w))
    derivative_values_at_edge = module_3_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates, parametric_derivative_orders=(1,0,0))
    displacement_values_at_edge = module_3_displacements_b_spline_at_time_t.evaluate(edge_parametric_coordinates)
    delta_u = (coefficients_parametric_in_front_of_module[:,0] - module_3_max_u)/(module_3_max_u - module_3_min_u)  # need to rescale delta_u or derivative due to parametric space mismatch
    delta_u = np.repeat(delta_u, 3).reshape((-1,3))   # repeat each one 3 times to scale the x,y,z
    displacements_at_coefficients = displacement_values_at_edge + derivative_values_at_edge*delta_u
    # indices = coefficient_indices[points_in_front_of_module_3_indices].reshape((-1))
    # deformed_fishy.coefficients[indices] = fishy.coefficients[indices] + displacements_at_coefficients
    deformed_fishy.coefficients = deformed_fishy.coefficients.set(csdl.slice[list(points_in_front_of_module_3_indices)],
                                                                    deformed_fishy.coefficients[list(points_in_front_of_module_3_indices)] \
                                                                        + displacements_at_coefficients)
    

    # for coefficients in module, evaluate displacements at corresponding location and add them to the current coefficients
    coefficients_parametric_in_module = coefficients_parametric[points_in_module_3_indices,:]
    coefficients_parametric_in_module[:,0] = (coefficients_parametric_in_module[:,0] - module_3_min_u)/(module_3_max_u - module_3_min_u)
    deformations_in_module = module_3_displacements_b_spline_at_time_t.evaluate(coefficients_parametric_in_module)
    # NOTE: For line above, a temporary structural displacement B-spline will have to be created for each time step. (where it's coefficients change)
    # indices = coefficient_indices[points_in_module_3_indices].reshape((-1))
    # deformed_fishy.coefficients[indices] = fishy.coefficients[indices] + deformations_in_module
    deformed_fishy.coefficients = deformed_fishy.coefficients.set(csdl.slice[list(points_in_module_3_indices)],
                                                                    deformed_fishy.coefficients[list(points_in_module_3_indices)] \
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
    # deformed_fishy.coefficients[indices] = fishy.coefficients[indices] + displacements_at_coefficients
    deformed_fishy.coefficients = deformed_fishy.coefficients.set(csdl.slice[list(points_behind_module_3_indices)],
                                                                    deformed_fishy.coefficients[list(points_behind_module_3_indices)] \
                                                                        + displacements_at_coefficients)
    # endregion Module 3

    # # region Evaluate Vortex Panel Method Mesh
    panel_mesh_this_timestep = deformed_fishy.evaluate(panel_method_parametric_mesh, plot=False)
    panel_mesh = panel_mesh.set(csdl.slice[i], panel_mesh_this_timestep)

    # # endregion Evaluate Vortex Panel Method Mesh

    deformed_fishy_plot = deformed_fishy.plot(opacity=0.5, show=False)
    plotter = vedo.Plotter(offscreen=True)
    # plotter.show([plotting_box, fishy_plot, deformed_fishy_plot], axes=1, viewup='y')
    vertices = []
    faces = []
    for u_index in range(panel_mesh_this_timestep.shape[0]):
        for v_index in range(panel_mesh_this_timestep.shape[1]):
            vertex = tuple(panel_mesh_this_timestep.value[u_index,v_index,:])
            vertices.append(vertex)
            if u_index != 0 and v_index != 0:
                face = tuple((
                    (u_index-1)*panel_mesh_this_timestep.shape[1]+(v_index-1),
                    (u_index-1)*panel_mesh_this_timestep.shape[1]+(v_index),
                    (u_index)*panel_mesh_this_timestep.shape[1]+(v_index),
                    (u_index)*panel_mesh_this_timestep.shape[1]+(v_index-1),
                ))
                faces.append(face)

    vedo_mesh = vedo.Mesh([vertices, faces]).wireframe().linewidth(2).color('#F5F0E6')
    plotter.show([plotting_box, deformed_fishy_plot, vedo_mesh], axes=1, viewup='y')
    video.add_frame()
video.close()
pickle.dump(panel_mesh.value, open('examples/advanced_examples/robotic_fish/temp/panel_mesh.pickle', 'wb'))
exit()

# endregion Construct Dynamic Geometry and Dynamic Panel Method Mesh






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


# # region Optimization

# length.set_as_design_variable(lower=0.3, upper=1., scaler=1.e1)
# width.set_as_design_variable(lower=0.02, upper=0.08, scaler=1.e2)
# height.set_as_design_variable(lower=0.03, upper=0.14, scaler=1.e2)

# objective.set_as_objective()


# sim = csdl.experimental.PySimulator(recorder=recorder)
# optimization_problem = CSDLAlphaProblem(problem_name='fishy_optimization', simulator=sim)
# # optimizer = SLSQP(optimization_problem, maxiter=100, ftol=1.e-7)
# optimizer = SLSQP(optimization_problem, maxiter=100, ftol=1.e-9)

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

