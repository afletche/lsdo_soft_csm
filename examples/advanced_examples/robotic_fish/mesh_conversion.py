import meshio

msh = meshio.read("examples/example_geometries/cube_14_node_mesh.msh")

cells = msh.get_cells_type("tetra")
meshio.write("examples/example_geometries/cube_14_node_mesh.xdmf", meshio.Mesh(points=msh.points/10, cells={"tetra": cells}))