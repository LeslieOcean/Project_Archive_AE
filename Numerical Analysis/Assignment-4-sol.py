# FEM for 2D convection-diffusion equation 
# compact Gaussian sources and spatiallly nonuniform wind
# runs on Python 3
# DOLFINx version: 0.9.0

import numpy as np
import matplotlib.pyplot as plt
import time
# import matplotlib
# matplotlib.use('qtagg')
import matplotlib.tri as mtri
from scipy.interpolate import griddata
import pyvista 

import ufl
from basix.ufl import element
from dolfinx import fem, mesh, geometry, plot, io
from ufl import ds, dx, grad, inner

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from petsc4py import PETSc
from dolfinx.fem.petsc import ( assemble_vector, assemble_matrix, create_vector,
                                apply_lifting, set_bc )

from dolfinx import __version__

dolfinx_version = '0.9.0'

if __version__ != dolfinx_version:
    print('dolfinx version is',__version__,', this code requires version',dolfinx_version)
else:
    print('dolfinx version',dolfinx_version,'found')


x0 = -3 # domain corner, x coordinate
y0 = -3 # domain corner, y coordinate
Lx = 6 # domain extent in x-direction
Ly = 6 # domain extent in y-direction
T = 5 # time interval
As = 5 # source magnitude
alpha = 10 # source decay rate in space
B = 0.5 # wind constant
D = 0.0025 # diffusivity constant
bcValue = 0 # Dirichlet BC value


nx = 181 # number of cells in x-direction
ny = 181 # number of cells in y-direction

# creating domain and mesh
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([x0, y0]), np.array([x0+Lx, y0+Ly])], [nx, ny])

# Scalar function space V with scalar elements P1
P1 = element("Lagrange",domain.topology.cell_name(),1)
V = fem.functionspace(domain, P1)
# Vector function space W with vector elements P2
P2 = element("Lagrange",domain.topology.cell_name(),1,shape=(2,))
W = fem.functionspace(domain, P2)

# Create Dirichlet boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(bcValue), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# source function class
class source_function():
    def __init__(self):
        self.t = 0
        self.alpha = 0
        self.A = 0
    def eval(self, x):
        x1 = -2.4 #source location
        y1 = 0
        x2 = 0
        y2 = 1
        f = self.A*(np.exp(-self.alpha*(x[0]-x1)**2-self.alpha*(x[1]-y1)**2)+np.exp(-self.alpha*(x[0]-x2)**2-self.alpha*(x[1]-y2)**2))
        return f
# velocity x-component function class
class velocity_x:
    def __init__(self):
        self.t = 0
        self.B = 0
    def eval(self, x):
        wx = self.B - x[1]/np.sqrt(x[0]**2+x[1]**2)
        return wx
# velocity y-component function class
class velocity_y:
    def __init__(self):
        self.t = 0
    def eval(self, x):
        wy = x[0]/np.sqrt(x[0]**2+x[1]**2)
        return wy
# initial condition Python function
def initial_condition(x):
    u = x[0]*0
    return u

def get_value(fun,x,domain):
    '''This function computes the value of the solution at a chosen point
    fun - FEM funciton, such as the current iteration u_k
    x - numpy 3D array with the point coordinates, e.g., x = np.array([0.5,0,0])
    domain - dolfinx mesh object
    '''
    bb_tree = geometry.bb_tree(domain,domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree,x=x)
    colliding_cells = geometry.compute_colliding_cells(domain,cell_candidates,x=x)
    value = fun.eval(x=x,cells=colliding_cells.array[0])
    return value[0]

# source fem function
fun = source_function()
fun.alpha = alpha
fun.A = As
f = fem.Function(V)
f.interpolate(fun.eval)

# velocity fem vector function
vxf = velocity_x()
vxf.B = B
vyf = velocity_y()
vx = fem.Function(V)
vx.interpolate(vxf.eval)
vy = fem.Function(V)
vy.interpolate(vyf.eval)
w = fem.Function(W)
w.sub(0).interpolate(vx)
w.sub(1).interpolate(vy)

#------------------------------------------------------------------------------
#visualize velocity vector field w
# pyvista.start_xvfb()                                            #pyvista doesn't work directly due to some GPU problem and cannot operate with matplotlib together
#                                                                 #run the code alone and open the saved file
# plotter = pyvista.Plotter()
# tdim = domain.topology.dim
# domain.topology.create_connectivity(tdim, tdim)
# topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
# w_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# w_values = np.zeros(geometry.shape)
# w_values[:,:2] = w.x.array.reshape(-1,2)
# w_grid.point_data['w'] = w_values
# plotter.add_mesh(w_grid, color='grey', opacity=0.3, show_edges=False)
# plotter.add_arrows(w_grid.points, w_values, mag=0.1, label='Velocity Vector Field w')
# plotter.add_legend()
# plotter.save_graphic("vectorfieldw_0.5.pdf")
# plotter.close()

# xdmf = io.XDMFFile(domain.comm, "velocityfieldw.xdmf", "w")                 #or visualize with paraview
# xdmf.write_mesh(domain)
# xdmf.write_function(w)
# xdmf.close()

# fig = plt.figure(2)                                                           #or use matplotlib to show magnitude
# ax = fig.add_subplot()
# coord = W.tabulate_dof_coordinates()
# wx_coord = coord[:,0]
# wy_coord = coord[:,1]
# #wx_coord, wy_coord = np.meshgrid(wx_coord, wy_coord)
# w_value = vx.x.array**2 + vy.x.array**2
# #ax.quiver(wx_coord, wy_coord, wx, wy)
# im_field = ax.tripcolor(wx_coord, wy_coord, w_value, shading='gouraud')
# ax.set_aspect('equal', adjustable='box')
# cbar = fig.colorbar(im_field,ax=ax,orientation='horizontal')
# ax.set_title('Velocity Vector Field w Magnitude')
# plt.show()

# fig = plt.figure(3)
# ax = fig.add_subplot()
# npoint = 31
# x = np.linspace(x0, x0+Lx, npoint)
# y = np.linspace(y0, y0+Ly, npoint)
# wx = w.sub(0).eval()
# x, y = np.meshgrid(x, y)



# initial condition fem function
u_k = fem.Function(V)
u_k.name = "u_k"
u_k.interpolate(initial_condition)

# solution variable fem function
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(lambda x: x[0]*0)

''' '''
#for question 2.b.c.e
nt = 50 # number of time steps
dt = (T-0)/nt # time step
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = u*v*dx + D*dt*inner(grad(u),grad(v))*dx + dt*inner(w, grad(u))*v*dx # bi-linear form inner or dot -> they are the same
L = (dt*f + u_k)*v*dx # linear form
u = fem.Function(V)
bilinear_form = fem.form(a)
linear_form = fem.form(L)
A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(linear_form)

solver = PETSc.KSP().create(domain.comm) #using petsc4py to creata a linear solver
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)


t = 0.0

# preparing to plot solution at each time step
plt.ion()
plt.figure(1)
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1, num=1)
unow = u_k.x.array
dof_coordinates = V.tabulate_dof_coordinates()
dof_x = dof_coordinates[:,0]
dof_y = dof_coordinates[:,1]
im =  ax.tripcolor(dof_x,dof_y,unow,shading='gouraud')
ax.set_aspect('equal', adjustable='box')
cbar = fig.colorbar(im,ax=ax,orientation='horizontal')
ax.set_title(r'$u(t)$, t = '+str(np.round(t,2)))
#plt.show()
#plt.savefig('initial.png')
ut = []
#start_time = time.time() #calculate the computational time

# Time stepping
for k in range(nt):
    t = t + dt # updating time

    # update the right hand side re-using the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode. REVERSE)
    set_bc(b, [bc])

    # solve linear system
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # overwrite previous time step
    u_k.x.array[:] = uh.x.array

    # plot solution snapshot
    unow = u_k.x.array
    im.remove()
    im = ax.tripcolor(dof_x,dof_y,unow,shading='gouraud')
    cbar.update_normal(im)
    ax.set_title(r'$u(t)$, t = '+str(np.round(t,2)))
    plt.pause(0.2)
    #plt.savefig(f'solution_{nx}_{t}.png')
# for i in range(100):
#     xt = 2+i/100
#     yt = np.array([xt,0,0])
#     utnow = get_value(u_k, yt, domain)
#     ut.append(utnow)
# ut = np.array(ut)
# print("ut =", np.array2string(ut, separator=', ', precision=6))
#end_time = time.time()

# elapsed_time = end_time - start_time
# print(f"Computation completed in {elapsed_time:.2f} seconds.")

plt.ioff()
plt.show()


'''
# for question 2.d
nt = np.array([10, 12, 15, 20, 50, 100]) # number of time steps
a = 0.0
ut = []

for n_t in nt:
    dt = (T-0)/n_t # time step
    # Weak form: a(u,v) = L(v) for backward Euler method
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = u*v*dx + D*dt*inner(grad(u),grad(v))*dx + dt*inner(w, grad(u))*v*dx # bi-linear form inner or dot -> they are the same
    L = (dt*f + u_k)*v*dx # linear form
    u = fem.Function(V)
    bilinear_form = fem.form(a)
    linear_form = fem.form(L)
    A = assemble_matrix(bilinear_form, bcs=[bc])
    A.assemble()
    b = create_vector(linear_form)

    solver = PETSc.KSP().create(domain.comm) #using petsc4py to creata a linear solver
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    t = 0.0
    # Time stepping
    
    for k in range(n_t):
        t = t + dt # updating time

        # update the right hand side re-using the initial vector
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form)

        # apply Dirichlet boundary condition to the vector
        apply_lifting(b, [bilinear_form], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode. REVERSE)
        set_bc(b, [bc])

        # solve linear system
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        # overwrite previous time step
        u_k.x.array[:] = uh.x.array

        # plot solution snapshot
        unow = u_k.x.array
        # print(k)

    # print solution value at a point
    xt = np.array([0, -1, 0]) # location of (xt,yt) in 3D array
    utnow = get_value(u_k, xt, domain) # solution at this point
    ut.append(utnow)
ut = np.array(ut, dtype=float)
print(ut)
plt.plot(nt, ut, marker='o')
plt.xlabel("nt")
plt.ylabel("Value at the point (0, -1)")
plt.show()

'''