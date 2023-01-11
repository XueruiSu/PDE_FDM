import numpy as np
import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import font_manager
# 解决中文乱码问题
myfont = font_manager.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc", size=14)

# 计算poisson方程右边常量
def build_up_b(b, rho, dt, u, v, dx, dy):
    
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    return b
# poisson 方程计算：
def pressure_poisson(p, dx, dy, b):
    pn = numpy.empty_like(p)
    pn = p.copy()
    nit = 50
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1,1:-1])

        p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
        p[-1, :] = 0        # p = 0 at y = 2
        
    return p
# 迭代nt个时间步：
def cavity_flow_parallel(nt, u, v, dt, dx, dy, p, rho, nu):
    un = numpy.empty_like(u)
    vn = numpy.empty_like(v)
    b = numpy.zeros((u.shape[0], u.shape[1]))
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        # b = build_up_b(b, rho, dt, u, v, dx, dy)
        # p = pressure_poisson(p, dx, dy, b)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                       (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                       (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        b = build_up_b(b, rho, dt, un, vn, dx, dy)
        p = pressure_poisson(p, dx, dy, b)
        
        u[0, :]  = 0
        u[:, 0]  = 0
        u[:, -1] = 0
        u[-1, :] = 1    # set velocity on cavity lid equal to 1
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0
     
    return u, v, p

def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    un = numpy.empty_like(u)
    vn = numpy.empty_like(v)
    b = numpy.zeros((u.shape[0], u.shape[1]))
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                       (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                       (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # b = build_up_b(b, rho, dt, un, vn, dx, dy)
        # p = pressure_poisson(p, dx, dy, b)
        
        u[0, :]  = 0
        u[:, 0]  = 0
        u[:, -1] = 0
        u[-1, :] = 1    # set velocity on cavity lid equal to 1
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0
     
    return u, v, p
'''
# 基本实验参数：
nx = 41
ny = 41
nt = 500
nit = 50
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)
X, Y = numpy.meshgrid(x, y)

rho = 1
nu = .1
dt = .001
nt = 800

# 串行式数值迭代：
u = numpy.zeros((ny, nx))
v = numpy.zeros((ny, nx))
p = numpy.zeros((ny, nx))
b = numpy.zeros((ny, nx))
u_list_serial = []
v_list_serial = []
p_list_serial = []
for index in range(nt):
    u, v, p = cavity_flow(1, u, v, dt, dx, dy, p, rho, nu)
    u_list_serial.append(u)
    v_list_serial.append(v)
    p_list_serial.append(p)
    fig = pyplot.figure(figsize=(11,7), dpi=100)
    # plotting the pressure field as a contour
    pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)  
    pyplot.colorbar()
    # plotting the pressure field outlines
    pyplot.contour(X, Y, p, cmap=cm.viridis)  
    # plotting velocity field
    pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
    pyplot.xlabel('X')
    pyplot.ylabel('Y')
    fig.savefig(f'./figure/_serial_{index+1}.jpg', fps=30)
    # pyplot.show()
# 画动图：
import imageio
with imageio.get_writer(uri='_serial_test_LR.gif', mode='I', fps=10) as writer:
    for index in range(nt):
        if index % 20 == 0:
            writer.append_data(imageio.imread(f'./figure/_serial_{index+1}.jpg'))

# 并行式数值迭代： 
u = numpy.zeros((ny, nx))
v = numpy.zeros((ny, nx))
p = numpy.zeros((ny, nx))
b = numpy.zeros((ny, nx))
u_list_parallel = []
v_list_parallel = []
p_list_parallel = []
for index in range(nt):
    u, v, p = cavity_flow_parallel(1, u, v, dt, dx, dy, p, rho, nu)
    u_list_parallel.append(u)
    v_list_parallel.append(v)
    p_list_parallel.append(p)
    fig = pyplot.figure(figsize=(11,7), dpi=100)
    # plotting the pressure field as a contour
    pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)  
    pyplot.colorbar()
    # plotting the pressure field outlines
    pyplot.contour(X, Y, p, cmap=cm.viridis)  
    # plotting velocity field
    pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
    pyplot.xlabel('X')
    pyplot.ylabel('Y')
    fig.savefig(f'./figure/_parallel_{index+1}.jpg', fps=30)
    # pyplot.show()
# 画动图：
import imageio
with imageio.get_writer(uri='_parallel_test_LR.gif', mode='I', fps=10) as writer:
    for index in range(nt):
        if index % 20 == 0:
            writer.append_data(imageio.imread(f'./figure/_parallel_{index+1}.jpg'))
'''   
   
         
# 方法一：基于Artist Animation函数，quiver类不可迭代导致这个部分加不进去。
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# fig = plt.figure()
# pyplot.xlabel('X')
# pyplot.ylabel('Y')

# artists = []
# for i in range(nt):
#     frame = []
#     u = u_list[i]
#     v = v_list[i]
#     p = p_list[i]
#     frame += pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis).collections    # 注意这里要+=，对列表操作而不是appand
#     frame += pyplot.contour(X, Y, p, cmap=cm.viridis).collections 
#     frame += iter(pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]))
#     artists.append(frame)
# # pyplot.colorbar()
# ani = animation.ArtistAnimation(fig=fig, artists=artists, repeat=False, interval=10)
# plt.show()
# ani.save('2.gif', fps=30)



# 方法二基于FuncAnimation，无法往里面填多类数据
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)

# x = np.linspace(0, 2 * np.pi, 5000)
# y = np.exp(-x) * np.cos(2 * np.pi * x)
# line, = ax.plot(x, y, color="cornflowerblue", lw=3)
# ax.set_ylim(-1.1, 1.1)

# # 清空当前帧
# def init():
#     line.set_ydata([np.nan] * len(x))
#     return line,

# # 更新新一帧的数据
# def update(frame):
#     line.set_ydata(np.exp(-x) * np.cos(2 * np.pi * x + float(frame)/100))
#     return line,

# # 调用 FuncAnimation
# ani = FuncAnimation(fig
#                    ,update
#                    ,init_func=init
#                    ,frames=200
#                    ,interval=2
#                    ,blit=True
#                    )

# ani.save("animation.gif", fps=25, writer="imagemagick")
# plt.show()


