import numpy as np
import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import font_manager
# 解决中文乱码问题
myfont = font_manager.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc", size=14)
import plot_gif_test

# 基本实验参数：
nx = 41
ny = 41
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)
X, Y = numpy.meshgrid(x, y)

rho = 1
nu = .1
dt = .001
nt = 2000
HR_len = 10

# 串行式和并行式数值迭代：
u_serial = numpy.zeros((ny, nx))
v_serial = numpy.zeros((ny, nx))
p_serial = numpy.zeros((ny, nx))
u_list_serial = []
v_list_serial = []
p_list_serial = []
u_parallel = numpy.zeros((ny, nx))
v_parallel = numpy.zeros((ny, nx))
p_parallel = numpy.zeros((ny, nx))
u_list_parallel = []
v_list_parallel = []
p_list_parallel = []
u_real = numpy.zeros((ny, nx))
v_real = numpy.zeros((ny, nx))
p_real = numpy.zeros((ny, nx))
u_list_real = []
v_list_real = []
p_list_real = []

for index in range(nt):
    # 串行数值迭代：
    u_serial, v_serial, p_serial = plot_gif_test.cavity_flow(1, u_serial, v_serial, dt, dx, dy, p_serial, rho, nu)
    u_list_serial.append(u_serial)
    v_list_serial.append(v_serial)
    p_list_serial.append(p_serial)
    if index % 50 == 0:
        print("串行数值迭代：", "mean(u):", np.mean(u_serial), 
            "mean(v):", np.mean(v_serial), 
            "mean(p):", np.mean(p_serial))
    for _ in range(HR_len):
        u_real, v_real, p_real = plot_gif_test.cavity_flow(1, u_real, v_real, dt/HR_len, dx, dy, p_real, rho, nu)
    u_list_real.append(u_real)
    v_list_real.append(v_real)
    p_list_real.append(p_real)

    # 并行数值迭代：
    u_parallel, v_parallel, p_parallel = plot_gif_test.cavity_flow_parallel(1, u_parallel, v_parallel, dt, dx, dy, p_parallel, rho, nu)
    u_list_parallel.append(u_parallel)
    v_list_parallel.append(v_parallel)
    p_list_parallel.append(p_parallel)
    if index % 50 == 0:    
        print("并行数值迭代：", "mean(u):", np.mean(u_parallel), 
            "mean(v):", np.mean(v_parallel), 
            "mean(p):", np.mean(p_parallel))
        print("两个数值算法的差距：", "mean(u1-u2):", np.mean(u_serial - u_parallel), 
            "mean(v1-v2):", np.mean(v_serial - v_parallel), 
            "mean(p1-p2):", np.mean(p_serial - p_parallel))
        print("串行数值算法与真值的差距：", "mean(u1-u):", np.mean(u_serial - u_real), 
            "mean(v1-v):", np.mean(v_serial - v_real), 
            "mean(p1-p):", np.mean(p_serial - p_real))
        print("并行数值算法与真值的差距：", "mean(u2-u):", np.mean(u_parallel - u_real), 
            "mean(v2-v):", np.mean(v_parallel - v_real), 
            "mean(p2-p):", np.mean(p_parallel - p_real))  
        
# 画动图：
import imageio
with imageio.get_writer(uri='_serial_test_11_2.gif', mode='I', fps=10) as writer:
    for index in range(nt):
        if index % 20 == 0:
            fig = pyplot.figure(figsize=(11,7), dpi=100)
            # plotting the pressure field as a contour
            pyplot.contourf(X, Y, p_list_serial[index], alpha=0.5, cmap=cm.viridis)  
            pyplot.colorbar()
            # plotting the pressure field outlines
            pyplot.contour(X, Y, p_list_serial[index], cmap=cm.viridis)  
            # plotting velocity field
            pyplot.quiver(X[::2, ::2], Y[::2, ::2], u_list_serial[index][::2, ::2], v_list_serial[index][::2, ::2]) 
            pyplot.xlabel('X')
            pyplot.ylabel('Y')
            fig.savefig(f'./figure/_serial_{index+1}.jpg', fps=30)
            writer.append_data(imageio.imread(f'./figure/_serial_{index+1}.jpg'))
import imageio
with imageio.get_writer(uri='_parallel_test_11_2.gif', mode='I', fps=10) as writer:
    for index in range(nt):
        if index % 20 == 0:
            fig = pyplot.figure(figsize=(11,7), dpi=100)
            # plotting the pressure field as a contour
            pyplot.contourf(X, Y, p_list_parallel[index], alpha=0.5, cmap=cm.viridis)  
            pyplot.colorbar()
            # plotting the pressure field outlines
            pyplot.contour(X, Y, p_list_parallel[index], cmap=cm.viridis)  
            # plotting velocity field
            pyplot.quiver(X[::2, ::2], Y[::2, ::2], u_list_parallel[index][::2, ::2], v_list_parallel[index][::2, ::2]) 
            pyplot.xlabel('X')
            pyplot.ylabel('Y')
            fig.savefig(f'./figure/_parallel_{index+1}.jpg', fps=30)
            writer.append_data(imageio.imread(f'./figure/_parallel_{index+1}.jpg'))


