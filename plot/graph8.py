import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rcParams['font.size'] = 14

# 创建一个图形和一个坐标轴
width_ = 9.
height_ = width_*3/4
fig, (ax, ax2) = plt.subplots(1, 2, dpi=800, figsize=(width_, height_))
##################################### TIME ##################################
##### 2 T 有关联 ########
# 定义数据点
x = np.array([2*8, 6*8, 12*8])
y = np.array([791.872,  264.772, 133.152])

# 绘制散点图
ax.scatter(x, y, color = "red")
linx = np.linspace(16, 320*8, 100)
ax.plot(linx, 9.8*84/(linx/16), 'r')
ax.plot(-linx, -9.8*84/(linx/16), 'r-o', label='4T, post-processing')
##### 16 T 有关联 ########
# 定义数据点
x = np.array([32*8])
y = np.array([17.182])
# 绘制散点图
ax.scatter(x, y, color = "blue", label='32T, post-processing')
# ax.plot(linx, 17.182*1/(linx/(32*8)), 'b')
# ax.plot(-linx, -17.182*1/(linx/(32*8)), 'b-o', label='32T, post-processing')

##### 2 T 无关联 ########
# 定义数据点
x = np.array([34*8, 132*8, 66*8, 264*8])
y = np.array([247.518, 63.662, 127.893, 32.509])

# 绘制散点图
ax.scatter(x, y, color = "r", marker = "s")
ax.plot(linx, 8.9*525/(linx/16), 'r')
ax.plot(-linx, -8.9*525/(linx/16), 'r-s', label = '4T, no post-processing')
##### 16 T 无关联 ########
# 定义数据点
x = np.array([32*8, 32*3*8, 32*5*8, 32*9*8])
y = np.array([127.473, 41.736, 28.767, 14.22])
# 绘制散点图
ax.scatter(x, y, color = "b", marker = "s")
ax.plot(linx, 14.9*9/(linx/(32*8)), 'b-')
ax.plot(-linx, -14.9*9/(linx/(32*8)), 'b-s', label='32T, no post-processing')
# ax.plot(linx, 304, 'green', label='Sunway')
ax.plot(linx, 600*np.ones(len(linx)), 'g', label='Sycamore')

ax.set_ylabel('Time-to-solution (seconds)')
ax.set_ylim([10, 10**5])
ax.legend(loc='best') # 在左上角显示图例
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_title('(a) Time-to-solution')
##################################### ENERGY ##################################
ax2.yaxis.set_label_position('right')
ax2.yaxis.tick_right()

##### 2 T 有关联 ########
# 定义数据点
x = np.array([2*8, 6*8, 12*8])
y = np.array([1.1253,  1.131048, 1.12373])

# 绘制散点图
ax2.scatter(x, y, c = "r",  marker = "v")
ax2.plot(linx, 1.13*np.ones(100), 'r--')
ax2.plot(-linx, -1.13*np.ones(100), 'r--v', label='4T, post-processing')
##### 16 T 有关联 ########
# 定义数据点
x = np.array([32*8])
y = np.array([0.29])

# 绘制散点图
ax2.scatter(x, y, c  ="b",  marker = "v", label='32T, post-processing')

# ax2.plot(linx, 0.29*np.ones(100), 'b--')
# ax2.plot(-linx, -0.29*np.ones(100), 'b--v', label='32T, post-processing')

##### 2 T 无关联 ########
# 定义数据点
x = np.array([34*8, 132*8, 66*8, 264*8])
y = np.array([6.022,  5.820, 5.929, 5.768])

# 绘制散点图
ax2.scatter(x, y, c = "r",  marker = "x")
ax2.plot(linx, np.mean(y)*np.ones(len(linx)), 'r--')
ax2.plot(-linx, -np.mean(y)*np.ones(len(linx)), 'r--x', label='4T, no post-processing')
##### 16 T 无关联 ########
# 定义数据点
x = np.array([32*8, 32*3*8, 32*5*8, 32*9*8])
y = np.array([2.3474, 2.3614, 2.607, 2.3980])

# 绘制散点图
ax2.scatter(x, y, c  ="b",  marker = "x")

ax2.plot(linx, np.mean(y)*np.ones(len(linx)), 'b--')
ax2.plot(-linx, -np.mean(y)*np.ones(len(linx)), 'b--x', label='32T, no post-processing')


# ax.plot(linx, 304, 'green', label='Sunway')
ax2.plot(linx, 4.3*np.ones(len(linx)), 'g--', label='Sycamore')

ax2.set_ylabel("Power consumption (kwh)")
# ax2.set_ylim([0, 3])
ax2.legend(loc='upper right') # 在右上角显示图例

ax2.legend(loc='best') # 在左上角显示图例

ax2.set_xscale("log")
ax2.set_title('(b) Energy consumption')
# 设置坐标范围
# plt.xlim([8, 1500])


ax.set_xlabel('The Number of GPU A100')
ax2.set_xlabel('The Number of GPU A100')
# 显示图像
plt.savefig("graph8.jpg")
