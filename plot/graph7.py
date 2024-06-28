import matplotlib.pyplot as plt
import numpy as np

labels = ['float', 'half', 'int8','int4(64)','int4(128)','int4(256)','int4(512)']
power = [19.78, 16.48, 15.01, 13.97, 13.8, 13.76, 13.8]

time = [5.85, 4.12, 3.25, 2.97, 2.92, 2.9, 2.92]

time_communicate = [3.55, 1.82, 1.05, 0.74, 0.71, 0.69, 0.66]

time_compute = [2.3, 2.3, 2.23, 2.23, 2.21, 2.21, 2.26]

fidelity = [1, 0.9999969, 0.99875855, 0.94884747, 0.93450814, 0.91991603, 0.90491968]

x = np.arange(len(labels))  # 标签的位置
width = 0.5  # 柱状图的宽度
# 设置图片大小

width_ = 9.
height_ = width_*3/4
fig, ax1 = plt.subplots(figsize=(width_+1, height_))
# 绘制时间柱状图
ax2 = ax1.twinx()
# ax2.spines['right'].set_position(('outward', 70))
ax2.bar(x, time, width, label='Time of communication (s)', color='tab:cyan', alpha=0.4)
ax2.bar(x, time_compute, width, label='Time   of   computation  (s)', color='tab:red', alpha=0.7)
for i,t in enumerate(time_communicate):
    ax2.text(i, (2.3+t)/2+0.8, t, ha='center', va='bottom', fontsize=16)
# # 在时间柱状图上添加时间数据
for i,t in enumerate(time_compute):
    ax2.text(i, 1., t, ha='center', va='bottom', fontsize=16)

ax2.set_ylim(0, 14);ax2.yaxis.set_ticklabels([]);ax2.set_ylabel(''); ax2.set_yticks([])

# 绘制功耗折线图
ax1.plot(x, power, marker='o', label='Power (kwh)     ',linewidth=4, color='tab:green')
ax1.set_ylabel('Power (kwh)', fontsize=16, color = "green")  # 设置y轴标签字体大小
ax1.set_ylim(min(power)*0.8, max(power) * 1.1)
ax1.tick_params(axis='y', labelcolor='tab:green', labelsize=16)  # 设置y轴刻度字体大小

# 绘制保真度折线图
ax4 = ax1.twinx()
ax4.plot(x, fidelity, marker='s', label='Relative fidelity',linewidth=4, color='tab:orange')
ax4.set_ylabel('Relative fidelity', fontsize=16)  # 设置y轴标签字体大小
ax4.yaxis.label.set_color('darkorange')
ax4.set_ylim(0.70, max(fidelity) * 1.03)
ax4.tick_params(axis='y', labelcolor='tab:orange', labelsize=16)  # 设置y轴刻度字体大小

# 设置x轴标签和标题
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=0, fontsize=16)  # 设置x轴标签字体大小

# 在int4(128)位置添加虚线
ax1.axvline(x=4, color='gray', linestyle='--')

# plt.title('Performance on single multi-node level task', fontsize=16)  # 设置标题字体大小
# ax1.legend(loc='upper left', fontsize=16) 
# ax4.legend(loc='upper left', fontsize=16, bbox_to_anchor=(0, 0.95))
ax2.legend(loc='upper right', fontsize=16, bbox_to_anchor=(1, 1))  # 添加图例
# ax3.legend(loc='upper right', fontsize=16, bbox_to_anchor=(1, 0.95))  # 添加图例
# 调整布局
# fig.tight_layout()

# 保存图像并显示
plt.savefig('graph7.jpg', dpi=800, bbox_inches='tight')
# plt.show()
