# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
x=24

x1 = [2**32, 2**35, 2**38, 2**41, 2**44, 2**47]
y1 = [22.501283899999997, 21.18074233, 20.33988592, 19.71909155, 19.61246386, 19.55991983]
size1 = 24
size2 = 24

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.2})

x_log2 = np.log2(x1)
ax1.plot(y1, x_log2, marker='o', markersize=10, color="black",linewidth=2.5, linestyle='--')
colors = ["brown", "purple", "red", "green", "orange", "blue"]

# for i, color in enumerate(colors):
#     ax1.plot(y1[i+1], x_log2[i+1], marker='o', markersize=10, color=color)
    
ax1.plot(y1[0], x_log2[0], marker='o', markersize=15, color="brown") 
ax1.plot(y1[1], x_log2[1], marker='o', markersize=15, color="purple") 
ax1.plot(y1[2], x_log2[2], marker='o', markersize=15, color="red") 
ax1.plot(y1[3], x_log2[3], marker='o', markersize=15, color="green") 
ax1.plot(y1[4], x_log2[4], marker='o', markersize=15, color="orange") 
ax1.plot(y1[5], x_log2[5], marker='o', markersize=15, color="blue") 

labels = ['64GB', '512GB', '4TB', '32TB', '256TB', '2PB']
ax1.set_yticks(x_log2)
ax1.set_yticklabels(labels, fontsize=size1)
# a = [19, 20, 21, 22, 23]
labels2 = [ '$10^{20}$', '$10^{21}$', '$10^{22}$', '$10^{23}$']
# ax1.set_xticks(a)
# ax1.set_xticklabels(labels2, fontsize=size1)
ax1.set_xlim(19.5, 23)
a = [20, 21, 22, 23]
ax1.set_xticks(a)
ax1.set_xticklabels(labels2, fontsize=size1)

ax1.set_ylabel('Space Complexity', fontsize=size1)
ax1.set_xlabel('Time Complexity', fontsize=size1)

ax1.scatter( [20.33988592], [np.log2(2**38)],marker='*', facecolors='none', edgecolors='red', s=1400,linewidths=3)
ax1.scatter( [19.71909155],[np.log2(2**41)], marker='*', facecolors='none', edgecolors='green', s=1400,linewidths=3)

# # 
# arrow_legend = mpatches.FancyArrowPatch((0, 0), (1, 1), arrowstyle='->', mutation_scale=20, color='black', label='memory')
# # 
# ax1.add_patch(arrow_legend)
# # 
# ax1.legend(handles=[arrow_legend], loc='upper right', fontsize=x)
# ax1.legend(labels=['(x)', '(y)', '(z)'], title='memory', loc='upper right', fontsize=x)
# 
ax1.plot( [21.18074233, 21.18074233],[35,38], color = "black", linewidth = 2)
ax1.plot( [21.18074233,20.33988592],[38,38], color = "black", linewidth = 2)
ax1.text( y1[1]+0.34,(35+38)/2 -0.05, "memory ↑x8", horizontalalignment= "center", verticalalignment= "bottom", fontsize=size2)
ax1.text( (y1[1]+y1[2])/2-0.3,38+0.6, f"time ↓x{round(10**(y1[1]-y1[2]),1)}", horizontalalignment= "left", verticalalignment= "center", fontsize=size2)

ax1.plot( [20.33988592, 20.33988592, 19.71909155],[38,41, 41], color = "black", linewidth = 2)
ax1.text( y1[2]+0.87, 38+1.5, "memory ↑x8", horizontalalignment= "left", verticalalignment= "bottom", fontsize=size2)
ax1.text( (y1[2]+y1[3])/2 +0.06,(41+44)/2 -1, f"time ↓x{round(10**(y1[2]-y1[3]),1)}", horizontalalignment= "center", verticalalignment= "center", fontsize=size2)

ax1.plot( [y1[3], y1[3], y1[4]],[41, 44, 44], color = "black", linewidth = 2)
ax1.text( y1[3]+1.80,(41+44)/2-0.05, "memory ↑x8", horizontalalignment= "center", verticalalignment= "bottom", fontsize=size2)
ax1.text( (y1[3]+y1[4])/2 +0.2 ,(44+47)/2 -0.8, f"time ↓x{round(10**(y1[3]-y1[4]),1)}", horizontalalignment= "center", verticalalignment= "center", fontsize=size2)

ax1.plot( [21.18074233, 21.18074233],[35,44], color = "black", linewidth = 2, linestyle='--')
# ax1.text( y1[3]+0.34,(41+44)/2-0.05, "memory ↑x8", horizontalalignment= "center", verticalalignment= "bottom", fontsize=size2)
ax1.plot( [ 21.18074233,y1[4]],[44,44], color = "black", linewidth = 2, linestyle='--')
ax1.plot( [ 21.18074233,y1[3]],[41,41], color = "black", linewidth = 2, linestyle='--')
# ax1.plot( [ y1[2],y1[2]],[38,44], color = "black", linewidth = 2, linestyle='--')
ax1.annotate("", xy=(21.18074233, 35), xytext=(21.18074233, 44),arrowprops=dict(arrowstyle="<|-", color="black", mutation_scale=40))
ax1.annotate("", xy=(21.18074233, 35), xytext=(21.18074233, 41),arrowprops=dict(arrowstyle="<|-", color="black", mutation_scale=40))


ax1.annotate("", xy=(20.33988592, 38), xytext=(20.33988592, 41),arrowprops=dict(arrowstyle="<|-", color="black", mutation_scale=40))
ax1.annotate("", xy=(20.33988592, 41), xytext=(19.71909155, 41),arrowprops=dict(arrowstyle="<|-", color="black", mutation_scale=40))
ax1.annotate("", xy=(21.18074233, 35), xytext=(21.18074233, 38),arrowprops=dict(arrowstyle="<|-", color="black", mutation_scale=40))
ax1.annotate("", xy=(21.18074233, 38), xytext=(20.33988592, 38),arrowprops=dict(arrowstyle="<|-", color="black", mutation_scale=40))
ax1.annotate("", xy=(y1[3], 41), xytext=(y1[3], 44),arrowprops=dict(arrowstyle="<|-", color="black", mutation_scale=30))
ax1.annotate("", xy=(y1[3], 44), xytext=(y1[4], 44),arrowprops=dict(arrowstyle="<|-", color="black", mutation_scale=30))

# ax1.annotate("", xy=(20.33988592, 38), xytext=(20.33988592, 41),arrowprops=dict(arrowstyle="<|-", color="black", mutation_scale=20))
# 
df1 = pd.read_csv('./output.csv', nrows=1)
df2 = pd.read_csv('./output.csv', skiprows=1, nrows=1)
df3 = pd.read_csv('./output.csv', skiprows=[0,1], nrows=1)
df4 = pd.read_csv('./output2.csv', nrows=1)
df5 = pd.read_csv('./output2.csv', skiprows=1, nrows=1)
df6 = pd.read_csv('./output2.csv', skiprows=[0,1], nrows=1)

df1 = df1.dropna(axis=1)
df2 = df2.dropna(axis=1)
df3 = df3.dropna(axis=1)
df4 = df4.dropna(axis=1)
df5 = df5.dropna(axis=1)
df6 = df6.dropna(axis=1)

data1 = df1.values.flatten()
data2 = df2.values.flatten()
data3 = df3.values.flatten()
data4 = df4.values.flatten()
data5 = df5.values.flatten()
data6 = df6.values.flatten()



sns.kdeplot(data6, ax=ax2, label='2PB', linewidth=4, linestyle='-')
sns.kdeplot(data5, ax=ax2, label='256TB', linewidth=4, linestyle='-')
sns.kdeplot(data4, ax=ax2, linewidth=4,label='32TB')
sns.kdeplot(data3, ax=ax2, linewidth=4,label='4TB')
sns.kdeplot(data2, ax=ax2, label='512GB', linewidth=4, linestyle='-')
sns.kdeplot(data1, ax=ax2, label='64GB', linewidth=4, linestyle='-')



# 
plt.xlabel('Time Complexity', fontsize=x)
plt.ylabel('Relative Frequency', fontsize=x)
# plt.title('Probability Density Plot')
plt.tick_params(axis='x', labelsize=x)
plt.tick_params(axis='y', labelsize=x)
# 显示图例
plt.legend(fontsize=x, bbox_to_anchor=(1, 0.85))

labels2 = [ '$10^{20}$', '$10^{22}$', '$10^{24}$', '$10^{26}$','$10^{28}$', '$10^{30}$', '$10^{32}$']
# ax1.set_xticks(a)
# ax1.set_xticklabels(labels2, fontsize=size1)
# ax2.set_xlim(19.5, 23)
a = [20, 22, 24, 26,28,30,32]
ax2.set_xticks(a)
ax2.set_xticklabels(labels2, fontsize=24)
# plt.plot([19.55991983, 2], [0, 2], linestyle='--', color='black', linewidth=1.0)
# plt.axis('off')
labels3 = [ '      0.0', '0.1', '0.2', '0.3','0.4', '0.5']
# ax1.set_xticks(a)
# ax1.set_xticklabels(labels2, fontsize=size1)
# ax2.set_xlim(19.5, 23)
a = [0.0, 0.1, 0.2, 0.3,0.4,0.5]
ax2.set_yticks(a)
ax2.set_yticklabels(labels3, fontsize=size1)
plt.show()
plt.show()

plt.savefig('graph2.jpg', dpi=1200, bbox_inches='tight')