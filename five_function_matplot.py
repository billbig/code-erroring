from matplotlib import pyplot as plt
import numpy as np
import train
import Friedman_function
nums = 50
k = 5
plt.rcParams['font.sans-serif'] = ['SimHei']
elm_list = []
it2_rvfl_list = []
trit2_rvfl_list = []
trapt2_rvfl_list = []
list1 = [0.0471541, 0.046231523, 0.04482354, 0.0481552, 0.04636321, 0.046725635, 0.0456894, 0.046363925, 0.04453115, 0.046464125]
list2 = [0.0473102, 0.045023508, 0.04418386, 0.0491962, 0.04563118, 0.046246278, 0.04576096, 0.044769802, 0.04595632, 0.04609426]
list3 = [0.0452843, 0.048231632, 0.04763432, 0.0455860, 0.04689333, 0.045266067, 0.04702895, 0.045006090, 0.04735318, 0.048980448]
list4 = [0.0448194, 0.046288152, 0.0446255, 0.0439655, 0.044586321, 0.04627235, 0.047257898, 0.04686495, 0.04500311, 0.04764129]
list5 = [0.0465055, 0.044001537, 0.04693046, 0.04625052, 0.04786028, 0.04355031, 0.042996992, 0.04505816, 0.04650216, 0.04452042]
tt2_rvfl_list = list1 + list2 + list3 + list4 + list5
index = 0
for i in range(50):
    acc1, acc2, acc3, acc4 = train.five_function_combine()
    elm_list.append(acc1)
    it2_rvfl_list.append(acc2)
    trit2_rvfl_list.append(acc3)
    trapt2_rvfl_list.append(acc4)
    index += 1
    print(index)
y_elm_list = elm_list
y_it2_rvfl_list = it2_rvfl_list
y_trit2_rvfl_list = trit2_rvfl_list
y_trapt2_rvfl_list = trapt2_rvfl_list
y_tt2_rvfl_list = tt2_rvfl_list
#制表#
y_elm_mean = np.mean(elm_list)
y_elm_std = np.std(elm_list)
y_it2_rvfl_mean = np.mean(it2_rvfl_list)
y_it2_rvfl_std = np.std(it2_rvfl_list)
y_trit2_rvfl_mean = np.mean(trit2_rvfl_list)
y_trit2_rvfl_std = np.std(trit2_rvfl_list)
y_trapt2_rvfl_mean = np.mean(trapt2_rvfl_list)
y_trapt2_rvfl_std = np.std(trapt2_rvfl_list)
y_tt2_rvfl_mean = np.mean(tt2_rvfl_list)
y_tt2_rvfl_std = np.std(tt2_rvfl_list)
print("ELM Mean", y_elm_mean)
print("ELM Std", y_elm_std)
print("IT2-RVFL Mean", y_it2_rvfl_mean)
print("IT2-RVFL Std", y_it2_rvfl_std)
print("TriT2-RVFL Mean", y_trit2_rvfl_mean)
print("TriT2-RVFL Std", y_trit2_rvfl_std)
print("TrapT2-RVFL Mean", y_trapt2_rvfl_mean)
print("TrapT2-RVFL Std", y_trapt2_rvfl_std)
print("TT2-RVFL Mean", y_tt2_rvfl_mean)
print("TT2_RVFL Std", y_tt2_rvfl_std)
x = list(range(1, 51))
# 设置图形大小
plt.figure(figsize=(5, 3), dpi=780)
# # 绘制散点图
plt.scatter(x, y_elm_list, s=9, label="ELM测试集根均方差", color="orange")
plt.scatter(x, y_it2_rvfl_list, s=9, label="IT2_RVFL测试集根均方差", color="blue")
plt.scatter(x, y_trit2_rvfl_list, s=9, label="TriT2_RVFL测试集根均方差", color="green")
plt.scatter(x, y_trapt2_rvfl_list, s=9, label="TrapT2_RVFL测试集根均方差", color="red")
plt.scatter(x, y_tt2_rvfl_list,  s=9, label="TT2_RVFL测试集根均方差", color="black")
# 调整x轴的刻度
plt.xticks(fontsize=6.5)
plt.yticks(fontsize=6.5)
# 添加图例 loc图例位置
plt.legend(loc=2, fontsize=6.5)
plt.grid(alpha=0.2)
# 添加描述信息
plt.xlabel("训练次数", fontsize=7.5)
plt.ylabel("根均方差", fontsize=7.5)
# plt.title("50次测试集不同数学模型的RMSE", fontsize=20)
plt.savefig("./picture.svg", dpi=780)
plt.savefig("./picture1.png", dpi=780)
plt.show()
# ***************************************************************** #
"""
弗里德曼(Friedman检验)
"""
# 合并这5种算法的50次根均方差(MSE)
# 即k=5,N=50
elm_list = np.array(elm_list).reshape(nums, 1)
it2_rvfl_list = np.array(it2_rvfl_list).reshape(nums, 1)
trit2_rvfl_list = np.array(trit2_rvfl_list).reshape(nums, 1)
trapt2_rvfl_list = np.array(trapt2_rvfl_list).reshape(nums, 1)
tt2_rvfl_list = np.array(tt2_rvfl_list).reshape(nums, 1)
Friedman_list = np.concatenate((elm_list, it2_rvfl_list, trit2_rvfl_list, trapt2_rvfl_list, tt2_rvfl_list), axis=1)
print(Friedman_list)
a = Friedman_function.trans(list(Friedman_list[0, :]))
for i in range(nums-1):
    a = np.concatenate((a, Friedman_function.trans(list(Friedman_list[i + 1, :]))), axis=0)
print(a)
a1 = np.mean(a[:, 0])
a2 = np.mean(a[:, 1])
a3 = np.mean(a[:, 2])
a4 = np.mean(a[:, 3])
a5 = np.mean(a[:, 4])
print(a1, a2, a3, a4, a5)
R = np.square(a1)+np.square(a2)+np.square(a3)+np.square(a4)+np.square(a5)
print(R)
Tx2 = 12*nums / (k*(k+1))
Tx2 = Tx2*(R-(k*(k+1)**2/4))
print(Tx2)
Tf = (nums-1)*Tx2
Tf = Tf / (nums*(k-1)-Tx2)
print(Tf)
# 查看F分布临界值表a=0.05时竖列(k-1)*(N-1)=4*49=196 横行 k-1=4 查表 F=2.37
F = 2.37
if Tf > F:
    print("所有算法的性能都相同这个假设被拒绝，说明算法性能显著不同")
else:
    print("所有算法的性能都相同")
"""

Nemenyi检验

"""
# 查看Nemenyi检验中常用的qa值,当k=5,a=0.05时, 查出qa=2.728
qa = 2.728
CD = qa * np.sqrt(k*(k + 1)/(6*nums))
print(CD)
# 算法k个需要比较k*(k-1)/2即5个算法比较10次
Friedman_function.algorithm_conspicuousness(a1, a2, a3, a4, a5, CD)
"""

绘制Friedman图

"""
rank_x = list(map(lambda x: np.mean(x), a.T))
name_y = ["ELM", "IT2_RVFL", "TriT2_RVFL", "TrapT2_RVFL", "TT2_RVFL"]
min_ = [x - CD/2 for x in rank_x]
max_ = [x + CD/2 for x in rank_x]
# 设置图形大小
plt.figure(figsize=(5, 3), dpi=780)
# 调整x轴的刻度
plt.xticks(fontsize=8)
plt.yticks(fontsize=7)
# plt.title("Friedman检验", fontsize=22)
plt.scatter(rank_x, name_y, s=9.5)
plt.hlines(name_y, min_, max_)
plt.grid(alpha=0.2)
plt.savefig("./picture2.png", dpi=780)
plt.show()
# 图所示是Friedman检验图，可以看出，算法A与B的临界值域横线段有交叠，
# 表示没有显著差别，算法A与C横线段没有交叠区域，表示A优于C
"""

保存numpy中的数据为txt格式

"""
np.savetxt("bank_success_data", Friedman_list, fmt='%.8e')
