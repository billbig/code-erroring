import numpy as np
import random as rd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tprod
"""

这是TT2-RVFL神经网络的编程

"""
shifti = [0.01, 0.05]
train_nums = 4000
test_nums = 4192
# 隐藏层神经元节点数
L = 25
# 输入层的特征数
n = 8
# 需要加上一个单位矩阵和1/C的乘积，即I/C,使H'H是列满秩
C = [2**(-5), 2**(-10), 2**(-15), 2**(-20)]
s = 2.5


# 构造激激活函数
def sigmoid(x):

    return 1.0 / (1 + np.exp(-x))


def all_weight():
    # 生成范围[-1, 1]之间的随机权重
    weight = np.random.uniform(low=-1, high=1, size=(n, L))
    return weight


def all_bias():
    # 随机生成偏置，一个隐藏层节点对应一个偏置
    for i in range(L):
        bias = np.random.uniform(-1, 1, (1, L))
    # 生成偏执矩阵，以隐藏层节点个数L为行，样本数4000为列
    return bias


# 输出层到激活函数矩阵
def all_temph(weight, train, bias):
    temph = np.dot(train, weight) + bias
    h = sigmoid(temph)
    return h


# 上下隶属度函数的输入矩阵，两个矩阵都为 [25*4000]
def membership_function_matrix(train, weight, bias):
    # 第一个矩阵与RVFL的第一个权重矩阵相同
    tempH = np.dot(train, weight) + bias
    # 第二个矩阵
    weight1 = all_weight()
    tempH1 = np.dot(train, weight1) + bias
    return weight1, tempH, tempH1


# 测试集上下隶属度函数
def test_membership_function_matrix(test, weight, weight1, bias):
    # 第一个矩阵与RVFL的第一个权重矩阵相同
    test_tempH = np.dot(test, weight) + bias
    # 第二个矩阵
    test_tempH1 = np.dot(test, weight1) + bias
    return test_tempH, test_tempH1


# 4阶张量矩阵的构造
def tensor_reshape(train, weight, weight1, bias, shifti, tempH, tempH1, train_nums, L):
    # 上隶属度函数值
    train_pro1 = train - shifti[0]
    tempH_pro1 = np.dot(train_pro1, weight) + bias
    tempH_pro2 = np.dot(train_pro1, weight1) + bias
    # 上隶属度函数值
    train_pro2 = train - shifti[1]
    tempL_pro1 = np.dot(train_pro2, weight1) + bias
    tempL_pro2 = np.dot(train_pro2, weight) + bias
    # 不确定加权法解模糊化，按列合并[8000*25]
    H1 = np.concatenate((tempH, tempH1), axis=0)
    H1_ = np.exp(-H1**2) / s
    # 上隶属度函数值按列合并成[8000*25]
    H2 = np.concatenate((tempH_pro1, tempH_pro2), axis=0)
    H2_ = np.exp(-H2**2) / s
    # 下隶属度函数值按列合并成[8000*25]
    H3 = np.concatenate((tempL_pro1, tempL_pro2), axis=0)
    H3_ = np.exp(-H3**2) / s
    # 构造三维数组[25*4000*2]=25块[4000*2]的构造
    # 必须使[8000*25]是按照列的顺序进行转换的，也就是第一列读完，读第二列...
    H1_3 = np.reshape(np.transpose(H1_), (L, train_nums, 2), order='F')
    H2_3 = np.reshape(np.transpose(H2_), (L, train_nums, 2), order='F')
    H3_3 = np.reshape(np.transpose(H3_), (L, train_nums, 2), order='F')
    # 开始合成一个四维张量-->3个25块[4000*2]的数组
    compose1 = np.append(H1_3, H2_3, axis=0)
    compose2 = np.append(compose1, H3_3, axis=0)
    # 最终四阶张量模型
    H = np.reshape(compose2, (3, L, train_nums, 2))
    return H


# RVFL的权重向量构造H[W], 后续需要构造H[w w]满足张量的条件
def train_it2_sigmoidrnd(weight, train, bias):
    sigran_mean = np.array([0.5 + 0.5*rd.random(), 1.5 + 0.5*rd.random()])
    hu = all_temph(weight, train, bias)
    ral_train = train-sigran_mean[1]
    print("ral_train", ral_train.shape)
    print("*"*50)
    h1 = all_temph(weight, ral_train, bias)
    h = hu+h1/(2*(1+hu-h1))
    return h, sigran_mean[1]


# 同上，这里是测试集RVFL的权重向量构造
def test_it2_sigmoidrnd(weight, test, bias, sigran_mean):
    hu = all_temph(weight, test, bias)
    ral_test = test-sigran_mean
    print("ral_test", ral_test.shape)
    print("*"*50)
    h1 = all_temph(weight, ral_test, bias)
    h = hu+h1/(2*(1+hu-h1))
    return h


def mp_tensor_prepare(H):
    siz1 = list(np.shape(H))
    p = 2
    return p


# 求张量的逆  H=[3*25*4000*2]四阶张量 p=2 M=4 ,siz1=[3,25,4000,2]一维数组
def mp_tensor_inverse(p, H):
    # dimen1 = siz1[p:]  # [4000,2]的一维数组
    # dimen2 = siz1[:p]  # [3,25]的一维数组
    # dimen1p = siz1[p]*siz1[p+1]    # 数字8000
    # dimen2p = siz1[p-2]*siz1[p-1]  # 数字75
    # 使 A_matrix按列构造成[75*8000],matlab中一步即可
    # 主要matlab中默认的转换是按列，这里默认reshape是按行
    Hp = np.reshape(H, (3*L, train_nums, p))
    hs = Hp[0].reshape((p*train_nums, 1), order='F')
    for j in range(3*L-1):
        hk = Hp[j+1].reshape((p*train_nums, 1), order='F')
        hs = np.concatenate((hs, hk), axis=1)
    print("A_matrix", hs.shape)
    print("*" * 50)
    # 对张量降维后以便后续SVD操作
    A_matrix = hs
    U, S, Vt = np.linalg.svd(A_matrix)
    # 把U,S,Vt变换到张量的结构方便计算
    # 公式 A+ = H * S+ * Ut  A+ 为张量的M-P逆， S+为张量逆...
    # 把S=[75,]变换为 [2*4000*25*3]
    S = np.diag(1/S)
    zero = np.zeros([(p+1)*L, p*train_nums-(p+1)*L])
    S = np.concatenate((S, zero), axis=1)
    S1 = np.reshape(np.transpose(S), (p*train_nums, L, p+1), order='F')
    S_final = np.reshape(S1, (2, train_nums, L, p+1))
    print("S_tensor", S_final.shape)
    print("*"*50)
    # U-->Ut 再把 Ut[8000*8000]变换为[2*4000*4000*2]，读[8000*8000]时按列读取
    Ut = np.transpose(U)
    U1 = np.reshape(np.transpose(Ut), (p*train_nums, train_nums, p), order='F')
    Ut_final = np.reshape(U1, (p, train_nums, train_nums, p))
    print("Ut_tensor", Ut_final.shape)
    print("*" * 50)
    # 先Vt-->V 然后V[75*75]变换为[3*25*25*3]，读[75*75]时列读取
    V = np.transpose(Vt)
    V1 = np.reshape(np.transpose(V), ((p+1)*L, L, p+1), order='F')
    V_final = np.reshape(V1, (p+1, L, L, p+1))
    print("V_tensor", V_final.shape)
    print("*" * 50)
    # 爱因斯坦积求解4阶张量的M-P逆
    Ztmp = tprod.tprod(V_final, S_final)
    # Ztmp=[2*4000*25*3]
    print("Ztmp", Ztmp)
    # A_MPInv=[2*4000*25*3]
    A_MPInv = tprod.tprod(Ztmp, Ut_final)
    print("*"*50)
    print(A_MPInv)
    return A_MPInv


# 含有RVFL的处理(包含广义逆)
def train_out_put(A_MP, h, y_train, x_train):
    y_train1 = np.repeat(y_train, 2, axis=1)
    # 构成beta1 =np.dot([2*4000*25*3], [4000*2])=[25*3]
    # beta1 = [25*3]
    beta1 = tprod.tprod1(A_MP, y_train1)
    # RVFL的操作
    H1 = np.concatenate((x_train, h), axis=1)
    H1t = np.transpose(H1)
    # 创建一个L*L的单位矩阵
    identity = np.identity(L+n)
    # 重对H重赋值，保证其列满秩
    beta2 = np.linalg.pinv(np.dot(H1t, H1) + identity/C[0])
    # print("beta:", beta.shape)
    beta3 = np.dot(beta2, H1t)
    # print(beta3.shape)
    # print("*"*50)
    # beta4=[33*1]
    beta4 = np.dot(beta3, y_train)
    return beta1, beta4


def test_out_put(H1, h2, beta, beta1):
    # Y_hat = [4192*2]
    Y_hat = tprod.tprod1(H1, beta)
    # Y_hat = [4192 * 1]
    Y_hat1 = np.dot(h2, beta1)
    return Y_hat, Y_hat1


def tt2_rvfl(x_train, x_test, y_train, y_test):
    weight = all_weight()
    print("weight", weight.shape)
    bias = all_bias()
    # 重复上述的bias = [1,25] 4000次，构成[4000, 25]的数组
    bias1 = np.repeat(bias, train_nums, axis=0)
    print("bias1", bias1.shape)
    # 特征处理归一化处理(all 数据)，实例化两个归一化API
    # 特征值归一化
    st = MinMaxScaler(feature_range=(-1, 1))
    x_train = st.fit_transform(x_train)
    x_test = st.transform(x_test)
    print("x_train", x_train.shape)
    # 目标值归一化
    y_train = np.array(y_train).reshape(len(y_train), 1)
    y_test = np.array(y_test).reshape(len(y_test), 1)
    std_label = MinMaxScaler(feature_range=(-1, 1))
    y_train = std_label.fit_transform(y_train)
    y_test = std_label.transform(y_test)
    print("y_train", y_train.shape)
    print("*"*50)
    print("bias", bias.shape)
    print("*"*50)
    weight1, tempH, tempH1 = membership_function_matrix(x_train, weight, bias1)
    print("weight1", weight1.shape)
    H = tensor_reshape(x_train, weight, weight1, bias1, shifti, tempH, tempH1, train_nums, L)
    print("H", H)
    print("*"*50)
    p = mp_tensor_prepare(H)
    A_MP = mp_tensor_inverse(p, H)
    h, sigran_mean = train_it2_sigmoidrnd(weight, x_train, bias1)
    print("h", h.shape)
    print("*" * 50)
    beta, beta1 = train_out_put(A_MP, h, y_train, x_train)
    print("beta", beta.shape)
    print("beta1", beta1.shape)

    # 测试集处理
    bias2 = np.repeat(bias, test_nums, axis=0)
    test_tempH, test_tempH1 = test_membership_function_matrix(x_test, weight, weight1, bias2)
    H1 = tensor_reshape(x_test, weight, weight1, bias2, shifti, test_tempH, test_tempH1, test_nums, L)
    print("H1", H1.shape)
    h1 = test_it2_sigmoidrnd(weight, x_test, bias2, sigran_mean)
    print("h1", h1.shape)
    h2 = np.concatenate((x_test, h1), axis=1)
    Y_hat, Y_hat1 = test_out_put(H1, h2, beta, beta1)
    print("Y_hat", Y_hat.shape)
    print("Y_hat1", Y_hat1.shape)
    # 反归一化，求原来的值,张量的准确率
    Y_hat = (Y_hat[:, 0] + Y_hat[:, 1]) / 2
    y_predict = std_label.inverse_transform(Y_hat.reshape(test_nums, 1))
    y_test1 = std_label.inverse_transform(y_test)
    y_predict = y_predict.flatten()
    y_test1 = y_test1.flatten()
    acc1 = mean_squared_error(y_test1, y_predict)
    acc1 = np.sqrt(acc1)
    # print("acc1", acc1)
    # 反归一化，求原来的值,RVFL的准确率
    y_predict1 = std_label.inverse_transform(Y_hat1)
    y_predict1 = y_predict1.flatten()
    acc2 = mean_squared_error(y_test1, y_predict1)
    acc2 = np.sqrt(acc2)
    # print("acc2", acc2)
    acc = 0.9*acc1 + (1 - 0.9)*acc2
    return acc

