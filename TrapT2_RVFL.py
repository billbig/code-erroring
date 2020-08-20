import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
train_nums = 4000
test_nums = 4192
# 隐藏层神经元节点数
L = 25
# 输入层的特征数
n = 8
# 需要加上一个单位矩阵和1/C的乘积，即I/C,使H'H是列满秩
C = [2**(-5), 2**(-10), 2**(-15), 2**(-20)]


# 构造激活函数
def sigmoid(x):

    return 1.0 / (1 + np.exp(-x))


def all_temph(weight, train, bias):
    # 输出层到激活函数矩阵
    temph = np.dot(train, weight) + bias
    h = sigmoid(temph)
    return h


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


#
def TrapT2_sigmoidrnd(weight, train, bias, numbers):
    hu = all_temph(weight, train, bias)
    # 四个梯形模糊集
    ral1_train = train-1
    ral2_train = train-2
    ral3_train = np.ones((numbers, n))
    ral4_train = np.zeros((numbers, n))
    # print("ral1_train", ral1_train.shape)
    # print("ral12_train", ral2_train.shape)
    # print("ral13_train", ral3_train.shape)
    # print("ral14_train", ral4_train.shape)
    # print("*"*50)
    h1 = all_temph(weight, ral1_train, bias)
    h2 = all_temph(weight, ral2_train, bias)
    h3 = all_temph(weight, ral3_train, bias)
    h4 = all_temph(weight, ral4_train, bias)
    hm = hu[:, 0].reshape(numbers, 1)
    # 处理hu, h1, h2
    for i in range(L):
        hu_ = hu[:, i].reshape(numbers, 1)
        h1_ = h1[:, i].reshape(numbers, 1)
        h2_ = h2[:, i].reshape(numbers, 1)
        h3_ = h3[:, i].reshape(numbers, 1)
        h4_ = h4[:, i].reshape(numbers, 1)
        if i == 0:
            hm = np.concatenate((hm, h1_), axis=1)
            hm = np.concatenate((hm, h2_), axis=1)
            hm = np.concatenate((hm, h3_), axis=1)
            hm = np.concatenate((hm, h4_), axis=1)
        else:
            hm = np.concatenate((hm, hu_), axis=1)
            hm = np.concatenate((hm, h1_), axis=1)
            hm = np.concatenate((hm, h2_), axis=1)
            hm = np.concatenate((hm, h3_), axis=1)
            hm = np.concatenate((hm, h4_), axis=1)
    Hpp = hm
    H = np.concatenate((train, Hpp), axis=1)
    return H


def last_control(H, y_train):
    # 创建一个(L*5+n)*(L*5+n)的单位矩阵
    identity = np.identity(L*5+n)
    Ht = np.transpose(H)
    beta = np.linalg.pinv(np.dot(Ht, H) + identity/C[1])
    beta1 = np.dot(beta, Ht)
    beta2 = np.dot(beta1, y_train)
    # print("beta1:", beta2.shape)
    error = np.dot(H, beta2) - y_train  # [4000*1]
    s = np.median(np.abs(error))/0.6745
    w = np.zeros(list(error.shape))  # [4000*1]
    a = 2.5
    b = 3
    standard = np.abs(error/s)
    for i in range(train_nums):
        if standard[i, 0] <= a:
            w[i, :] = 1
        elif (standard[i, 0] > a) & (standard[i, 0] < b):
            w[i, :] = (b - np.abs(error[i, 0]/s)) / (b - a)
        else:
            w[i, :] = np.exp(-4)
    # print("w", w.shape)
    return w


# 含有RVFL的处理
def out_put(w, H, y_train):
    Hp = np.repeat(np.sqrt(w), 5*L+n, axis=1) * H
    Yp = np.sqrt(w) * y_train
    Hpt = np.transpose(Hp)
    return Hp, Hpt, Yp


def TrapT2_RVFL(x_train, x_test, y_train, y_test):
    weight = all_weight()
    # print("weight", weight.shape)
    bias = all_bias()
    # 重复上述的bias = [1,25] 4000次，构成[4000, 25]的数组
    bias1 = np.repeat(bias, train_nums, axis=0)
    # print("bias1", bias1.shape)
    # print("x_train", x_train.shape)
    # 特征处理归一化处理(all 数据)，实例化两个归一化API
    # 特征值归一化
    st = MinMaxScaler(feature_range=(-1, 1))
    x_train = st.fit_transform(x_train)
    x_test = st.transform(x_test)
    # 目标值归一化
    y_train = np.array(y_train).reshape(len(y_train), 1)
    y_test = np.array(y_test).reshape(len(y_test), 1)
    std_label = MinMaxScaler(feature_range=(-1, 1))
    y_train = std_label.fit_transform(y_train)
    y_test = std_label.transform(y_test)
    # print("y_train", y_train.shape)
    # print("*"*50)
    # print("bias", bias.shape)
    # print("*"*50)
    h = TrapT2_sigmoidrnd(weight, x_train, bias1, train_nums)
    # print("H", h.shape)
    w = last_control(h, y_train)
    H, Ht, y_train1 = out_put(w, h, y_train)
    # 创建一个L*L的单位矩阵
    identity = np.identity(5*L+n)
    # 重对H重赋值，保证其列满秩
    beta = np.linalg.pinv(np.dot(Ht, H) + identity/C[1])
    # print("*" * 50)
    # print("beta:", beta.shape)
    beta1 = np.dot(beta, Ht)
    # print("beta1", beta1.shape)
    beta2 = np.dot(beta1, y_train1)
    # print("beta2", beta2.shape)
    # print("*" * 50)

    # 测试集
    bias2 = np.repeat(bias, test_nums, axis=0)
    # print("bias2", bias2.shape)
    test_h = TrapT2_sigmoidrnd(weight, x_test, bias2, test_nums)
    # print("x_test_ral", test_h.shape)
    y_predict = np.dot(test_h, beta2)
    # 反归一化，求原来的值
    y_predict = std_label.inverse_transform(y_predict)
    y_test1 = std_label.inverse_transform(y_test)

    y_predict = y_predict.flatten()
    y_test1 = y_test1.flatten()
    # print("y_predict", y_predict)
    # print("y_test1", y_test1)
    acc = mean_squared_error(y_test1, y_predict)
    acc = np.sqrt(acc)
    return acc

