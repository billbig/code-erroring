import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
"""

这是ELM神经网络的编程
最终在five_function_matplot.py
得出结果

"""
train_nums = 4000
test_nums = 4192
# 隐藏层神经元节点数
L = 25
# 输入层的特征数
n = 8
# 需要加上一个单位矩阵和1/C的乘积，即I/C,使H'H是列满秩
C = [2**(-5), 2**(-10), 2**(-15), 2**(-20)]


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


def all_temph(weight, train, bias):
    # 输出层到激活函数矩阵
    temph = np.dot(train, weight) + bias
    h = sigmoid(temph)
    return h


def out_put(weight, train, bias):
    h = all_temph(weight, train, bias)
    ht = np.transpose(h)
    return h, ht


def elm(x_train, x_test, y_train, y_test):
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
    # print(y_train)
    # print("y_train", y_train)
    # print("*"*50)
    # print("y_test", y_test)
    # print("bias", bias.shape)
    # print("*"*50)
    H, Ht = out_put(weight, x_train, bias1)
    # 创建一个L*L的单位矩阵
    identity = np.identity(L)
    # 重对H重赋值，保证其列满秩
    beta = np.linalg.pinv(np.dot(Ht, H) + identity/C[1])
    # print("beta:", beta.shape)
    beta1 = np.dot(beta, Ht)
    # print(beta1.shape)
    # print("*"*50)
    beta2 = np.dot(beta1, y_train)
    # print("beta2", beta2.shape)
    # 测试集
    bias2 = np.repeat(bias, test_nums, axis=0)
    # print("bias2", bias2.shape)
    test_h = all_temph(weight, x_test, bias2)
    # print("test_h", test_h.shape)
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

