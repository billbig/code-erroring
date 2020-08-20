import numpy as np


# 四维与四维的爱因斯坦乘积
def tprod(tensor_a, tensor_b):
    # 第四阶张量的个数
    a1 = np.size(tensor_a, 0)
    b1 = np.size(tensor_b, 0)
    # 第三阶张量的个数
    a2 = np.size(tensor_a, 1)
    b2 = np.size(tensor_b, 1)
    # 张量的行数
    a3 = np.size(tensor_a, 2)
    b3 = np.size(tensor_b, 2)
    # 张量的列数
    a4 = np.size(tensor_a, 3)
    b4 = np.size(tensor_b, 3)
    # 处理张量A,张量B,以张量B的高维做新数组temp的行
    # 张量B的低维做新数组temp的列
    temp = np.reshape(tensor_b[0, 0, :, :], (1, b3 * b4), order='F')
    for i in range(b1):
        for j in range(b2):
            temp = np.concatenate((temp, np.reshape(tensor_b[i, j, :, :], (1, b3*b4), order='F')))
    temp = np.delete(temp, 0, 0)  # temp=[b1*b2, b3*b4]
    # 处理张量A,张量B,以张量A的高维做新数组temp1的行
    # 张量A的低维做新数组temp1的列
    temp1 = np.reshape(tensor_a[0, 0, :, :], (1, a3 * a4), order='F')
    for i in range(a1):
        for j in range(a2):
            temp1 = np.concatenate((temp1, np.reshape(tensor_a[i, j, :, :], (1, a3 * a4), order='F')))
    temp1 = np.delete(temp1, 0, 0)  # temp1=[a1*a2, a3*a4]
    temp2 = np.dot(temp, temp1)
    # 重新以列排列构造张量，这就是最终返回的高阶张量
    temp3 = np.reshape(temp2[0, :], (a3, a4), order="f")
    for i in range(b1*b2-1):
        temp3 = np.concatenate((temp3, np.reshape(temp2[i + 1, :], (a3, a4), order="f")))
    temp4 = temp3.reshape((b1, b2, a3, a4))
    return temp4


# 四维与二维的爱因斯坦乘积
def tprod1(tensor_a, tensor_b):
    # 第四阶张量a的个数
    a1 = np.size(tensor_a, 0)
    # 第三阶张量a的个数
    a2 = np.size(tensor_a, 1)
    # 张量a和b的行数
    a3 = np.size(tensor_a, 2)
    b3 = np.size(tensor_b, 0)
    # 张量a和b的列数
    a4 = np.size(tensor_a, 3)
    b4 = np.size(tensor_b, 1)
    # 构造三维全零张量，块数为张量a的列数，
    # 因为最后相乘结果还是张量a的列和列
    # 行数为张量a的行数，列数为张量a的高阶数相乘的结果a1*a2 = b3*b4
    # 三维张量temp=[a4*a3*(a1*a2)],由 a4块 a3行 (a1*a2)列的张量
    temp = np.zeros((a4, a3, a1*a2))
    p = 0
    for k in range(a4):
        for i in range(a1):
            for j in range(a2):
                if p == b3 * b4:
                    p = 0
                temp[k, :, p:] = tensor_a[i, j, :, k].reshape((a3, 1))
                p += 1
    temp1 = np.reshape(tensor_b, (b3*b4, 1), order='F')
    temp2 = np.dot(temp[0, :, :], temp1)
    for index in range(a4):
        temp2 = np.concatenate((temp2, np.dot(temp[index, :, :], temp1)), axis=1)
    temp2 = np.delete(temp2, (0, 0), axis=1)
    return temp2

