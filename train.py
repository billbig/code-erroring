import numpy as np
from sklearn.model_selection import train_test_split
import ELM
import IT2_RVFL
import TriT2_RVFL
import TrapT2_RVFL
import TT2_RVFL
"""

这是所有神经网络的编程准备的
相同的训练集合测试集
4000个训练集和4192个测试集
"""


def prepare_numbers():
    bank_data_path = "./bank.csv"
    bank_data = np.loadtxt(bank_data_path, delimiter=",", dtype="float")
    bank_data_label = bank_data[:, -1]
    bank_data = np.delete(bank_data, -1, axis=1)
    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(bank_data, bank_data_label, test_size=0.5116)
    return x_train, x_test, y_train, y_test


def five_function_combine():
    x_train, x_test, y_train, y_test = prepare_numbers()
    acc1 = ELM.elm(x_train, x_test, y_train, y_test)
    acc2 = IT2_RVFL.IT2_RVFL(x_train, x_test, y_train, y_test)
    acc3 = TriT2_RVFL.TriT2_RVFL(x_train, x_test, y_train, y_test)
    acc4 = TrapT2_RVFL.TrapT2_RVFL(x_train, x_test, y_train, y_test)
    return acc1, acc2, acc3, acc4


def hh():
    x_train, x_test, y_train, y_test = prepare_numbers()
    acc5 = TT2_RVFL.tt2_rvfl(x_train, x_test, y_train, y_test)
    return acc5


if __name__ == "__main__":
    acc5 = hh()
    print("acc5", acc5)


