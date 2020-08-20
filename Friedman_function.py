import numpy as np


# 不特殊情况弗里德曼(Friedman检验)

# 这个函数是构造Friedman降序排序矩阵
def trans(a):
    b = []
    c = list(set(a))
    c.sort()
    m = 1
    for i in range(len(a)):
        b.append(a.index(c[i]))
    for i in b:
        a[i] = m
        m += 1
    a = np.array(a).reshape(1, len(a))
    return a


# 这个函数是Nemenyi检验算法性能差异
def algorithm_conspicuousness(a1, a2, a3, a4, a5, cd):
    k1 = np.abs(a1 - a2)
    k2 = np.abs(a1 - a3)
    k3 = np.abs(a1 - a4)
    k4 = np.abs(a1 - a5)
    k5 = np.abs(a2 - a3)
    k6 = np.abs(a2 - a4)
    k7 = np.abs(a2 - a5)
    k8 = np.abs(a3 - a4)
    k9 = np.abs(a3 - a5)
    k10 = np.abs(a4 - a5)
    if k1 > cd:
        print("ELM和IT2_RVFL的性能显著不同")
    else:
        print("ELM和IT2_RVFL的性能没有显著差别")

    if k2 > cd:
        print("ELM和TriT2_RVFL的性能显著不同")
    else:
        print("ELM和TriT2_RVFL的性能没有显著差别")

    if k3 > cd:
        print("ELM和TrapT2_RVFL的性能显著不同")
    else:
        print("ELM和TrapT2_RVFL的性能没有显著差别")

    if k4 > cd:
        print("ELM和TT2_RVFL的性能显著不同")
    else:
        print("ELM和TT2_RVFL的性能没有显著差别")

    if k5 > cd:
        print("T2_RVFL和TriT2_RVFL的性能显著不同")
    else:
        print("IT2_RVFL和TriT2_RVFL的性能没有显著差别")

    if k6 > cd:
        print("IT2_RVFL和TrapT2_RVFL的性能显著不同")
    else:
        print("IT2_RVFL和TrapT2_RVFL的性能没有显著差别")

    if k7 > cd:
        print("IT2_RVFL和TT2_RVFL的性能显著不同")
    else:
        print("IT2_RVFL和TT2_RVFL的性能没有显著差别")

    if k8 > cd:
        print("TriT2_RVFL和TrapT2_RVFL的性能显著不同")
    else:
        print("TriT2_RVFL和TrapT2_RVFL的性能没有显著差别")

    if k9 > cd:
        print("TriT2_RVFL和TT2_RVFL的性能显著不同")
    else:
        print("TriT2_RVFL和TT2_RVFL的性能没有显著差别")

    if k10 > cd:
        print("TrapT2_RVFL和TT2_RVFL的性能显著不同")
    else:
        print("TrapT2_RVFL和TT2_RVFL的性能没有显著差别")

