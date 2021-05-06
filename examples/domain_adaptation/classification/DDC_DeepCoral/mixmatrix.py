# 由真值和预测值的txt文档得到混淆矩阵

import numpy as np

def mixmatrix():
    # 初始化，a矩阵用来存结果
    a = np.zeros((7,7), dtype = int)

    try:
        f_pred = open('predvalue.txt', mode='r')
        f_true = open('truevalue.txt', mode='r')
        lines_t = f_true.readlines()
        lines_p = f_pred.readlines()
        for i in range(len(lines_t)):
            tt = lines_t[i]
            pp = lines_p[i]
            tt = int(tt[0])
            pp = int(pp[0])
            a[tt][pp] += 1
        print(a)
    finally:
        f_pred.close()
        f_true.close()
