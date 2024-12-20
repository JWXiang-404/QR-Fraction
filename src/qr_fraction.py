'''
Copyright (c) 2024 Xiang Jiawei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import numpy as np

class QRFractor:
    def __init__(
        self,
        matrix_A: np.ndarray,
        op: int
    ):
        self.matrix_A = matrix_A
        if matrix_A.ndim != 2:
            raise TypeError(f"matrix_A维度错误：{matrix_A.shape}")
        self.A_m = matrix_A.shape[0]
        self.A_n = matrix_A.shape[1]
        self.op = op
        if self.A_m < self.A_n or self.A_n != np.linalg.matrix_rank(matrix_A):
            raise ValueError(f"A矩阵不满足 rank(A)=n 条件")
        elif op == 2 and self.A_m != self.A_n:
            raise ValueError(f"使用Householder方法，A矩阵不满足 满秩方阵 条件")

    '''
    QR-fraction with Modefied Gram-Schmidt Algorithm
    '''
    def _qrMGSFraction(self): 
        # 将列向量操作换成对行向量的操作，便于计算和存储
        At = np.transpose(self.matrix_A)
        Qt = np.zeros((self.A_n, self.A_m), dtype=float)
        matrix_R = np.zeros((self.A_n, self.A_n), dtype=float)
        # 初始化 k = 0
        matrix_R[0][0] = np.linalg.norm(At[0])
        Qt[0] = At[0] / matrix_R[0][0]
        for i in range(1, self.A_n):
            Qt[i] = At[i]
        # k > 1
        for k in range(1, self.A_n):
            for j in range(k, self.A_n):
                # uj <- Ek * uj
                alpha = np.dot(Qt[k-1], Qt[j])
                matrix_R[k-1][j] = alpha
                Qt[j] = Qt[j] - alpha * Qt[k-1]
            # uk = uk / ||uk||
            matrix_R[k][k] = np.linalg.norm(Qt[k])
            Qt[k] = Qt[k] / matrix_R[k][k]
        # 转置得Q-
        matrix_Q = np.transpose(Qt)
        return matrix_Q, matrix_R
    
    def _makePlaneRotationMatirx(
        self,
        idx_i: int,
        idx_j: int,
        num_i: float,
        num_j: float
    ):
        re_matirx = np.eye(self.A_m)
        c = num_i / ((num_i ** 2 + num_j ** 2) ** 0.5)
        s = num_j / ((num_i ** 2 + num_j ** 2) ** 0.5)
        re_matirx[idx_i][idx_i] = re_matirx[idx_j][idx_j] = c
        re_matirx[idx_i][idx_j] = s
        re_matirx[idx_j][idx_i] = -s
        return re_matirx
    
    def _qrGivensFraction(self):
        # 初始化
        matrix_R = self.matrix_A.copy()
        Qt = np.eye(self.A_m)
        # 主循环
        for i in range(self.A_n):
            for j in range(i+1, self.A_m):
                if matrix_R[j][i] != 0:
                    t_martix = self._makePlaneRotationMatirx(i, j, matrix_R[i][i], matrix_R[j][i])
                    Qt = np.matmul(t_martix, Qt)
                    matrix_R = np.matmul(t_martix, matrix_R)
        # 返回
        matrix_Q = np.transpose(Qt)
        return matrix_Q, matrix_R
        
    def _qrHouseholdFraction(self):
        # 初始化
        matrix_R = self.matrix_A.copy()
        Qt = np.eye(self.A_m)
        # 主循环
        for i in range(self.A_n - 1):
            tmp_u = np.copy(matrix_R[i:, i]).reshape(-1, 1)
            tmp_u[0][0] -= np.linalg.norm(tmp_u)
            tmp_Rhat = np.eye(self.A_m - i) - (2 / (np.linalg.norm(tmp_u) ** 2)) * np.matmul(tmp_u, np.transpose(tmp_u))
            matrix_R[i:, i:] = np.matmul(tmp_Rhat, matrix_R[i:, i:])
            # 将R^拼接成R
            tmp_R = np.eye(self.A_m)
            tmp_R[i:, i:] = tmp_Rhat
            # 计算Qt
            Qt = np.matmul(tmp_R, Qt)
        # 返回
        matrix_Q = np.transpose(Qt)
        return matrix_Q, matrix_R
    
    def exec(self):
        if   (self.op == 0): return self._qrMGSFraction()
        elif (self.op == 1): return self._qrGivensFraction()
        elif (self.op == 2): return self._qrHouseholdFraction()           
        
    
    
    