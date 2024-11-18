import numpy as np

'''
利用QR分解的结果求解Ax=b
    Ax = QRx = b
    Rx = Qt b
有解返回解，无解返回None
'''
def solveEqSys(
    _Q: np.ndarray,
    _R: np.ndarray,
    _bt: np.ndarray,
    _m: int,
    _n: int,
):
    if _bt.ndim != 1 or _bt.shape[0] != _m:
        raise TypeError(f"bt维度错误：{_bt.shape}")
    rt = np.matmul(_bt, _Q) # 按行向量存
    # 由于 m>=n，当向量r的n ~ m-1号位不为0的时候，此方程无解
    if _m > _n and np.all(rt[_n: ] != 0):
        return None
    # 模拟回代法解方程
    xt = np.zeros(_n, dtype=float)
    for i in range(_n-1, -1, -1):
        r_value = rt[i]
        for j in range(_n-1, i, -1):
            r_value -= _R[i][j] * xt[j]
        xt[i] = r_value / _R[i][i] # rank(A) = n, 不用担心出现_R[i][i] = 0的情况
    return xt.reshape(-1, 1)
    
    
    