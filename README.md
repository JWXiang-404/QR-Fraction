# QR分解实现

使用Python实现三种QR分解，以及解方程$Ax = b$

## 1. 环境配置

* **系统：**在`Ubuntu-22.04`下测试稳定；建议使用 `MiniConda/Anaconda` 虚拟环境

* **第三方依赖：**

  > python == 3.8
  >
  > numpy == 1.24.4
  >
  > pandas == 2.0.3

## 2. 安装和运行

1. **下载源码**

```

```

2. **安装conda环境以及python依赖**

```shell
conda create -n qr-frac python=3.8
pip install -r requirements.txt
```

3. **在`input/mA.csv`中输入矩阵$A$（支持浮点数）**

如：
$$
A = 
\begin{bmatrix}
1 & 19 & -34 \\
-2 & -5 & 20 \\
2 & 8 & 37
\end{bmatrix}
$$
`mA.csv`输入为：

```
1,19,-34
-2,-5,20
2,8,37
```

4. **在`input/bt.csv`中输入行向量$b^t$（支持浮点数）**

如：
$$
b = 
\begin{bmatrix}
-63 \\
48 \\
129
\end{bmatrix}
$$
`bt.csv`输入为：

```
-63,48,129
```

5. **运行程序**

运行 `python3 main.py -h` 以获取帮助：

```shell
usage: main.py [-h] (-m | -g | -hh)

QR Fraction Method

optional arguments:
  -h, --help         show this help message and exit
  -m, --modified_gs  Use modified Gram-Schmidt
  -g, --givens       Use Givens rotations
  -hh, --household   Use Householder reflections
```

如以Givens方式分解：

```shell
python3 main.py -g
```

输出为：

```
--- QR FRACTION RESULT ---
-Matirx Q:
[[ 0.33333333  0.93333333 -0.13333333]
 [-0.66666667  0.33333333  0.66666667]
 [ 0.66666667 -0.13333333  0.73333333]]
-Matirx R:
[[  3.  15.   0.]
 [  0.  15. -30.]
 [  0.   0.  45.]]
-Result x of (Ax = b):
[[1.]
 [2.]
 [3.]]
```