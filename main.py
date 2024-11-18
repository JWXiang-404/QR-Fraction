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
from src.qr_fraction import QRFractor
from src.solve_eq_sys import solveEqSys
import numpy as np
import pandas as pd
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='QR Fraction Method')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-m', '--modified_gs', action='store_const', const=0, dest='method', help='Use modified Gram-Schmidt')
    group.add_argument('-g', '--givens', action='store_const', const=1, dest='method', help='Use Givens rotations')
    group.add_argument('-hh', '--household', action='store_const', const=2, dest='method', help='Use Householder reflections')

    args = parser.parse_args()
    return args.method

def read_csv_to_numpy(file_path):
    df = pd.read_csv(file_path, header=None, dtype=float)
    matrix = df.to_numpy()
    return matrix

if __name__ == "__main__":
    method = parse_arguments()
    mA = read_csv_to_numpy("./input/mA.csv")
    bt = read_csv_to_numpy("./input/bt.csv").flatten()
    qr_fractor = QRFractor(mA, method)
    mQ, mR = qr_fractor.exec()
    # 设置阈值，对于过小数据处理为0
    threshold = 1e-13
    mQ = np.where(np.abs(mQ) < threshold, 0, mQ)
    mR = np.where(np.abs(mR) < threshold, 0, mR)
    # 解方程
    x = solveEqSys(mQ, mR, bt, mA.shape[0], mA.shape[1])
    print("--- QR FRACTION RESULT ---")
    print("-Matirx Q:")
    print(mQ)
    print("-Matirx R:")
    print(mR)
    if x is None:
        print("-No solution to the system (Ax = b)")
    else:
        print("-Result x of (Ax = b):")
        print(x)
    
    
    