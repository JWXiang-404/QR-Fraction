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
    
    
    