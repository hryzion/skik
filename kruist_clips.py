import numpy as np
from scipy.stats import kruskal
import argparse
import glob
import os

def main():
    parser = argparse.ArgumentParser(description="Kruskal-Wallis test for CLIP scores")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Folder containing .npy files for each method"
    )
    parser.add_argument(
        "--pattern", type=str, default="*.npy",
        help="Filename pattern to match, default '*.npy'"
    )
    args = parser.parse_args()

    # 找到所有符合条件的文件
    files = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    if len(files) < 2:
        print("Need at least 2 .npy files for Kruskal-Wallis test!")
        return

    method_names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    print(f"Found methods: {method_names}")

    # 读入数据
    data_list = []
    for f in files:
        scores = np.load(f)
        data_list.append(scores)
        print(f"{os.path.basename(f)}: n={len(scores)}, mean={scores.mean():.4f}, std={scores.std():.4f}")

    # Kruskal-Wallis H-test
    H_statistic, p_value = kruskal(*data_list)
    print("\n=== Kruskal-Wallis Test Result ===")
    print(f"H = {H_statistic:.4f}")
    print(f"p = {p_value:.6f}")

    if p_value < 0.05:
        print("Result: Significant differences exist between groups (reject H0)")
    else:
        print("Result: No significant difference between groups (fail to reject H0)")

if __name__ == "__main__":
    main()
