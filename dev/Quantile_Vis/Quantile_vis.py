import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def stratified_quantile_sampling(low_values, idx_pool, n_samples, num_bins=10):
    """LOW 값 기준으로 Stratified Quantile Sampling"""
    if n_samples <= 0:
        return np.array([], dtype=int)
    quantiles = pd.qcut(low_values[idx_pool], q=num_bins, labels=False, duplicates="drop")
    idx_sampled = []
    per_bin = max(1, n_samples // num_bins)
    for bin_id in np.unique(quantiles):
        bin_idxs = idx_pool[quantiles == bin_id]
        sampled = np.random.choice(bin_idxs, size=min(per_bin, len(bin_idxs)), replace=False)
        idx_sampled.extend(sampled)
    return np.array(idx_sampled)


def plot_qt_sampling(LOW_values, idx_pool, sampled_idx, num_bins=10, logscale=True, out_path=None):
    """LOW 분포 히스토그램과 QT 샘플링 결과 시각화"""
    vals = LOW_values[idx_pool]
    if logscale:
        vals = np.log1p(vals)  # skew 줄이기

    # Quantile bin 구간
    bin_edges = np.quantile(vals, q=np.linspace(0, 1, num_bins + 1))

    plt.figure(figsize=(8, 5))
    # 전체 분포 히스토그램
    plt.hist(vals, bins=30, alpha=0.6, color="gray", label="Pool Distribution")

    # Quantile bin 경계선 표시
    for edge in bin_edges:
        plt.axvline(edge, color="red", linestyle="--", alpha=0.5)

    # 뽑힌 샘플 표시
    sampled_vals = LOW_values[sampled_idx]
    if logscale:
        sampled_vals = np.log1p(sampled_vals)
    plt.scatter(sampled_vals, np.zeros_like(sampled_vals),
                color="blue", marker="x", s=80, label="Sampled (QT)")

    plt.xlabel("LOW (log scale)" if logscale else "LOW")
    plt.ylabel("Count")
    plt.title(f"QT Sampling Visualization (bins={num_bins})")
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300)
        print(f"[INFO] Figure saved to {out_path}")

    plt.show()


def main(args):
    # 데이터 로드
    df = pd.read_csv(args.data)
    LOW_all = df.iloc[:, -2].values.astype(np.float32)  # 마지막 feature가 LOW라고 가정

    idx_all = np.arange(len(LOW_all))
    sampled_idx = stratified_quantile_sampling(
        np.log10(LOW_all) if args.logscale else LOW_all,
        idx_all,
        args.n_samples,
        args.num_bins
    )

    # 시각화
    plot_qt_sampling(LOW_all, idx_all, sampled_idx,
                     num_bins=args.num_bins,
                     logscale=args.logscale,
                     out_path=args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize QT Sampling on LOW feature")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to draw")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of quantile bins")
    parser.add_argument("--logscale", action="store_true", help="Apply log1p transform to LOW values")
    parser.add_argument("--out", type=str, default=None, help="Path to save figure (e.g., qt_sampling.png)")
    args = parser.parse_args()

    main(args)
