import os
import gc
import datetime
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ── Utils: Safe scaler (no-op when x_scale is False) ───────────────────────────
class NoOpScaler:
    def fit(self, X): return self
    def transform(self, X): 
        # Accept torch.Tensor or np.ndarray
        if isinstance(X, torch.Tensor):
            return X.numpy()
        return X

# ── Dataset & Model definitions ────────────────────────────────────────────────
class GCMCDataset(Dataset):
    def __init__(self, X, Y_log, LOW_logX):
        """
        X        : np.ndarray (features) - 마지막 피처는 이미 log1p 반영됨
        Y_log    : np.ndarray (log1p(y)) - 학습은 log-space에서 진행
        LOW_logX : np.ndarray (log1p(low)) - 필요 시 함께 전달(여기선 학습 입력은 X 전부)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y_log = torch.tensor(Y_log, dtype=torch.float32).unsqueeze(1)
        self.LOW_logX = torch.tensor(LOW_logX, dtype=torch.float32).unsqueeze(1)

    def __len__(self): 
        return len(self.X)

    def __getitem__(self, i): 
        return self.X[i], self.Y_log[i], self.LOW_logX[i]


class GCMCModel(nn.Module):
    def __init__(self, dim, hidden, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, 1)
        )

    def forward(self, x): 
        return self.net(x)  # outputs log(y) in our setup


# ── MC Dropout prediction in log-space ─────────────────────────────────────────
def mc_dropout_predict(model, X_tensor, n_simulations):
    """
    X_tensor: torch.float32 tensor (already scaled)
    Returns: mean_log, std_log (both numpy, shape [N,1])
    """
    model.train()  # enable dropout
    preds = []
    with torch.no_grad():
        for _ in range(n_simulations):
            preds.append(model(X_tensor).cpu().numpy())
    arr = np.array(preds)          # [T, N, 1]
    return arr.mean(axis=0), arr.std(axis=0)  # [N,1], [N,1]


# ── Sampling ───────────────────────────────────────────────────────────────────
def stratified_quantile_sampling(low_values, idx_pool, n_samples, num_bins=10):
    quantiles = pd.qcut(low_values[idx_pool], q=num_bins, labels=False, duplicates='drop')
    idx_sampled = []
    per_bin = max(1, n_samples // num_bins)
    for bin_id in np.unique(quantiles):
        bin_idxs = idx_pool[quantiles == bin_id]
        if len(bin_idxs) == 0: 
            continue
        take = min(per_bin, len(bin_idxs))
        sampled = np.random.choice(bin_idxs, size=take, replace=False)
        idx_sampled.extend(sampled)
    # 모자라면 무작위로 추가
    if len(idx_sampled) < n_samples:
        remain = np.setdiff1d(idx_pool, np.array(idx_sampled, dtype=int))
        if len(remain) > 0:
            extra_take = min(n_samples - len(idx_sampled), len(remain))
            extra = np.random.choice(remain, size=extra_take, replace=False)
            idx_sampled.extend(extra)
    return np.array(idx_sampled, dtype=int)


# ── Training loops (log-space) ─────────────────────────────────────────────────
def train_with_early_stopping(model, optimizer, loss_fn, train_dl, val_dl,
                              scaler_X, epochs, patience):
    best_loss = float('inf')
    patience_ctr = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb_log, _ in train_dl:
            # scale X
            feats = scaler_X.transform(xb)  # np.ndarray
            xb_s = torch.tensor(feats, dtype=torch.float32)
            preds_log = model(xb_s)
            loss = loss_fn(preds_log, yb_log)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        total_val = 0.0
        n_samples = 0
        with torch.no_grad():
            for xb, yb_log, _ in val_dl:
                feats = scaler_X.transform(xb)
                xb_s = torch.tensor(feats, dtype=torch.float32)
                pred_log = model(xb_s)
                batch_loss = loss_fn(pred_log, yb_log).item() * len(xb)
                total_val += batch_loss
                n_samples += len(xb)

        avg_val = total_val / max(1, n_samples)
    #    logging.info(f"[EarlyStop] epoch={epoch} val_loss(log-space)={avg_val:.6f}")

        if avg_val < best_loss:
            best_loss = avg_val
            best_state = model.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logging.info(f"Early stopping at epoch {epoch} (best val loss={best_loss:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)


# ── Main function ──────────────────────────────────────────────────────────────
def main(args):
    # ── Logging ────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    # ── CPU Thread control ──────────────────────────
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_interop_threads)
    logging.info(f"CPU Threads set: num_threads={args.num_threads}, interop_threads={args.num_interop_threads}")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Data Load ───────────────────────
    df = pd.read_csv(args.data)
    mis_lis = ["BIWSEG_ion_b","LETQAE01_ion_b","VEWLAM_clean","ja406030p_si_007_manual",
               "HIHGEM_manual","ja406030p_si_002_manual","POZHUI_ion_b","DONNAW01_SL"]

    if "filename" in df.columns:
        before = len(df)
        df = df[~df["filename"].isin(mis_lis)].reset_index(drop=True)
        logging.info(f"Removed {before - len(df)} rows from mis_lis")

    if "Unnamed: 0" in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    if "name" in df.columns:
        df = df.drop("name", axis=1)

    # Columns: [id?, feat1, feat2, ..., feat_{k-1}, LOW, Y]
    X_all = df.iloc[:, 1:-1].values.astype(np.float32)   # features
    Y_all = df.iloc[:, -1].values.astype(np.float32)     # target (original scale)
    # 마지막 피처(LOW-pressure feature) 추출
    LOW_feat = X_all[:, -1].copy()

    # ── Log transforms ─────────────────────────────────
    # 안정성을 위해 log1p 사용 (0 포함 가능). 필요하면 --log_eps 옵션으로 조정 가능.
    # 1) X 마지막 피처만 log1p
    LOW_log = np.log1p(LOW_feat)
    X_proc = X_all.copy()
    X_proc[:, -1] = LOW_log

    # 2) Y 전체 log1p (학습은 log-space)
    Y_log_all = np.log1p(Y_all)

    # ── Scalers ───────────────────────────────────────
    scaler_X = (StandardScaler().fit(X_proc) if args.x_scale else NoOpScaler().fit(None))

    # ── Pools ─────────────────────────────────────────
    idx_all = np.arange(len(X_proc))
    n_samples_init = int(args.initial_ratio * len(X_proc))
    # 초기 표본은 LOW (원본이 아닌 log 변환된 LOW 사용 권장) — 분포 안정화
    idx_labeled = stratified_quantile_sampling(LOW_log, idx_all, n_samples_init, args.num_bins)
    idx_unlabeled = np.setdiff1d(idx_all, idx_labeled)

    # ── Output dir ────────────────────────────────────
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f"{args.prefix}AL_{ts}"
    os.makedirs(f"{out_dir}/predictions", exist_ok=True)
    metrics = []

    # ── Model ─────────────────────────────────────────
    model = GCMCModel(X_proc.shape[1], args.hidden_dim, args.dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()  # log-space MSE

    # ── Active Learning loop (Pure Uncertainty) ──────
    n_target = int(args.target_ratio * len(X_proc))
    max_iters = (n_target - len(idx_labeled)) // args.samples_per_iter + 1

    for it in range(max_iters):
        logging.info(f"Iteration {it} | labeled={len(idx_labeled)}")

        # Train on current labeled set (log-space)
        d_train = GCMCDataset(X_proc[idx_labeled], Y_log_all[idx_labeled], LOW_log[idx_labeled])
        dl_train = DataLoader(d_train, batch_size=args.batch_size, shuffle=True)
        dl_val = DataLoader(d_train, batch_size=args.batch_size, shuffle=False)  # self-val for early stop
        train_with_early_stopping(model, optimizer, loss_fn, dl_train, dl_val,
                                  scaler_X, args.epochs, args.patience)

        # Evaluate on remaining (metrics in ORIGINAL scale)
        model.eval()
        X_rem_scaled = torch.tensor(scaler_X.transform(X_proc[idx_unlabeled]), dtype=torch.float32)
        y_true_rem = Y_all[idx_unlabeled]  # original scale
        preds_mean_log, _ = mc_dropout_predict(model, X_rem_scaled, args.mcd_n)
        y_pred_rem = np.expm1(preds_mean_log).flatten()  # back to original scale

        r2 = r2_score(y_true_rem, y_pred_rem)
        mae = mean_absolute_error(y_true_rem, y_pred_rem)
        mse = mean_squared_error(y_true_rem, y_pred_rem)
        logging.info(f"[Eval@orig] R2={r2:.4f}, MAE={mae:.6f}, MSE={mse:.6f}")

        metrics.append({"iter": it, "n_labeled": int(len(idx_labeled)), "r2": r2, "mae": mae, "mse": mse})
        pd.DataFrame({"y_true": y_true_rem, "y_pred": y_pred_rem}) \
            .to_csv(f"{out_dir}/predictions/rem_preds_iter_{it}.csv", index=False)

        if len(idx_labeled) >= n_target:
            logging.info("Target ratio reached; stopping AL loop.")
            break

        # Query selection (Pure Active Learning by MC Dropout Uncertainty)
        logging.info("[Query] MC Dropout Uncertainty...")
        unl_ds = GCMCDataset(X_proc[idx_unlabeled], Y_log_all[idx_unlabeled], LOW_log[idx_unlabeled])
        unl_dl = DataLoader(unl_ds, batch_size=args.batch_size, shuffle=False)

        uncertainties = []
        for xb, _, _ in tqdm(unl_dl, desc="Uncertainty Estimation"):
            feats = scaler_X.transform(xb)  # np
            xb_s = torch.tensor(feats, dtype=torch.float32)
            _, stds_log = mc_dropout_predict(model, xb_s, args.mcd_n)
            uncertainties.extend(stds_log.flatten())
        uncertainties = np.array(uncertainties)

        # 상위 N개 불확실한 샘플 선택 (log-space에서의 std 기준)
        take = min(args.samples_per_iter, len(idx_unlabeled))
        uncert_order = np.argsort(uncertainties)[-take:]
        new_idx = idx_unlabeled[uncert_order]

        # Update pools
        idx_labeled = np.concatenate([idx_labeled, new_idx])
        idx_unlabeled = np.setdiff1d(idx_unlabeled, new_idx)

        # GC
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    pd.DataFrame(metrics).to_csv(f"{out_dir}/active_learning_metrics.csv", index=False)
    torch.save(model.state_dict(), f"{out_dir}/final_model.pth")
    logging.info("Done.")


# ── Argparse ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure AL (MC Dropout) for GCMC Surrogate with log-transforms")

    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--initial_ratio", type=float, default=0.01)
    parser.add_argument("--samples_per_iter", type=int, default=10)
    parser.add_argument("--target_ratio", type=float, default=0.30)
    parser.add_argument("--mcd_n", type=int, default=20)
    parser.add_argument("--num_bins", type=int, default=10)

    parser.add_argument("--x_scale", action="store_true", help="Apply StandardScaler to X after log1p on last feature")
    parser.add_argument("--prefix", type=str, required=True)

    parser.add_argument("--num_threads", type=int, default=4,
                        help="Maximum number of threads for PyTorch intraop parallelism")
    parser.add_argument("--num_interop_threads", type=int, default=2,
                        help="Number of threads for PyTorch interop parallelism")

    args = parser.parse_args()
    main(args)

