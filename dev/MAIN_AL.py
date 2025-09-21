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


# ── Dataset & Model definitions ─────────────────────────────
class GCMCDataset(Dataset):
    def __init__(self, X, Y, LOW):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
        self.LOW = torch.tensor(LOW, dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.X)

    def __getitem__(self, i): return self.X[i], self.Y[i], self.LOW[i]


class GCMCModel(nn.Module):
    def __init__(self, dim, hidden, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, 1)
        )

    def forward(self, x): return self.net(x)


# ── Utility functions ──────────────────────────────────────
def mc_dropout_predict(model, X, n_simulations):
    model.train()  # Enable dropout
    preds = []
    with torch.no_grad():
        for _ in range(n_simulations):
            preds.append(model(X).cpu().numpy())
    arr = np.array(preds)
    return arr.mean(axis=0), arr.std(axis=0)


def stratified_quantile_sampling(low_values, idx_pool, n_samples, num_bins=10):
    quantiles = pd.qcut(low_values[idx_pool], q=num_bins, labels=False, duplicates='drop')
    idx_sampled = []
    per_bin = max(1, n_samples // num_bins)
    for bin_id in np.unique(quantiles):
        bin_idxs = idx_pool[quantiles == bin_id]
        sampled = np.random.choice(bin_idxs, size=min(per_bin, len(bin_idxs)), replace=False)
        idx_sampled.extend(sampled)
    return np.array(idx_sampled)


def train_with_early_stopping(model, optimizer, loss_fn, train_dl, val_dl,
                              scaler_X, epochs, patience, relative):
    best_loss = float('inf')
    patience_ctr = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb, lb in train_dl:
            feats = scaler_X.transform(xb)
            xb = torch.tensor(feats).float()
            y_rel = yb / lb if relative else yb
            preds = model(xb)
            loss = loss_fn(preds, y_rel)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        total_val = 0
        with torch.no_grad():
            for xb, yb, lb in val_dl:
                feats = scaler_X.transform(xb)
                xb = torch.tensor(feats).float()
                y_rel = yb / lb if relative else yb
                total_val += loss_fn(model(xb), y_rel).item()

        avg_val = total_val / len(val_dl.dataset)
        if avg_val < best_loss:
            best_loss = avg_val
            best_state = model.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)


# ── Main function ──────────────────────────────────────────
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
    df = df.drop("name", axis=1)

    X_all = df.iloc[:, 1:-1].values.astype(np.float32)
    Y_all = df.iloc[:, -1].values.astype(np.float32)
    LOW_all = X_all[:, -1]

    idx_all = np.arange(len(X_all))

    scaler_X = StandardScaler().fit(X_all) if args.x_scale else None

    # Initialize pools
    n_samples_init = int(args.initial_ratio * len(X_all))
    idx_labeled = stratified_quantile_sampling(LOW_all, idx_all, n_samples_init, args.num_bins)
    idx_unlabeled = np.setdiff1d(idx_all, idx_labeled)

    # Output dir
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f"{args.prefix}AL_{ts}"
    os.makedirs(f"{out_dir}/predictions", exist_ok=True)
    metrics = []

    # Model
    model = GCMCModel(X_all.shape[1], args.hidden_dim, args.dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Active Learning loop
    n_target = int(args.target_ratio * len(X_all))
    max_iters = (n_target - len(idx_labeled)) // args.samples_per_iter + 1

    for it in range(max_iters):
        logging.info(f"Iteration {it} | labeled={len(idx_labeled)}")

        # Train
        d_train = GCMCDataset(X_all[idx_labeled], Y_all[idx_labeled], LOW_all[idx_labeled])
        dl_train = DataLoader(d_train, batch_size=args.batch_size, shuffle=True)
        dl_val = DataLoader(d_train, batch_size=args.batch_size, shuffle=False)
        train_with_early_stopping(model, optimizer, loss_fn, dl_train, dl_val,
                                  scaler_X, args.epochs, args.patience, args.relative)

        # Evaluate
        model.eval()
        X_rem = torch.tensor(scaler_X.transform(X_all[idx_unlabeled])).float()
        y_true_rem = Y_all[idx_unlabeled]
        preds_mean, _ = mc_dropout_predict(model, X_rem, args.mcd_n)
        y_pred_rem = preds_mean.flatten() * LOW_all[idx_unlabeled] if args.relative else preds_mean.flatten()

        r2 = r2_score(y_true_rem, y_pred_rem)
        mae = mean_absolute_error(y_true_rem, y_pred_rem)
        mse = mean_squared_error(y_true_rem, y_pred_rem)
        logging.info(f"Remaining R2={r2:.4f}, MAE={mae:.4f}, MSE={mse:.4f}")

        metrics.append({"iter": it, "n_labeled": len(idx_labeled), "r2": r2, "mae": mae, "mse": mse})
        pd.DataFrame({"y_true": y_true_rem, "y_pred": y_pred_rem}) \
            .to_csv(f"{out_dir}/predictions/rem_preds_iter_{it}.csv", index=False)

        if len(idx_labeled) >= n_target:
            break

        # Query selection (Pure Active Learning)
        logging.info("[Query] MC Dropout Uncertainty...")

        unl_ds = GCMCDataset(X_all[idx_unlabeled], Y_all[idx_unlabeled], LOW_all[idx_unlabeled])
        unl_dl = DataLoader(unl_ds, batch_size=args.batch_size, shuffle=False)

        uncertainties = []
        for xb, _, _ in tqdm(unl_dl, desc="Uncertainty Estimation"):
            xb_scaled = torch.tensor(scaler_X.transform(xb)).float()
            _, stds = mc_dropout_predict(model, xb_scaled, args.mcd_n)
            uncertainties.extend(stds.flatten())
        uncertainties = np.array(uncertainties)

        # 상위 N개 불확실한 샘플 선택
        uncert_order = np.argsort(uncertainties)[-args.samples_per_iter:]
        new_idx = idx_unlabeled[uncert_order]

        # labeled/unlabeled set 갱신
        idx_labeled = np.concatenate([idx_labeled, new_idx])
        idx_unlabeled = np.setdiff1d(idx_unlabeled, new_idx)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    pd.DataFrame(metrics).to_csv(f"{out_dir}/active_learning_metrics.csv", index=False)
    torch.save(model.state_dict(), f"{out_dir}/final_model.pth")
    logging.info("Done.")


# ── Argparse ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Active Learning for GCMC Surrogate")

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
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--x_scale", action="store_true")
    parser.add_argument("--prefix", type=str, required = True)
    # 기존 인자들 ...
    parser.add_argument("--num_threads", type=int, default=4,
                        help="Maximum number of threads for PyTorch intraop parallelism")
    parser.add_argument("--num_interop_threads", type=int, default=2,
                        help="Number of threads for PyTorch interop parallelism")

    args = parser.parse_args()
    main(args)


