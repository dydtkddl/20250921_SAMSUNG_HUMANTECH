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


# ── Utility: Early stopping training ─────────────────────────
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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # CPU Thread control
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_interop_threads)
    logging.info(f"CPU Threads set: num_threads={args.num_threads}, interop_threads={args.num_interop_threads}")

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

    X_all = df.iloc[:, 1:-1].values.astype(np.float32)
    Y_all = df.iloc[:, -1].values.astype(np.float32)
    LOW_all = X_all[:, -1]
    idx_all = np.arange(len(X_all))

    scaler_X = StandardScaler().fit(X_all) if args.x_scale else None

    # Initialize pools (Random sampling)
    n_samples_init = int(args.initial_ratio * len(X_all))
    idx_labeled = np.random.choice(idx_all, size=n_samples_init, replace=False)
    idx_unlabeled = np.setdiff1d(idx_all, idx_labeled)

    # Output dir
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f"{args.prefix}AL_Random_{ts}"
    os.makedirs(f"{out_dir}/predictions", exist_ok=True)
    metrics = []

    # Model
    model = GCMCModel(X_all.shape[1], args.hidden_dim, args.dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Active Learning loop (Random sampling)
    n_target = int(args.target_ratio * len(X_all))
    max_iters = (n_target - len(idx_labeled)) // args.samples_per_iter + 1

    for it in range(max_iters + 1):
        logging.info(f"Iteration {it} | labeled={len(idx_labeled)}")

        # Train
        d_train = GCMCDataset(X_all[idx_labeled], Y_all[idx_labeled], LOW_all[idx_labeled])
        dl_train = DataLoader(d_train, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
        dl_val = DataLoader(d_train, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
        train_with_early_stopping(model, optimizer, loss_fn, dl_train, dl_val,
                                  scaler_X, args.epochs, args.patience, args.relative)

        # Evaluate
        model.eval()
        X_rem = torch.tensor(scaler_X.transform(X_all[idx_unlabeled])).float()
        y_true_rem = Y_all[idx_unlabeled]
        preds = model(X_rem).detach().numpy().flatten()
        y_pred_rem = preds * LOW_all[idx_unlabeled] if args.relative else preds

        r2 = r2_score(y_true_rem, y_pred_rem)
        mae = mean_absolute_error(y_true_rem, y_pred_rem)
        mse = mean_squared_error(y_true_rem, y_pred_rem)
        logging.info(f"Remaining R2={r2:.4f}, MAE={mae:.4f}, MSE={mse:.4f}")

        metrics.append({"iter": it, "n_labeled": len(idx_labeled),
                        "r2": r2, "mae": mae, "mse": mse})
        pd.DataFrame({"y_true": y_true_rem, "y_pred": y_pred_rem}) \
            .to_csv(f"{out_dir}/predictions/rem_preds_iter_{it}.csv", index=False)

        # Stop if enough labeled
        if len(idx_labeled) >= n_target:
            break

        # Random query selection
        new_idx = np.random.choice(idx_unlabeled, size=args.samples_per_iter, replace=False)
        idx_labeled = np.concatenate([idx_labeled, new_idx])
        idx_unlabeled = np.setdiff1d(idx_unlabeled, new_idx)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    pd.DataFrame(metrics).to_csv(f"{out_dir}/active_learning_metrics_random.csv", index=False)
    torch.save(model.state_dict(), f"{out_dir}/final_model_random.pth")
    logging.info("Done.")


# ── Argparse ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Sampling for Active Learning in GCMC Surrogate")

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
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--x_scale", action="store_true")
    parser.add_argument("--prefix", type=str, default="[Ar_0.01bar_5bar_Random]")
    parser.add_argument("--num_threads", type=int, default=4,
                        help="Number of threads for PyTorch intraop parallelism")
    parser.add_argument("--num_interop_threads", type=int, default=2,
                        help="Number of interop threads for PyTorch parallelism")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader num_workers (0 = single process)")

    args = parser.parse_args()
    main(args)

