import os
import argparse
import logging
import numpy as np
import pandas as pd
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_interop_threads)

    # ── Data Load ───────────────────────
    df = pd.read_csv(args.data)

    # ⬇️ mis_lis 필터링 + 불필요 컬럼 제거
    mis_lis = [
        "BIWSEG_ion_b","LETQAE01_ion_b","VEWLAM_clean","ja406030p_si_007_manual",
        "HIHGEM_manual","ja406030p_si_002_manual","POZHUI_ion_b","DONNAW01_SL"
    ]
    if "filename" in df.columns:
        before = len(df)
        df = df[~df["filename"].isin(mis_lis)].reset_index(drop=True)
        logging.info(f"Removed {before - len(df)} rows from mis_lis")

    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    if "name" in df.columns:
        df = df.drop("name", axis=1)

    X_all = df.iloc[:, 1:-1].values.astype(np.float32)
    Y_all = df.iloc[:, -1].values.astype(np.float32)
    LOW_all = X_all[:, -1]

    scaler_X = StandardScaler().fit(X_all) if args.x_scale else None

    # ── Train/Test Split ───────────────────────
    n_train = int(args.train_ratio * len(X_all))
    idx_all = np.arange(len(X_all))
    idx_train = np.random.choice(idx_all, size=n_train, replace=False)
    idx_test = np.setdiff1d(idx_all, idx_train)

    logging.info(f"Train size={len(idx_train)}, Test size={len(idx_test)}")

    d_train = GCMCDataset(X_all[idx_train], Y_all[idx_train], LOW_all[idx_train])
    d_val = d_train  # 여기서는 별도 검증셋 없이 train 재사용
    d_test = GCMCDataset(X_all[idx_test], Y_all[idx_test], LOW_all[idx_test])

    dl_train = DataLoader(d_train, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers)
    dl_val = DataLoader(d_val, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    # ── Model ───────────────────────
    model = GCMCModel(X_all.shape[1], args.hidden_dim, args.dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Train
    train_with_early_stopping(model, optimizer, loss_fn, dl_train, dl_val,
                              scaler_X, args.epochs, args.patience, args.relative)

    # Test
    model.eval()
    X_test = torch.tensor(scaler_X.transform(X_all[idx_test])).float()
    y_true_test = Y_all[idx_test]
    preds = model(X_test).detach().numpy().flatten()
    y_pred_test = preds * LOW_all[idx_test] if args.relative else preds

    r2 = r2_score(y_true_test, y_pred_test)
    mae = mean_absolute_error(y_true_test, y_pred_test)
    mse = mean_squared_error(y_true_test, y_pred_test)

    logging.info(f"[Test Results] R2={r2:.4f}, MAE={mae:.4f}, MSE={mse:.4f}")

    # Save
    out_dir = f"{args.prefix}_train{int(args.train_ratio*100)}"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"y_true": y_true_test, "y_pred": y_pred_test}) \
        .to_csv(f"{out_dir}/test_predictions.csv", index=False)
    torch.save(model.state_dict(), f"{out_dir}/final_model.pth")
    logging.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Split Training for GCMC Surrogate")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--train_ratio", type=float, default=0.1, help="Train set ratio (0.1=10%, 0.2=20%)")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--x_scale", action="store_true")
    parser.add_argument("--prefix", type=str, default="[Ar_single_split]")
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--num_interop_threads", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    main(args)
