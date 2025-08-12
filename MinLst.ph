# Minimal LSTM forecaster for time series (next-step prediction)
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----- hyperparams -----
seq_len   = 30
hidden_sz = 32
batch_sz  = 64
epochs    = 10
lr        = 1e-3
device    = "cuda" if torch.cuda.is_available() else "cpu"
g = torch.Generator().manual_seed(42)

# ----- make a toy series (sine with noise) -----
n = 2000
t = torch.linspace(0, 40*math.pi, n)
y = torch.sin(t) + 0.1*torch.randn(n, generator=g)

# turn it into sliding windows
def make_windows(series, seq_len):
    X = torch.stack([series[i:i+seq_len] for i in range(len(series)-seq_len-1)])
    y = torch.stack([series[i+seq_len]    for i in range(len(series)-seq_len-1)])
    return X.unsqueeze(-1), y.unsqueeze(-1)   # [N, seq_len, 1], [N, 1, 1]

X, Y   = make_windows(y, seq_len)
split  = int(0.8*len(X))
X_tr, Y_tr = X[:split], Y[:split]
X_te, Y_te = X[split:], Y[split:]

train_dl = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_sz, shuffle=True)
test_dl  = DataLoader(TensorDataset(X_te, Y_te), batch_size=batch_sz)

# ----- model -----
class LSTMForecaster(nn.Module):
    def __init__(self, in_sz=1, hidden_sz=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_sz, hidden_size=hidden_sz, batch_first=True)
        self.head = nn.Linear(hidden_sz, 1)  # map last hidden → next value

    def forward(self, x):
        # x: [B, T, 1]
        out, _ = self.lstm(x)          # out: [B, T, H]
        last = out[:, -1, :]           # last timestep
        return self.head(last).unsqueeze(1)  # [B, 1, 1]

model = LSTMForecaster(hidden_sz=hidden_sz).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=lr)
lossf = nn.MSELoss()

# ----- train -----
for ep in range(1, epochs+1):
    model.train()
    tr_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = lossf(pred, yb)
        loss.backward()
        opt.step()
        tr_loss += loss.item()*xb.size(0)
    tr_loss /= len(train_dl.dataset)

    # quick eval
    model.eval()
    te_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            te_loss += lossf(model(xb), yb).item()*xb.size(0)
    te_loss /= len(test_dl.dataset)
    print(f"epoch {ep:02d} | train MSE {tr_loss:.5f} | test MSE {te_loss:.5f}")

# ----- use it (one-step forecast on last window) -----
model.eval()
with torch.no_grad():
    last_window = X_te[-1: ].to(device)          # shape [1, seq_len, 1]
    next_val    = model(last_window).cpu().squeeze().item()
print("Next-step prediction:", next_val)
