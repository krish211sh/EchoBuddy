"""Mood-trend LSTM (PROOF OF CONCEPT on SYNTHETIC sequences).
Input: sequence of per-message emotion scores [sad, anxiety, anger, happy].
Output: probability that the recent trend is concerning.
NOTE: there is no clinical ground truth here. Real validation needs ethically
collected, labelled longitudinal data. Results are NOT evidence of clinical accuracy.
Run: python trend_lstm.py
"""
import json
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42); np.random.seed(42)
T, F = 10, 4


def make(n):
    X = np.random.rand(n, T, F).astype("float32") * 0.4
    y = np.zeros(n, dtype="float32")
    for i in range(n):
        if np.random.rand() < 0.5:  # concerning: negative scores rise over time
            ramp = np.linspace(0, 0.6, T)
            X[i, :, :3] += ramp[:, None] * np.random.rand(3)
            X[i, :, 3] -= ramp * 0.3
            y[i] = 1
    return torch.tensor(np.clip(X, 0, 1)), torch.tensor(y)


class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.l = nn.LSTM(F, 16, batch_first=True); s.o = nn.Linear(16, 1)
    def forward(s, x):
        return s.o(s.l(x)[0][:, -1]).squeeze(-1)


Xtr, ytr = make(2000); Xte, yte = make(500)
net, opt, loss = Net(), None, nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
for ep in range(30):
    opt.zero_grad(); l = loss(net(Xtr), ytr); l.backward(); opt.step()
acc = ((net(Xte) > 0).float() == yte).float().mean().item()
print(f"Synthetic test accuracy: {acc:.3f}")
torch.save(net.state_dict(), "model/trend_lstm.pt")
json.dump({"synthetic_test_accuracy": acc}, open("results/trend_lstm_metrics.json", "w"))
