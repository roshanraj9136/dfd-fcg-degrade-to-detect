"""CPU smoke test of train_gend.py for PE: trainable tensors, gradient checkpointing, one forward/backward."""
import sys
from pathlib import Path

sys.argv = ["train_gend.py", r"D:\dfd-data\ens\smoke_pe", "1", "1e-4", "pe"]
sys.path[:0] = [str(Path(__file__).resolve().parent)]
import torch  # noqa: E402

import train_gend as T  # noqa: E402

T.NAME = "pe"
sc = T.Scorer("pe", device="cpu")
sc.model.float()
params = T.trainable(sc.model)
print("trainable", sum(p.numel() for p in params.values()), "tensors", len(params), flush=True)
enc = sc.model.feature_extractor
enc.backbone.set_grad_checkpointing(True)
sc.model.train()
x = sc.to_tensor((torch.rand(2, 150, 150, 3) * 255).byte().numpy())
f = enc(x)
out = sc.model.model(f)
loss = torch.nn.functional.cross_entropy(out, torch.tensor([0, 1]))
loss.backward()
g = [p.grad is not None and float(p.grad.abs().sum()) > 0 for p in params.values()]
print("loss", float(loss), "grads ok", sum(g), "/", len(g), flush=True)
