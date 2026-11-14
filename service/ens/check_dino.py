"""CPU check that GenD-DINOv3 loads from its own checkpoint (the DINOv3 repo on Hugging Face is gated) and scores
2 FF++ real and 2 FF++ Deepfakes validation videos sensibly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import gend  # noqa: E402

torch.set_num_threads(4)
sc = gend.Scorer("dino", device="cpu")
sc.model.float()
print("mean/std", sc.mean.flatten().tolist(), sc.std.flatten().tolist(), flush=True)
for vid in ["FF_REAL__004", "FF_REAL__046", "FF_DF__325_492", "FF_DF__362_354"]:
    z = np.load(rf"D:\dfd-data\crops\{vid}.npz")
    fr = np.unique(z["clip_idx"].ravel())[::10]
    with torch.inference_mode():
        out = sc.model.model(sc.model.feature_extractor(sc.to_tensor(z["face0"][fr])))
    print(vid, out.softmax(-1)[:, 1].numpy().round(3), flush=True)
