"""Check that every head logit in the table equals the sum of its exact terms + constant."""
import sys

import numpy as np

sys.path.insert(0, ".")
from qa.methods import HEAD_IDX, head_terms, p_from_d   # noqa: E402
from qa.table import build                              # noqa: E402

T = build()
terms = head_terms(T)
for h, i in HEAD_IDX.items():
    err = np.abs(terms[h].sum(1) + T["const"][:, i] - T["d"][:, i]).max()
    print(f"head {h}: max |sum(terms)+const - d| = {err:.2e}")
err = np.abs(T["cs_patch"].sum(1) + T["const"][:, 0] - T["d"][:, 0]).max()
print(f"spatial patches: {err:.2e}")
err = np.abs(T["ct_patch"].sum(1) + T["const"][:, 1] - T["d"][:, 1]).max()
print(f"temporal patches: {err:.2e}")
print("P from d vs stored p_final:", np.abs(p_from_d(T["d"].astype(np.float64)) - T["p_final"]).max())
print("decomposition P vs model P (fp16 heads):", np.abs(T["p_final"] - T["model_p"]).max())
print(len(T["vid"]), "clips")
