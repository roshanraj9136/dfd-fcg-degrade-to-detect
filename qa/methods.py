r"""
Decision rules compared in the study. Every rule outputs a clip-level P(FAKE); the
video-level score is the mean over clips (as in the DFD-FCG test protocol).

Notation (per head h in {spatial s, temporal t, fusion a}):
    c_h   exact per-term contributions (qa/engine.py):  d_h = sum_k c_h[k] + const_h
          s: 24 layers x 4 facial queries = 96 terms
          t: 24 layers                    = 24 terms
          a: 96 spatial + 24 temporal     = 120 terms
    P_final = mean_h sigma(d_h)

Rules
  original     the released model
  platt        Platt scaling of logit(P_final) on clean fit videos                 (needs fakes)
  qshift       ridge from face-crop quality descriptors to the head logits of REAL fit videos,
               subtract the predicted offset                                         (reals only)
  en           EVIDENCE NULLING (ours). Learn, from paired clean/degraded REAL videos, how
               degradation moves every exact attribution term as a function of
               BACKGROUND-referenced quality (pixels outside the face, which a face
               manipulation never touches):  Delta c ~= W Delta phi(q_bg).
               At test time remove that predicted evidence:
                   c' = c - W (phi(q_bg) - mean phi_clean-real),   d'_h = sum_k c'_h[k] + const_h
               Reals only; no fake video and no clean reference of the test video are needed.
  en-face      ablation: the same with face-crop quality (entangled with the forgery)
  en-proj      ablation: project the attribution onto the top drift direction and remove it
               (no quality signal; shows that attribution alone cannot separate the two)
  refit        logistic regression on all 240 terms, trained on fit videos under every
               training condition                                                    (needs fakes)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


def head_terms(T):
    return {
        "s": T["cs_lq"].reshape(len(T["cs_lq"]), -1).astype(np.float64),
        "t": T["ct_l"].astype(np.float64),
        "a": np.concatenate([T["ca_lq"].reshape(len(T["ca_lq"]), -1), T["ca_tl"]], 1).astype(np.float64),
    }


HEAD_IDX = {"s": 0, "t": 1, "a": 2}
HEADS = "sta"


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def p_from_d(d):
    """d: (N,3) head logits -> P_final."""
    return sig(d).mean(1)


def pairs(T, mask):
    """(src, dst): clean and degraded rows of the same video and clip, both inside mask."""
    key = {(T["vid"][i], T["clip"][i]): i for i in np.where(mask & (T["cond"] == "clean"))[0]}
    src, dst = [], []
    for i in np.where(mask & (T["cond"] != "clean"))[0]:
        j = key.get((T["vid"][i], T["clip"][i]))
        if j is not None:
            src.append(j)
            dst.append(i)
    return np.array(src, int), np.array(dst, int)


def paired_drift(T, terms, mask):
    src, dst = pairs(T, mask)
    return {h: terms[h][dst] - terms[h][src] for h in terms}, src, dst


class Original:
    name = "original"

    def fit(self, T, fit_mask):
        return self

    def predict(self, T, mask):
        return T["p_final"][mask]


class Platt:
    name = "platt"

    def fit(self, T, fit_mask):
        m = fit_mask & (T["cond"] == "clean")
        self.lr = LogisticRegression(C=1e3).fit(logit(T["p_final"][m])[:, None], T["label"][m])
        return self

    def predict(self, T, mask):
        return self.lr.predict_proba(logit(T["p_final"][mask])[:, None])[:, 1]


def _phi(q):
    q = q[:, :7].astype(np.float64)              # the last descriptor (face scale) is constant per video
    return np.concatenate([q, q ** 2], 1)


class QShift:
    name = "qshift"

    def fit(self, T, fit_mask):
        real = fit_mask & (T["label"] == 0)
        X = _phi(T["q"][real])
        self.sc = StandardScaler().fit(X)
        Xs = self.sc.transform(X)
        self.reg = [Ridge(alpha=10.0).fit(Xs, T["d"][real][:, h]) for h in range(3)]
        clean = real & (T["cond"] == "clean")
        Xc = self.sc.transform(_phi(T["q"][clean]))
        self.base = np.array([r.predict(Xc).mean() for r in self.reg])
        return self

    def nuisance(self, T, mask):
        X = self.sc.transform(_phi(T["q"][mask]))
        return np.stack([r.predict(X) for r in self.reg], 1) - self.base

    def predict(self, T, mask):
        return p_from_d(T["d"][mask] - self.nuisance(T, mask))


class EvidenceNulling:
    """Paired, per-term quality regression on background-referenced descriptors."""

    def __init__(self, qkey="q_bg", alpha=10.0, name=None):
        self.qkey, self.alpha = qkey, alpha
        self.name = name or ("en" if qkey == "q_bg" else "en-face")

    def fit(self, T, fit_mask):
        real = fit_mask & (T["label"] == 0)
        terms = head_terms(T)
        src, dst = pairs(T, real)
        ok = np.isfinite(T[self.qkey][src]).all(1) & np.isfinite(T[self.qkey][dst]).all(1)
        src, dst = src[ok], dst[ok]
        dphi = _phi(T[self.qkey][dst]) - _phi(T[self.qkey][src])
        self.scale = dphi.std(0) + 1e-6
        X = dphi / self.scale
        self.W = {}
        for h in HEADS:
            Y = terms[h][dst] - terms[h][src]
            self.W[h] = Ridge(alpha=self.alpha, fit_intercept=False).fit(X, Y)
        clean = real & (T["cond"] == "clean") & np.isfinite(T[self.qkey]).all(1)
        self.phi0 = _phi(T[self.qkey][clean]).mean(0)
        self.n_pairs = int(len(src))
        return self

    def term_nuisance(self, T, mask, h):
        """Predicted quality evidence on every term of head h (N, K_h)."""
        q = T[self.qkey][mask]
        X = (_phi(np.nan_to_num(q)) - self.phi0) / self.scale
        X[~np.isfinite(q).all(1)] = 0.0
        return self.W[h].predict(X)

    def nuisance(self, T, mask):
        return np.stack([self.term_nuisance(T, mask, h).sum(1) for h in HEADS], 1)

    def predict(self, T, mask):
        return p_from_d(T["d"][mask] - self.nuisance(T, mask))

    def quality_direction(self, h):
        """Per-term response to a unit change of each quality descriptor (K_h, n_phi)."""
        return self.W[h].coef_


class Projection:
    """Ablation: remove the evidence along the top-r directions of the real-video drift."""

    def __init__(self, rank=1):
        self.rank = rank
        self.name = "en-proj"

    def fit(self, T, fit_mask):
        terms = head_terms(T)
        real = fit_mask & (T["label"] == 0)
        drift, _, _ = paired_drift(T, terms, real)
        clean_real = real & (T["cond"] == "clean")
        self.U, self.mu, self.energy = {}, {}, {}
        for h in HEADS:
            _, s, Vt = np.linalg.svd(drift[h], full_matrices=False)
            self.U[h] = Vt[:self.rank].T
            self.mu[h] = terms[h][clean_real].mean(0)
            self.energy[h] = (np.cumsum(s ** 2) / np.sum(s ** 2))[:10]
        return self

    def nuisance(self, T, mask):
        terms = head_terms(T)
        out = []
        for h in HEADS:
            c = terms[h][mask] - self.mu[h]
            out.append((c @ self.U[h]) @ (self.U[h].T @ np.ones(self.U[h].shape[0])))
        return np.stack(out, 1)

    def predict(self, T, mask):
        return p_from_d(T["d"][mask] - self.nuisance(T, mask))


class Refit:
    name = "refit"

    def fit(self, T, fit_mask):
        terms = head_terms(T)
        X = np.concatenate([terms[h][fit_mask] for h in HEADS], 1)
        self.sc = StandardScaler().fit(X)
        self.lr = LogisticRegression(C=0.05, max_iter=3000).fit(self.sc.transform(X), T["label"][fit_mask])
        return self

    def predict(self, T, mask):
        terms = head_terms(T)
        X = np.concatenate([terms[h][mask] for h in HEADS], 1)
        return self.lr.predict_proba(self.sc.transform(X))[:, 1]


def all_methods():
    return [Original(), Platt(), QShift(), EvidenceNulling(), EvidenceNulling(qkey="q"), Projection(),
            Refit()]
