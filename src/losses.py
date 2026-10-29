"""Student training objectives for week 3 (experiment E3).

All take (logits, y, idx) where idx indexes the training set, so teacher
logits can be precomputed once instead of running the teacher every batch.

    la      logit-adjusted CE (Menon et al., ICLR 2021), no teacher
    kd      Hinton KD: soft KL to the teacher + plain CE
    kd_wce  Hinton KD with class-weighted CE (naive combination)
    rap_kd  proposed: class-balanced KL (weights^beta) + logit-adjusted CE
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

T_DEFAULT, ALPHA_DEFAULT, BETA_DEFAULT, TAU_DEFAULT = 4.0, 0.7, 0.5, 1.0

# (T, alpha) per method, chosen by mean *validation* macro-F1 of mlp_s (seed 0)
# over TON_IoT, UNSW-NB15 and Edge-IIoTset from the grid T in {1, 2} x
# alpha in {0.5, 0.9}. N-BaIoT and CICIoT2023 were not used for selection.
# T = 4 (the week-3 first pass) lost to CE everywhere: the ~8k-parameter
# students cannot fit the flattened targets (train F1 == test F1).
SELECTED = {"kd": (1.0, 0.9), "kd_wce": (1.0, 0.9), "rap_kd": (1.0, 0.9), "la_kd": (2.0, 0.9),
            # Not tuned separately: reuses the kd setting.
            "bt_kd": (1.0, 0.9)}


def class_prior(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=n_classes).float().clamp(min=1)
    return counts / counts.sum()


def _kl(student, teacher, T):
    """Per-sample KL(teacher_T || student_T), scaled by T^2 (Hinton et al.)."""
    return F.kl_div(F.log_softmax(student / T, 1), F.log_softmax(teacher / T, 1),
                    reduction="none", log_target=True).sum(1) * T * T


def hyper(name: str) -> dict:
    T, alpha = SELECTED.get(name, (T_DEFAULT, ALPHA_DEFAULT))
    return {"T": T, "alpha": alpha, "beta": BETA_DEFAULT, "tau": TAU_DEFAULT}


def make(name: str, y_train: torch.Tensor, n_classes: int, teacher_logits=None,
         T=None, alpha=None, beta=BETA_DEFAULT, tau=TAU_DEFAULT):
    h = hyper(name)
    T = h["T"] if T is None else T
    alpha = h["alpha"] if alpha is None else alpha
    prior = class_prior(y_train, n_classes)
    log_prior = tau * prior.log()
    w = (1.0 / (n_classes * prior))
    w = w / w.mean()                                    # inverse frequency, mean 1

    if name == "la":
        # Train on logits + tau*log(prior); predict with raw logits.
        return lambda s, y, idx: F.cross_entropy(s + log_prior, y)

    if teacher_logits is None:
        raise ValueError(f"{name} needs teacher logits")
    t_all = teacher_logits

    if name == "kd":
        return lambda s, y, idx: (alpha * _kl(s, t_all[idx], T).mean()
                                  + (1 - alpha) * F.cross_entropy(s, y))
    if name == "kd_wce":
        return lambda s, y, idx: (alpha * _kl(s, t_all[idx], T).mean()
                                  + (1 - alpha) * F.cross_entropy(s, y, weight=w))
    if name == "rap_kd":
        wb = w.pow(beta)
        wb = wb / wb.mean()

        def loss(s, y, idx):
            kl = _kl(s, t_all[idx], T)
            kd = (wb[y] * kl).sum() / wb[y].sum()       # class-balanced soft term
            return alpha * kd + (1 - alpha) * F.cross_entropy(s + log_prior, y)
        return loss
    if name == "la_kd":
        # Both terms see the prior-adjusted student logits s + tau*log(prior).
        # Matching the teacher there means the raw logits s approximate the
        # teacher's posterior with the training prior divided out, i.e. a
        # class-balanced teacher; inference uses raw s, as in logit adjustment.
        def loss(s, y, idx):
            s_adj = s + log_prior
            return (alpha * _kl(s_adj, t_all[idx], T).mean()
                    + (1 - alpha) * F.cross_entropy(s_adj, y))
        return loss
    if name == "bt_kd":
        # Intended for a balanced (logit-adjusted) teacher: the soft term
        # matches its balanced raw logits, the hard term is logit-adjusted CE,
        # so both push the student's raw logits toward a balanced posterior.
        return lambda s, y, idx: (alpha * _kl(s, t_all[idx], T).mean()
                                  + (1 - alpha) * F.cross_entropy(s + log_prior, y))
    raise ValueError(name)


METHODS = ("la", "kd", "kd_wce", "rap_kd", "la_kd", "bt_kd")
NEEDS_TEACHER = ("kd", "kd_wce", "rap_kd", "la_kd", "bt_kd")
