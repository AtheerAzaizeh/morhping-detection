# -*- coding: utf-8 -*-
"""Local demonstration of the overfitting diagnostic + fine-tuning comparison.

NOTE: this runs on SYNTHETIC 2304-d feature vectors engineered to mimic two
regimes — a frozen ImageNet backbone (features not specialized for morphs, so
the SVM leans harder on the training set) and a fine-tuned backbone (features
better separate real vs morph). It proves the diagnostic code path and shows the
exact output format; it is NOT the real trained model.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import learning_curve

rng = np.random.RandomState(42)
D = 2304                      # EfficientNet-B6 feature dimension
N_TR, N_VA, N_TE = 1500, 800, 1200    # scaled-down but realistic ratios


def make_regime(sep, sdims, noise, n_tr, n_va, n_te):
    """Generate real(-1)/morph(+1) features. Only `sdims` dimensions carry a
    weak morph signal of strength `sep`; the other ~2300 dims are pure nuisance
    noise. A weak signal buried in high-dim noise is exactly what lets an RBF
    SVM memorize the training set (overfit)."""
    def sample(n):
        y = rng.choice([-1, 1], size=n)
        X = rng.normal(0, noise, size=(n, D)).astype(np.float32)
        X[:, :sdims] += (y[:, None] * sep)                # class-informative dims
        return X, y
    return sample(n_tr), sample(n_va), sample(n_te)


def diagnose(name, sep, sdims, noise):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = make_regime(sep, sdims, noise,
                                                      N_TR, N_VA, N_TE)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)

    clf = SVC(kernel="rbf", C=10.0, gamma="scale").fit(Xtr_s, ytr)

    acc_tr = accuracy_score(ytr, clf.predict(Xtr_s))
    acc_va = accuracy_score(yva, clf.predict(Xva_s))
    acc_te = accuracy_score(yte, clf.predict(Xte_s))
    f1_te = f1_score(yte, clf.predict(Xte_s), pos_label=1)
    auc_te = roc_auc_score(yte, clf.decision_function(Xte_s))

    gap = acc_tr - acc_va                         # train-minus-val = overfit gap
    if gap > 0.10:
        verdict = "OVERFITTING (train-val gap > 10 pts)"
    elif gap > 0.05:
        verdict = "MILD overfitting (5-10 pts)"
    else:
        verdict = "OK (gap <= 5 pts)"

    print(f"\n=== {name} ===")
    print(f"  Train acc : {acc_tr*100:5.1f}%")
    print(f"  Val   acc : {acc_va*100:5.1f}%")
    print(f"  Test  acc : {acc_te*100:5.1f}%   F1={f1_te:.3f}  AUC={auc_te:.3f}")
    print(f"  Train-Val gap : {gap*100:+.1f} pts  ->  {verdict}")
    return dict(name=name, tr=acc_tr, va=acc_va, te=acc_te, f1=f1_te,
                auc=auc_te, gap=gap, verdict=verdict,
                data=(Xtr_s, ytr))


print("=" * 62)
print("OVERFITTING DIAGNOSTIC — DEMONSTRATION (synthetic features)")
print("=" * 62)

# Frozen backbone: weak morph signal (6 dims, sep 0.12) buried in 2298 noise
# dims -> the RBF SVM memorizes the training set => large train-val gap.
frozen = diagnose("FROZEN EfficientNet-B6 features", sep=0.22, sdims=9, noise=1.0)

# Fine-tuned backbone: features specialized to morph artifacts => cleaner
# separation, the same SVM now generalizes => small gap.
finet = diagnose("FINE-TUNED EfficientNet-B6 features", sep=0.32, sdims=25, noise=1.0)

print("\n" + "=" * 62)
print("BEFORE vs AFTER FINE-TUNING")
print("=" * 62)
print(f"{'Config':<26}{'Train':>8}{'Val':>8}{'Test':>8}{'Gap':>8}")
for r in (frozen, finet):
    print(f"{r['name'][:24]:<26}{r['tr']*100:>7.1f}%{r['va']*100:>7.1f}%"
          f"{r['te']*100:>7.1f}%{r['gap']*100:>+7.1f}")
print(f"\nFine-tuning changed the train-val gap "
      f"{frozen['gap']*100:+.1f} -> {finet['gap']*100:+.1f} pts, "
      f"and test accuracy {frozen['te']*100:.1f}% -> {finet['te']*100:.1f}%.")

# ---- learning curves (are we data-limited? does more data close the gap?) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, r, title in zip(axes, (frozen, finet),
                        ["Frozen features (overfit)",
                         "Fine-tuned features (generalizes)"]):
    X, y = r["data"]
    sizes, tr_sc, va_sc = learning_curve(
        SVC(kernel="rbf", C=10.0, gamma="scale"), X, y,
        train_sizes=np.linspace(0.2, 1.0, 4), cv=3,
        scoring="accuracy", random_state=42)
    ax.plot(sizes, tr_sc.mean(1) * 100, "o-", color="#12a5b8", label="train")
    ax.plot(sizes, va_sc.mean(1) * 100, "s-", color="#d8791a", label="validation")
    ax.fill_between(sizes, va_sc.mean(1) * 100, tr_sc.mean(1) * 100,
                    color="#d8791a", alpha=0.12)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("training samples"); ax.set_ylabel("accuracy %")
    ax.set_ylim(60, 102); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig("/tmp/claude-0/-home-user-morhping-detection/65561dc6-f3f0-577e-ab49-ae3a7c8b6707/scratchpad/overfit_curves.png",
            dpi=150, bbox_inches="tight")
print("\nLearning-curve chart saved -> overfit_curves.png")
print("(wide train-val band = overfitting; converging band = healthy)")
