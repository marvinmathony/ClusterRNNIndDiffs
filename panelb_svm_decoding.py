"""
Panel-b NONLINEAR diag decoding (SVM-RBF), leak-free LOO.

Drop-in companion to the linear `diag_decode_loo` used in
dezfouli_publication_panels.ipynb panel b. The linear decoder showed no
IDRNN-vs-Vanilla difference; this tests whether an RBF-kernel SVM recovers
structure a linear classifier misses.

Why an SVM-RBF needs different plumbing than the LR version:
  * RBF kernels are scale-sensitive -> StandardScaler fitted on the TRAIN
    fold only (never the held-out subject) inside every LOO iteration.
  * predict_proba via Platt scaling can place 0 mass on a class -> clip
    before taking logs for the per-subject Bayes-factor comparison.

The function signature/returns mirror `diag_decode_loo` exactly
(macro_auc, accuracy, per_class_aucs, P) so the existing
`bootstrap_auc_ci`, per-class markers, and `paired_bf` code work unchanged.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score


def diag_decode_loo_svm(X_raw, k, y, n_class=3, C=1.0, gamma='scale'):
    """Leave-one-out SVM-RBF diag decoding with per-fold scaling (+optional PCA).

    Parameters
    ----------
    X_raw : (n_subjects, D) latent feature matrix.
    k     : int or None. If int and D > k, reduce to k PCs (fit on train fold)
            for an apples-to-apples comparison with the 2-PC linear panel.
            If None, use the full latent (scaled) so the RBF decoder has
            maximal access to nonlinear structure.
    y     : (n_subjects,) integer class labels in {0..n_class-1}.
    C, gamma : SVC hyper-parameters. Defaults (1.0, 'scale') are sensible
            for n~100; see `nested_svm_grid` below to tune them honestly.

    Returns
    -------
    macro_auc, accuracy, per_class_aucs (list), P (n_subjects, n_class) of
    LOO out-of-fold predicted probabilities.
    """
    X_raw = np.asarray(X_raw, dtype=float)
    y = np.asarray(y)
    n, D = X_raw.shape
    P = np.zeros((n, n_class))
    for tr, te in LeaveOneOut().split(X_raw):
        Xtr, Xte = X_raw[tr], X_raw[te]
        # scale on train fold only
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        # optional dim match
        if k is not None and D > k:
            pca = PCA(n_components=k).fit(Xtr)
            Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        clf = SVC(kernel='rbf', C=C, gamma=gamma,
                  class_weight='balanced', probability=True, random_state=0)
        clf.fit(Xtr, y[tr])
        # align class columns in case a fold is missing a class
        proba = clf.predict_proba(Xte)
        for j, c in enumerate(clf.classes_):
            P[te, c] = proba[:, j]
    aucs = [float(roc_auc_score((y == c).astype(int), P[:, c])) for c in range(n_class)]
    return float(np.mean(aucs)), float(accuracy_score(y, P.argmax(1))), aucs, P


def nested_svm_grid(X_raw, k, y, n_class=3,
                    Cs=(0.1, 1.0, 10.0),
                    gammas=('scale', 0.01, 0.1, 1.0),
                    inner_splits=5, seed=0):
    """Honest nested-CV variant: pick (C, gamma) per OUTER LOO fold via inner
    stratified-k-fold macro-AUC, so a 'no difference' conclusion can't be
    blamed on a single bad hyper-parameter guess. Slower but leak-free.
    Returns the same (macro_auc, accuracy, per_class_aucs, P) tuple."""
    X_raw = np.asarray(X_raw, dtype=float)
    y = np.asarray(y)
    n, D = X_raw.shape
    P = np.zeros((n, n_class))
    for tr, te in LeaveOneOut().split(X_raw):
        Xtr_raw, Xte_raw, ytr = X_raw[tr], X_raw[te], y[tr]
        best, best_auc = ('scale', 1.0), -np.inf
        for C in Cs:
            for g in gammas:
                skf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
                Pin = np.zeros((len(ytr), n_class))
                ok = True
                for itr, ite in skf.split(Xtr_raw, ytr):
                    sc = StandardScaler().fit(Xtr_raw[itr])
                    Xi, Xj = sc.transform(Xtr_raw[itr]), sc.transform(Xtr_raw[ite])
                    if k is not None and D > k:
                        pca = PCA(n_components=k).fit(Xi)
                        Xi, Xj = pca.transform(Xi), pca.transform(Xj)
                    try:
                        clf = SVC(kernel='rbf', C=C, gamma=g, class_weight='balanced',
                                  probability=True, random_state=0).fit(Xi, ytr[itr])
                    except Exception:
                        ok = False; break
                    pr = clf.predict_proba(Xj)
                    for jj, cc in enumerate(clf.classes_):
                        Pin[ite, cc] = pr[:, jj]
                if not ok:
                    continue
                try:
                    a = np.mean([roc_auc_score((ytr == c).astype(int), Pin[:, c])
                                 for c in range(n_class)])
                except Exception:
                    a = -np.inf
                if a > best_auc:
                    best_auc, best = a, (C, g)
        C, g = best
        sc = StandardScaler().fit(Xtr_raw)
        Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)
        if k is not None and D > k:
            pca = PCA(n_components=k).fit(Xtr)
            Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        clf = SVC(kernel='rbf', C=C, gamma=g, class_weight='balanced',
                  probability=True, random_state=0).fit(Xtr, ytr)
        pr = clf.predict_proba(Xte)
        for jj, cc in enumerate(clf.classes_):
            P[te, cc] = pr[:, jj]
    aucs = [float(roc_auc_score((y == c).astype(int), P[:, c])) for c in range(n_class)]
    return float(np.mean(aucs)), float(accuracy_score(y, P.argmax(1))), aucs, P
