"""
Rerun the marginal log-likelihood for sloutsky test participants with
n_samples=500 instead of 2000. Uses the already-fit EM priors saved in
em_results.npz so EM is not redone.

Writes results to cog_model_results_n500.csv alongside the existing
cog_model_results.csv (which used n_samples=2000) for direct comparison.
"""
import numpy as np
import pandas as pd
from sloutsky_cog_model import (
    build_data_arrays,
    compute_marginal_loglik_logsumexp,
    loglik,
)

N_SAMPLES_NEW = 500


def rerun_for_dir(data_dir):
    em = np.load(f"{data_dir}/em_results.npz", allow_pickle=True)
    m, v = em["m"], em["v"]

    df_test = pd.read_csv(f"{data_dir}/df_test.csv").sort_values(
        ["subid", "TrainingTrial"]
    )
    test_participants, value_test, choice_test, novelty_test, lags_test = (
        build_data_arrays(df_test)
    )
    P_test = len(test_participants)

    marginal_ll = np.zeros(P_test)
    for p in range(P_test):
        marginal_ll[p] = compute_marginal_loglik_logsumexp(
            m, v,
            value_test[p], novelty_test[p], lags_test[p], choice_test[p],
            n_samples=N_SAMPLES_NEW, seed=p,
        )
    marginal_nll = -marginal_ll

    out_csv = f"{data_dir}/cog_model_results_n500.csv"
    pd.DataFrame({
        "subid": test_participants,
        "normalized_likelihood": marginal_nll,
        "model": f"CogModel_EM_n{N_SAMPLES_NEW}",
    }).to_csv(out_csv, index=False)

    old_csv = f"{data_dir}/cog_model_results.csv"
    try:
        old = pd.read_csv(old_csv)
        key = "subid" if "subid" in old.columns else "session"
        merged = old.merge(
            pd.DataFrame({key: test_participants, "nll_n500": marginal_nll}),
            on=key, how="inner",
        )
        delta = merged["nll_n500"] - merged["normalized_likelihood"]
        print(f"\n[{data_dir}]")
        print(f"  P_test = {P_test}")
        print(f"  n=2000 mean NLL: {merged['normalized_likelihood'].mean():.3f}  "
              f"sd: {merged['normalized_likelihood'].std():.3f}")
        print(f"  n=500  mean NLL: {merged['nll_n500'].mean():.3f}  "
              f"sd: {merged['nll_n500'].std():.3f}")
        print(f"  delta (n500 - n2000): mean {delta.mean():+.3f}  "
              f"sd {delta.std():.3f}  "
              f"min {delta.min():+.3f}  max {delta.max():+.3f}")
        print(f"  saved -> {out_csv}")
    except FileNotFoundError:
        print(f"[{data_dir}] saved -> {out_csv} (no n=2000 file to compare)")


if __name__ == "__main__":
    for d in ["data_sloutsky",
              "data_sloutsky/fold0",
              "data_sloutsky/fold1",
              "data_sloutsky/fold2"]:
        rerun_for_dir(d)
