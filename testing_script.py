import json
import os
import torch
import numpy as np
from modelsandtraining import *
import RL_fittingfunctions2 as fit
import pickle
import argparse
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse arguments
parser = argparse.ArgumentParser(description="Test trained models")
parser.add_argument('--latent', type=lambda x: x.lower() == 'true', default=True, help="latent or vanilla modeling")
parser.add_argument('--dataset_id', type=int, default=0, help="dataset ID for multi-dataset experiments")
parser.add_argument('--model_fitting', type=lambda x: x.lower() == 'true', default=False, help="whether to fit cognitive models")
parser.add_argument('--dgp', type=str, default=None,
                    help="Data generating process type (e.g., 'bimodal', 'uniform'). Must match data_generation.py --dgp")
parser.add_argument('--fold', type=int, default=None,
                    help="Outer CV fold index (0, 1, 2). If set, loads data from data_{DGP}/fold{fold}/ "
                         "and models from runs_{DGP}/fold{fold}/seed_{seed}/")
parser.add_argument('--run_suffix', type=str, default=None,
                    help="Optional suffix appended to the IDRNN run directory, "
                         "matching --run_suffix used in run_Q_model.py.")
args = parser.parse_args()

############################
#### LOAD CORRECT MODEL ####
############################
latent = args.latent
DATASET_ID = args.dataset_id
model_fitting = args.model_fitting
DGP = args.dgp
FOLD = args.fold

# Determine if this is human data (sloutsky/palminteri/spatial_bandit/dezfouli/thalmann)
is_human_data = DGP in ("sloutsky", "palminteri", "spatial_bandit", "dezfouli", "thalmann")

# Build directory names with optional DGP prefix
if is_human_data:
    DATA_DIR = f"data_{DGP}" if FOLD is None else f"data_{DGP}/fold{FOLD}"
elif DGP:
    DATA_DIR = f"data_{DGP}_dataset{DATASET_ID}"
else:
    DATA_DIR = f"data_dataset{DATASET_ID}"
n_fit_iter = 5
vanilla_nametag = "vanilla"
latent_nametag = "latentmodel"
nametag = latent_nametag if latent else vanilla_nametag

# Load data
df_train = pd.read_csv(f"{DATA_DIR}/df_train.csv")
df_test = pd.read_csv(f"{DATA_DIR}/df_test.csv")

# Rename subject column for human data
if DGP == "sloutsky":
    df_train = df_train.rename(columns={'subid': 'session'})
    df_test = df_test.rename(columns={'subid': 'session'})
elif DGP == "palminteri":
    df_train = df_train.rename(columns={'sub_across_groups': 'session'})
    df_test = df_test.rename(columns={'sub_across_groups': 'session'})
elif DGP in ("spatial_bandit", "dezfouli", "thalmann"):
    df_train = df_train.rename(columns={'subid': 'session'})
    df_test = df_test.rename(columns={'subid': 'session'})

# session_ll_df_test only exists for synthetic data
if not is_human_data:
    session_ll_df_test = pd.read_csv(f"{DATA_DIR}/session_ll_df_test.csv")
else:
    session_ll_df_test = None

xin_train = np.load(f"{DATA_DIR}/xin_train.npy", allow_pickle=True)
xin_test = np.load(f"{DATA_DIR}/xin_test.npy", allow_pickle=True)
c_test = np.load(f"{DATA_DIR}/c_test.npy", allow_pickle=True)
c_train = np.load(f"{DATA_DIR}/c_train.npy", allow_pickle=True)

xin_train = torch.from_numpy(xin_train).float().to(device)
xin_test = torch.from_numpy(xin_test).float().to(device)
c_test = torch.from_numpy(c_test).float().to(device)
c_train = torch.from_numpy(c_train).float().to(device)

# Encoder-specific inputs (participant-specific dims only; None if not available)
_enc_test_path  = f"{DATA_DIR}/xin_enc_test.npy"
_enc_train_path = f"{DATA_DIR}/xin_enc_train.npy"
xin_enc_test  = torch.from_numpy(np.load(_enc_test_path)).float().to(device)  if os.path.exists(_enc_test_path)  else None
xin_enc_train = torch.from_numpy(np.load(_enc_train_path)).float().to(device) if os.path.exists(_enc_train_path) else None

# Build runs directory
if is_human_data:
    _suffix = f"_{args.run_suffix}" if args.run_suffix else ""
    _runs_base = f"runs_{DGP}{_suffix}" if latent else f"runs_vanilla_{DGP}{_suffix}"
    BASE_DIR = os.path.join(_runs_base, f"fold{FOLD}") if FOLD is not None else _runs_base
elif DGP:
    BASE_DIR = f"runs_{DGP}_dataset{DATASET_ID}" if latent else f"runs_vanilla_{DGP}_dataset{DATASET_ID}"
else:
    BASE_DIR = f"runs_dataset{DATASET_ID}" if latent else f"runs_vanilla_dataset{DATASET_ID}"


# ── For human data: iterate over all seeds using per-seed CV checkpoint ────────
if is_human_data:
    seed_dirs = sorted([
        d for d in os.listdir(BASE_DIR)
        if d.startswith("seed_") and os.path.isdir(os.path.join(BASE_DIR, d))
    ])

    for seed_dir_name in seed_dirs:
        seed = int(seed_dir_name.split("_")[1])
        run_dir = os.path.join(BASE_DIR, seed_dir_name)
        config_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"Seed {seed}: config.json not found, skipping.")
            continue

        with open(config_path) as f:
            cfg = json.load(f)

        # Replay any input-dim masking applied at training time so the model
        # is evaluated on the same input distribution it was trained on.
        _mask_str = cfg.get("mask_input_dims", "") or ""
        _mask_dims = [int(d) for d in _mask_str.split(",")] if _mask_str else []
        if _mask_dims:
            for _x in (xin_train, xin_test):
                for _d in _mask_dims:
                    _pad = (_x[..., _d] == -100.0)
                    _x[..., _d] = torch.where(_pad, _x[..., _d],
                                               torch.zeros_like(_x[..., _d]))
            if xin_enc_train is not None:
                for _d in _mask_dims:
                    _pad = (xin_enc_train[..., _d] == -100.0)
                    xin_enc_train[..., _d] = torch.where(
                        _pad, xin_enc_train[..., _d],
                        torch.zeros_like(xin_enc_train[..., _d]))
            if xin_enc_test is not None:
                for _d in _mask_dims:
                    _pad = (xin_enc_test[..., _d] == -100.0)
                    xin_enc_test[..., _d] = torch.where(
                        _pad, xin_enc_test[..., _d],
                        torch.zeros_like(xin_enc_test[..., _d]))
            print(f"  Seed {seed}: replayed mask_input_dims={_mask_dims}")

        # Determine which epoch to load
        if "cv_selected_epoch" in cfg:
            best_epoch = cfg["cv_selected_epoch"]
            print(f"Seed {seed}: using CV-selected epoch {best_epoch}")
        else:
            # Fallback: use the last available checkpoint
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            available = []
            if os.path.exists(ckpt_dir):
                available = sorted([
                    int(f.replace("epoch", "").replace(".pt", ""))
                    for f in os.listdir(ckpt_dir)
                    if f.startswith("epoch") and f.endswith(".pt")
                ])
            if not available:
                print(f"Seed {seed}: no checkpoints found, skipping.")
                continue
            best_epoch = available[-1]
            print(f"Seed {seed}: no cv_selected_epoch in config, using last checkpoint epoch {best_epoch}")

        ckpt_path = os.path.join(run_dir, "checkpoints", f"epoch{best_epoch:04d}.pt")
        if not os.path.exists(ckpt_path):
            print(f"Seed {seed}: checkpoint {ckpt_path} not found, skipping.")
            continue

        # Decide encoder input (same_enc_dec check)
        if latent:
            _enc_in_dim = cfg.get("model_config", {}).get("enc_in_dim", cfg["in_dim"])
            if _enc_in_dim == cfg["in_dim"]:
                xin_enc_test_use  = xin_test
                xin_enc_train_use = xin_train
            else:
                xin_enc_test_use  = xin_enc_test
                xin_enc_train_use = xin_enc_train
        else:
            xin_enc_test_use  = None
            xin_enc_train_use = None

        # Load model
        _mc = cfg.get("model_config", {})
        _dec_in_dim              = _mc.get("dec_in_dim", cfg["in_dim"])
        _task_emb_dim            = _mc.get("task_emb_dim", 0)
        _n_tasks                 = _mc.get("n_tasks", None)
        _reinit_decoder_per_block = _mc.get("reinit_decoder_per_block", False)
        if latent:
            enc_in_dim = _mc.get("enc_in_dim", cfg["in_dim"])
            enc_hidden = _mc.get("enc_hidden", cfg.get("hidden", 8))
            encoder_IDRNN = IDRNN(in_dim=enc_in_dim, z_dim=cfg["z_dim"], hid=enc_hidden,
                                  n_tasks=_n_tasks, task_emb_dim=_task_emb_dim)
            frozen_decoder = Decoder(in_dim=_dec_in_dim, z_dim=cfg["z_dim"], hid=cfg["hidden"], A=cfg["A"])
            for p in frozen_decoder.parameters():
                p.requires_grad = False
            model = LatentRNN_secondstep(
                encoder=encoder_IDRNN, hid=cfg["hidden"], z_dim=cfg["z_dim"],
                in_dim=_dec_in_dim, A=cfg["A"], decoder=frozen_decoder,
                n_tasks=_n_tasks, task_emb_dim=_task_emb_dim,
                reinit_decoder_per_block=_reinit_decoder_per_block
            )
        else:
            _block_structure = (xin_test.dim() == 4)
            model = AblatedRNN(hid=cfg["hidden"], in_dim=_dec_in_dim, A=cfg["A"],
                               block_structure=_block_structure,
                               n_tasks=_n_tasks, task_emb_dim=_task_emb_dim)
        model.to(device)

        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # Register task IDs for models with task embeddings (e.g., thalmann)
        if _n_tasks is not None and _task_emb_dim > 0 and DGP == "thalmann":
            _thal_root = "data_thalmann"
            _task_ids_np = np.load(os.path.join(_thal_root, "task_ids_per_block.npy"))
            model.set_task_ids(torch.from_numpy(_task_ids_np).long().to(device))

        # Compute likelihoods and latents
        rnn_df, latent_tensor, latent_tensor_train, geometric_mean_prob_rnn = \
            compute_rnn_likelihoods_torch(
                test_latentrnn_secondstep_causal_posterior_weighting,
                model, xin_test, c_test, xin_train,
                latent=latent, choice_train=c_train, id=True,
                test_xin_enc=xin_enc_test_use, train_xin_enc=xin_enc_train_use
            )

        if latent:
            rnn_no_id, _, _, geometric_mean_prob_common_process = \
                compute_rnn_likelihoods_torch(
                    test_latentrnn_secondstep_causal_posterior_weighting,
                    model, xin_test, c_test, xin_train,
                    latent=latent, choice_train=c_train, id=False,
                    test_xin_enc=xin_enc_test_use, train_xin_enc=xin_enc_train_use
                )
            print(f"  Seed {seed} normalized_ll IDRNN: {geometric_mean_prob_rnn}")
            print(f"  Seed {seed} normalized_ll common process: {geometric_mean_prob_common_process}")
        else:
            print(f"  Seed {seed} normalized_ll vanilla: {geometric_mean_prob_rnn}")

        # Save per-seed output directory.  When --run_suffix is set, nest
        # under a sub-folder so old latents at the unsuffixed path are preserved.
        if args.run_suffix:
            seed_data_dir = os.path.join(DATA_DIR, f"seed_{seed}", args.run_suffix)
        else:
            seed_data_dir = os.path.join(DATA_DIR, f"seed_{seed}")
        os.makedirs(seed_data_dir, exist_ok=True)

        torch.save(latent_tensor,       os.path.join(seed_data_dir, f"latents_tensor{nametag}.pt"))
        torch.save(latent_tensor_train, os.path.join(seed_data_dir, f"latents_tensor{nametag}_traindata.pt"))
        rnn_df.to_csv(os.path.join(seed_data_dir, f"rnn_results{nametag}.csv"), index=False)
        if latent:
            rnn_no_id.to_csv(os.path.join(seed_data_dir, "rnn_results_common_process.csv"), index=False)

        print(f"Seed {seed}: saved latents and results to {seed_data_dir}")

    print("\n✅ All seeds processed.")


# ── For synthetic data: original single-model logic ───────────────────────────
else:
    # Load best epoch from selection file
    if latent:
        with open(os.path.join(BASE_DIR, "best_epoch_by_nll.json"), "r") as f:
            meta = json.load(f)
    else:
        loss_path = os.path.join(BASE_DIR, "best_epoch_by_nll.json")
        rsa_path  = os.path.join(BASE_DIR, "best_epoch_by_rsa.json")
        if os.path.exists(loss_path):
            with open(loss_path, "r") as f:
                meta = json.load(f)
        elif os.path.exists(rsa_path):
            with open(rsa_path, "r") as f:
                meta = json.load(f)
        else:
            raise FileNotFoundError(f"No best epoch selection file found in {BASE_DIR}")

    best_epoch = meta["best_epoch"]
    SEED_FOR_ANALYSIS = meta["best_seed"]
    print(f"Using epoch {best_epoch:04d} from seed {SEED_FOR_ANALYSIS} for analysis.")

    run_dir = f"{BASE_DIR}/seed_{SEED_FOR_ANALYSIS}"
    config_path = os.path.join(run_dir, "config.json")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    if latent:
        _enc_in_dim = cfg.get("model_config", {}).get("enc_in_dim", cfg["in_dim"])
        if _enc_in_dim == cfg["in_dim"]:
            xin_enc_test  = xin_test
            xin_enc_train = xin_train

    if latent:
        enc_in_dim = cfg.get("model_config", {}).get("enc_in_dim", cfg["in_dim"])
        enc_hidden = cfg.get("model_config", {}).get("enc_hidden", cfg.get("hidden", 8))
        encoder_IDRNN = IDRNN(in_dim=enc_in_dim, z_dim=cfg["z_dim"], hid=enc_hidden)
        frozen_decoder = Decoder(in_dim=cfg["in_dim"], z_dim=cfg["z_dim"], hid=cfg["hidden"], A=cfg["A"])
        for p in frozen_decoder.parameters():
            p.requires_grad = False
        model = LatentRNN_secondstep(encoder=encoder_IDRNN, hid=cfg["hidden"], z_dim=cfg["z_dim"],
                                     in_dim=cfg["in_dim"], A=cfg["A"], decoder=frozen_decoder)
        model.to(device)
    else:
        _block_structure = (xin_test.dim() == 4)
        model = AblatedRNN(hid=cfg["hidden"], in_dim=cfg["in_dim"], A=cfg["A"], block_structure=_block_structure).to(device)

    ckpt_path = os.path.join(BASE_DIR, f"seed_{SEED_FOR_ANALYSIS}", "checkpoints", f"epoch{best_epoch:04d}.pt")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    rnn_df, latent_tensor, latent_tensor_train, geometric_mean_prob_rnn = compute_rnn_likelihoods_torch(
        test_latentrnn_secondstep_causal_posterior_weighting, model, xin_test, c_test, xin_train,
        latent=latent, choice_train=c_train, id=True,
        test_xin_enc=xin_enc_test, train_xin_enc=xin_enc_train
    )
    if latent:
        rnn_no_id, _, _, geometric_mean_prob_common_process = compute_rnn_likelihoods_torch(
            test_latentrnn_secondstep_causal_posterior_weighting, model, xin_test, c_test, xin_train,
            latent=latent, choice_train=c_train, id=False,
            test_xin_enc=xin_enc_test, train_xin_enc=xin_enc_train
        )
        print(f"normalized_ll IDRNN: {geometric_mean_prob_rnn}")
        print(f"normalized_ll common process: {geometric_mean_prob_common_process}")

        df = pd.DataFrame([{
            "model": "IDRNN",
            "geometric_mean_prob_rnn": geometric_mean_prob_rnn,
            "geometric_mean_prob_common_process": geometric_mean_prob_common_process,
            "dataset_id": DATASET_ID
        }])
        out_path = f"{DATA_DIR}/latent_results.csv"
    else:
        print(f"normalized_ll vanilla: {geometric_mean_prob_rnn}")
        df = pd.DataFrame([{
            "model": "vanillaRNN",
            "geometric_mean_prob_rnn": geometric_mean_prob_rnn,
            "dataset_id": DATASET_ID
        }])
        out_path = f"{DATA_DIR}/vanilla_results.csv"

    if not os.path.exists(out_path):
        df.to_csv(out_path, index=False)
    else:
        df.to_csv(out_path, mode='a', header=False, index=False)
    print(f"Saved results in {out_path}")

    # Cognitive model fitting (synthetic data only)
    model_configs = {
        "Q":  {"asymmetric_alpha": False, "forgetting_type": "none",  "choice_trace": False},
        "FQ": {"asymmetric_alpha": False, "forgetting_type": "fixed", "choice_trace": False}
    }
    if model_fitting:
        model_eval_Q_df, params_dict, p1_common_dict, p1_ML_dict, p1_MAP_dict, q_common_dict = fit.fit_all_models(
            model_configs, df_train, df_test, n_iter=n_fit_iter, fit_ML=False
        )
        with open(f"{DATA_DIR}/params_dict.pkl", "wb") as f:
            pickle.dump(params_dict, f)
        pd.DataFrame({"model": list(params_dict.keys()),
                      "params": [list(v) for v in params_dict.values()]
                      }).to_csv(f"{DATA_DIR}/params_dict.csv", index=False)
        np.savez(f"{DATA_DIR}/q_common_dict.npz", **q_common_dict)

        dfs_to_concat = [model_eval_Q_df]
        if session_ll_df_test is not None:
            dfs_to_concat.append(session_ll_df_test)
        dfs_to_concat.append(rnn_df)
        if latent:
            dfs_to_concat.append(rnn_no_id)
        model_eval_df = pd.concat(dfs_to_concat, ignore_index=True)
        np.savez(f"{DATA_DIR}/p1_common_dict.npz", **p1_common_dict)

    if latent:
        rnn_df.to_csv(f"{DATA_DIR}/rnn_results{latent_nametag}.csv", index=False)
        rnn_no_id.to_csv(f"{DATA_DIR}/rnn_results_common_process.csv", index=False)
        torch.save(latent_tensor,       f"{DATA_DIR}/latents_tensor{latent_nametag}.pt")
        torch.save(latent_tensor_train, f"{DATA_DIR}/latents_tensor{latent_nametag}_traindata.pt")
        if model_fitting:
            model_eval_df.to_csv(f"{DATA_DIR}/model_eval_df{latent_nametag}.csv", index=False)
    else:
        rnn_df.to_csv(f"{DATA_DIR}/rnn_results{vanilla_nametag}.csv", index=False)
        torch.save(latent_tensor,       f"{DATA_DIR}/latents_tensor{vanilla_nametag}.pt")
        torch.save(latent_tensor_train, f"{DATA_DIR}/latents_tensor{vanilla_nametag}_traindata.pt")
        if model_fitting:
            model_eval_df.to_csv(f"{DATA_DIR}/model_eval_df{vanilla_nametag}.csv", index=False)

    print("✅ Training and fitting done. Results saved to data/ and checkpoints/")
