import sim_Q_data as sim
import os
import argparse
import pandas as pd
import RL_fittingfunctions2 as fit
import plot_functions as plf
import random
import numpy as np
import torch
#from compare_models import *
import copy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from modelsandtraining import *
import json
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#### Parameters and Variables ####
sim_nametag = "Katahira_setup"
latent_nametag = "latentmodel"
vanilla_nametag = "vanilla"
#latent = True
palminteri = False  # Legacy flag, use --dgp palminteri instead
animation = False

n_fit_iter = 5 # iteration for RL fitting (from random initialization)
#flg_on_policy_check = 1 # True # True: do on-policy check
#nTrial_sim = 5000 # nTrial for on-policy check
#file_id = 'scenario1_FQ_discrete'  

#### LOAD DATA ####
if palminteri:
    df_train = pd.read_csv("data/train_df_palminteri.csv")
    df_test = pd.read_csv("data/test_df_palminteri.csv")
    df_val = pd.read_csv("data/val_df_palminteri.csv")
    df_train = df_train.rename(columns={'sub_across_groups': 'session'})
    df_test = df_test.rename(columns={'sub_across_groups': 'session'})
    df_val = df_val.rename(columns={'sub_across_groups': 'session'})
    # ---- load training arrays ----
    xin_train = np.load("data/train_xin_palminteri.npy", allow_pickle=True)
    choice_one_hot_train = np.load("data/train_choice_one_hot_palminteri.npy", allow_pickle=True)
    c_train = np.load("data/c_train_palminteri.npy", allow_pickle=True)

    # ---- load validation arrays ----
    xin_val = np.load("data/val_xin_palminteri.npy", allow_pickle=True)
    choice_one_hot_val = np.load("data/val_choice_one_hot_palminteri.npy", allow_pickle=True)
    c_val = np.load("data/c_val_palminteri.npy", allow_pickle=True)

    # ---- load test arrays ----
    xin_test = np.load("data/test_xin_palminteri.npy", allow_pickle=True)
    choice_one_hot_test = np.load("data/test_choice_one_hot_palminteri.npy", allow_pickle=True)
    c_test = np.load("data/c_test_palminteri.npy", allow_pickle=True)
    pA = np.load("data/pA_train.npy") #not relevant here, but just so that we don't have to adapt script too much
    pA_test = np.load("data/pA_test.npy") #not relevant here, but just so that we don't have to adapt script too much
    c_train = torch.from_numpy(c_train).float().to(device)
    xin_val = torch.from_numpy(xin_val).float().to(device)
    choice_one_hot_val = torch.from_numpy(choice_one_hot_val).float().to(device)

else:
    # This will be updated in __main__ section to use DATA_DIR
    df_train = None
    df_test = None
    session_ll_df_test = None
    test_parameter_df = None
    train_parameter_df = None
    xin_train = None
    xin_test = None
    choice_one_hot_train = None
    choice_one_hot_test = None
    c_test = None
    pA = None
    pA_test = None
    c_train = None
    alphaP_test = None
    params = None
    alphaP_train = None
    params_train = None



# Data loading will be done in __main__ section



if __name__ == '__main__':
    #### TRAINING #####

    parser = argparse.ArgumentParser(description="hello")
    parser.add_argument('--lmbd', type=float, help="how to weigh the two losses", default=1.)
    parser.add_argument('--unif_weight', type=float, default=0.0,
                        help="Weight for uniformity loss on encoder z (Wang & Isola 2020). "
                             "0 = disabled. Encourages participants to be spread in z-space.")
    parser.add_argument('--block_weight_mode', type=str, default="task0_zero",
                        choices=["task0_zero", "uniform"],
                        help="thalmann IDRNN decoder-NLL block weighting. 'task0_zero' "
                             "(default) zero-weights the short 2-armed task (id 0) as in the "
                             "main 2-task model; 'uniform' weights every block equally so each "
                             "task contributes — used for the data-scaling (more-tasks) analysis "
                             "so the model actually learns from every task it is given.")
    parser.add_argument('--z', type=int, help="dimension of latent space", default=1)
    parser.add_argument('--seed', type=int, help="random seed", default=42)
    parser.add_argument('--latent', type=lambda x: x.lower() == 'true', help="latent or vanilla modeling", default=False)
    parser.add_argument('--dataset_id', type=int, help="dataset ID for multi-dataset experiments", default=0)
    parser.add_argument('--joint', type=lambda x: x.lower() == 'true', help="joint training (True) vs two-step (False)", default=False)
    parser.add_argument('--beta', type=float, help="KL weight for joint training (beta-VAE style)", default=0.1)
    parser.add_argument('--epochs', type=int, help="number of training epochs (Step 2)", default=10000)
    parser.add_argument('--step1_epochs', type=int, help="number of Step 1 (decoder pretraining) epochs; if None, uses --epochs", default=None)
    parser.add_argument('--dgp', type=str, default=None,
                        help="Data generating process type (e.g., 'bimodal', 'uniform'). Must match data_generation.py --dgp")
    parser.add_argument('--same_enc_dec', type=lambda x: x.lower() == 'true', default=False,
                        help="If True, encoder and decoder see the same full input (disables xin_enc split)")
    parser.add_argument('--fold', type=int, default=None,
                        help="Outer CV fold index (0, 1, 2). If set, loads data from data_{DGP}/fold{fold}/ "
                             "and saves runs to runs_{DGP}/fold{fold}/seed_{seed}/")
    parser.add_argument('--hidden', type=int, default=8,
                        help="Hidden size of the common-process (decoder) GRU")
    parser.add_argument('--enc_hidden', type=int, default=None,
                        help="Hidden size of the IDRNN encoder GRU; defaults to --hidden if not set")
    parser.add_argument('--task_emb_dim', type=int, default=0,
                        help="Dimension of the learned task embedding appended to decoder input "
                             "(thalmann DGP only). 0 = no task embedding.")
    parser.add_argument('--run_suffix', type=str, default=None,
                        help="Optional suffix appended to the run directory, e.g. 'unif01'. "
                             "Produces runs_{DGP}_{suffix}/ instead of runs_{DGP}/.")
    parser.add_argument('--step2_5_epochs', type=int, default=0,
                        help="If > 0, after step-2 final training runs a brief "
                             "joint encoder+decoder fine-tune with per-timestep z "
                             "(IDRNN only).  K=200-500 closes ~half the dezfouli "
                             "IDRNN-vs-vanilla NLL gap.  0 disables.")
    parser.add_argument('--step2_5_lr', type=float, default=5e-4,
                        help="Learning rate for step 2.5 fine-tune.")
    parser.add_argument('--mask_input_dims', type=str, default="",
                        help="Comma-separated 0-indexed input dims to zero out "
                             "before training (e.g. '0' drops prev_choice). "
                             "Padding values (-100) are left untouched. "
                             "Used for input-ablation experiments; persisted to "
                             "config.json so testing_script.py can replay it.")
    parser.add_argument('--continuous_encoder', type=lambda x: x.lower() == 'true',
                        default=False,
                        help="If True, IDRNN encoder GRU carries hidden state across all "
                             "block boundaries (use for multi-task datasets like thalmann). "
                             "Default False preserves per-block independent behaviour.")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="Optional explicit data directory override. If set, loads "
                             "train/test arrays from this dir instead of the DGP-derived "
                             "default (e.g. data_thalmann_full/ for a full-cohort canonical "
                             "retrain). Purely additive; default None preserves all behaviour.")
    args = parser.parse_args()

    # Step 1 epochs defaults to main epochs if not specified
    if args.step1_epochs is None:
        args.step1_epochs = args.epochs

    DATASET_ID = args.dataset_id
    DGP = args.dgp
    FOLD = args.fold

    # Determine if this is human data (sloutsky/palminteri/spatial_bandit/dezfouli/thalmann) via DGP
    is_human_data = DGP in ("sloutsky", "palminteri", "spatial_bandit", "dezfouli", "thalmann") or palminteri

    # use_inner_cv: True whenever we want the palminteri-style 3-split inner CV
    # (selects best epoch from a train/val split of the training fold).  Human
    # data always uses this.  Synthetic data with --fold also uses it so that
    # nested-CV HP search can pick winners from config.json["cv_val_loss"].
    use_inner_cv = is_human_data or (FOLD is not None)

    # Build directory names with optional DGP prefix
    # Human data (sloutsky/palminteri) doesn't use dataset IDs
    if is_human_data:
        DATA_DIR = f"data_{DGP}" if FOLD is None else f"data_{DGP}/fold{FOLD}"
        PLOT_DIR = f"plots_{DGP}"
    elif DGP:
        DATA_DIR = (f"data_{DGP}_dataset{DATASET_ID}" if FOLD is None
                    else f"data_{DGP}_dataset{DATASET_ID}/fold{FOLD}")
        PLOT_DIR = f"plots_{DGP}_dataset{DATASET_ID}"
    else:
        DATA_DIR = (f"data_dataset{DATASET_ID}" if FOLD is None
                    else f"data_dataset{DATASET_ID}/fold{FOLD}")
        PLOT_DIR = f"plots_dataset{DATASET_ID}"

    # Explicit override (e.g. full-cohort canonical retrain reading from
    # data_thalmann_full/).  Additive: only takes effect when --data_dir is set.
    if args.data_dir:
        DATA_DIR = args.data_dir

    wandb_name = "RNNIndDiffs"

    # Load data with dataset_id
    if is_human_data:
        # Human data (sloutsky/palminteri) - no ground truth parameters
        df_train = pd.read_csv(f"{DATA_DIR}/df_train.csv")
        df_test = pd.read_csv(f"{DATA_DIR}/df_test.csv")

        # Rename subject column to 'session' for consistency
        if DGP == "sloutsky":
            df_train = df_train.rename(columns={'subid': 'session'})
            df_test = df_test.rename(columns={'subid': 'session'})
        elif DGP == "palminteri" or palminteri:
            df_train = df_train.rename(columns={'sub_across_groups': 'session'})
            df_test = df_test.rename(columns={'sub_across_groups': 'session'})
        elif DGP == "spatial_bandit":
            df_train = df_train.rename(columns={'subid': 'session'})
            df_test = df_test.rename(columns={'subid': 'session'})
        elif DGP == "dezfouli":
            df_train = df_train.rename(columns={'subid': 'session'})
            df_test = df_test.rename(columns={'subid': 'session'})
        elif DGP == "thalmann":
            df_train = df_train.rename(columns={'subid': 'session'})
            df_test = df_test.rename(columns={'subid': 'session'})

        # --- Load training arrays ---
        xin_train = np.load(f"{DATA_DIR}/xin_train.npy", allow_pickle=True)
        choice_one_hot_train = np.load(f"{DATA_DIR}/choice_one_hot_train.npy", allow_pickle=True)
        c_train = np.load(f"{DATA_DIR}/c_train.npy", allow_pickle=True)
        # Encoder-specific input (participant-specific dims only; fallback to full input)
        enc_train_path = f"{DATA_DIR}/xin_enc_train.npy"
        xin_enc_train = np.load(enc_train_path, allow_pickle=True) if os.path.exists(enc_train_path) else xin_train
        if args.same_enc_dec:
            xin_enc_train = xin_train  # encoder sees same full input as decoder

        # --- Load validation arrays (optional) ---
        val_path = f"{DATA_DIR}/xin_val.npy"
        has_val = os.path.exists(val_path)
        if has_val:
            xin_val_raw = np.load(val_path, allow_pickle=True)
            if xin_val_raw.shape[0] == 0:
                has_val = False
                print("Validation file exists but has 0 subjects — treating as no validation data")
        if has_val:
            df_val = pd.read_csv(f"{DATA_DIR}/df_val.csv")
            if DGP == "sloutsky":
                df_val = df_val.rename(columns={'subid': 'session'})
            elif DGP == "palminteri" or palminteri:
                df_val = df_val.rename(columns={'sub_across_groups': 'session'})
            elif DGP == "spatial_bandit":
                df_val = df_val.rename(columns={'subid': 'session'})
            xin_val = xin_val_raw
            choice_one_hot_val = np.load(f"{DATA_DIR}/choice_one_hot_val.npy", allow_pickle=True)
            c_val = np.load(f"{DATA_DIR}/c_val.npy", allow_pickle=True)
            enc_val_path = f"{DATA_DIR}/xin_enc_val.npy"
            xin_enc_val = np.load(enc_val_path, allow_pickle=True) if os.path.exists(enc_val_path) else xin_val
            if args.same_enc_dec:
                xin_enc_val = xin_val  # encoder sees same full input as decoder
        else:
            print("No validation data found — using train loss for model selection")
            xin_val = None
            xin_enc_val = None
            choice_one_hot_val = None
            c_val = None

        # --- Load test arrays ---
        xin_test = np.load(f"{DATA_DIR}/xin_test.npy", allow_pickle=True)
        choice_one_hot_test = np.load(f"{DATA_DIR}/choice_one_hot_test.npy", allow_pickle=True)
        c_test = np.load(f"{DATA_DIR}/c_test.npy", allow_pickle=True)

        # Placeholders for pA (not relevant for human data)
        pA = None
        pA_test = None
        params = None
        params_train = None

        # Convert to tensors
        xin_train = torch.from_numpy(xin_train).float().to(device)
        xin_enc_train = torch.from_numpy(xin_enc_train).float().to(device)
        choice_one_hot_train = torch.from_numpy(choice_one_hot_train).float().to(device)
        c_train = torch.from_numpy(c_train).float().to(device)
        if has_val:
            xin_val = torch.from_numpy(xin_val).float().to(device)
            xin_enc_val = torch.from_numpy(xin_enc_val).float().to(device)
            choice_one_hot_val = torch.from_numpy(choice_one_hot_val).float().to(device)
            c_val = torch.from_numpy(c_val).float().to(device)
        xin_test = torch.from_numpy(xin_test).float().to(device)
        choice_one_hot_test = torch.from_numpy(choice_one_hot_test).float().to(device)
        c_test = torch.from_numpy(c_test).float().to(device)

        # ── Optional input-dim masking (e.g. drop prev_choice) ────────────────
        # Zeros the specified input columns wherever they aren't already padding.
        _mask_dims = ([int(d) for d in args.mask_input_dims.split(",")]
                      if args.mask_input_dims else [])
        if _mask_dims:
            for _x in (xin_train, xin_enc_train, xin_test):
                for _d in _mask_dims:
                    _pad = (_x[..., _d] == -100.0)
                    _x[..., _d] = torch.where(_pad, _x[..., _d],
                                               torch.zeros_like(_x[..., _d]))
            if has_val:
                for _x in (xin_val, xin_enc_val):
                    for _d in _mask_dims:
                        _pad = (_x[..., _d] == -100.0)
                        _x[..., _d] = torch.where(_pad, _x[..., _d],
                                                   torch.zeros_like(_x[..., _d]))
            print(f"[mask_input_dims] zeroed input dim(s) {_mask_dims} on "
                  f"train/val/test (padding -100 left untouched)")

        # xin_train is 3-D (B, T, in_dim) for sloutsky/spatial_bandit, or
        # 4-D (B, n_blocks, T, in_dim) for dezfouli/thalmann (block-structured data).
        B      = xin_train.shape[0]
        in_dim = xin_train.shape[-1]
        B_test = xin_test.shape[0]
        enc_in_dim = xin_enc_train.shape[-1]  # 5 if split, else equals in_dim

        # ── Thalmann multi-task setup ──────────────────────────────────────────
        # Load per-block task IDs and prepare a learned task embedding.
        # The decoder receives base_in_dim + task_emb_dim per timestep;
        # the encoder (IDRNN) receives only raw base_in_dim features.
        task_emb_dim    = args.task_emb_dim if DGP == "thalmann" else 0
        task_ids_tensor = None
        if DGP == "thalmann":
            # Prefer task_ids from the active data dir (e.g. data_thalmann_s2_full
            # has 62 blocks for pooled S1+S2); fall back to the 31-block S1 root.
            _tid_path = os.path.join(DATA_DIR, "task_ids_per_block.npy")
            if not os.path.exists(_tid_path):
                _tid_path = os.path.join("data_thalmann", "task_ids_per_block.npy")
            task_ids_np = np.load(_tid_path)
            task_ids_tensor = torch.from_numpy(task_ids_np).long().to(device)
            n_tasks = int(task_ids_np.max()) + 1
            dec_in_dim = in_dim + task_emb_dim   # effective decoder input dim
            print(f"[thalmann] n_tasks={n_tasks}, task_emb_dim={task_emb_dim}, "
                  f"dec_in_dim={dec_in_dim}, enc_in_dim={enc_in_dim}")
        else:
            n_tasks    = None
            dec_in_dim = in_dim

        # Use c_train directly as integer labels so that -100 padding positions
        # (dezfouli: variable-length blocks) are automatically ignored by
        # F.cross_entropy (default ignore_index=-100). For sloutsky/spatial_bandit
        # this is equivalent to argmax(choice_one_hot_train).
        _y_train_labels = c_train.long()
        _y_test_labels  = c_test.long()
    else:
        # Synthetic data - has ground truth parameters
        # --- Load CSVs ---
        df_train = pd.read_csv(f"{DATA_DIR}/df_train.csv")
        df_test = pd.read_csv(f"{DATA_DIR}/df_test.csv")
        # session_ll_df_test only exists in the original (non-fold) synthetic
        # dataset, where it was produced by data_generation.py for the global
        # train/test pair.  In --fold mode, the held-out set is a subject-disjoint
        # partition of the training simulation (no session_ll precomputed).
        _sllt_path = f"{DATA_DIR}/session_ll_df_test.csv"
        session_ll_df_test = (pd.read_csv(_sllt_path) if os.path.exists(_sllt_path)
                              else None)
        # Parameter CSVs: original layout vs fold-subset layout (emitted by
        # make_synthetic_folds.py).
        if os.path.exists(f"{DATA_DIR}/true_param_train.csv"):
            train_parameter_df = pd.read_csv(f"{DATA_DIR}/true_param_train.csv")
            test_parameter_df  = pd.read_csv(f"{DATA_DIR}/true_param_test.csv")
        else:
            train_parameter_df = pd.read_csv(f"{DATA_DIR}/true_parameter_values.csv")
            test_parameter_df  = pd.read_csv(f"{DATA_DIR}/true_test_parameter_values.csv")

        # --- Load NumPy arrays ---
        xin_train = np.load(f"{DATA_DIR}/xin_train.npy")
        xin_test = np.load(f"{DATA_DIR}/xin_test.npy")
        choice_one_hot_train = np.load(f"{DATA_DIR}/choice_one_hot_train.npy")
        choice_one_hot_test = np.load(f"{DATA_DIR}/choice_one_hot_test.npy")
        c_test = np.load(f"{DATA_DIR}/c_test.npy")
        pA = np.load(f"{DATA_DIR}/pA_train.npy")
        pA_test = np.load(f"{DATA_DIR}/pA_test.npy")
        c_train = np.load(f"{DATA_DIR}/c_train.npy")

        alphaP_test = test_parameter_df["alphaP_list"].to_numpy()
        params = test_parameter_df["alphaP_list"].values
        alphaP_train = train_parameter_df["alphaP_list"].to_numpy()
        params_train = train_parameter_df["alphaP_list"].values

        # Convert to tensors
        xin_train = torch.from_numpy(xin_train).float().to(device)
        choice_one_hot_train = torch.from_numpy(choice_one_hot_train).float().to(device)
        choice_one_hot_test = torch.from_numpy(choice_one_hot_test).float().to(device)
        pA = torch.from_numpy(pA).float().to(device)
        pA_test = torch.from_numpy(pA_test).float().to(device)
        xin_test = torch.from_numpy(xin_test).float().to(device)
        c_test = torch.from_numpy(c_test).float().to(device)
        c_train = torch.from_numpy(c_train).float().to(device)

        B, T, in_dim = xin_train.shape
        B_test, T_test, in_dim_test = xin_test.shape

        # Labels used by palminteri-style inner-CV training (fold-mode synthetic).
        # Mirrors the human-data branch so downstream code paths can share them.
        _y_train_labels = c_train.long()
        _y_test_labels  = c_test.long()

        # Synthetic has no task embedding and no separate encoder input; mirror
        # the human-data variables so downstream model-config logging works.
        task_emb_dim = 0
        task_ids_tensor = None
        n_tasks = None
        dec_in_dim = in_dim
        enc_in_dim = in_dim
        xin_enc_train = xin_train

        # Synthetic-fold mode has no separate val set (only train/test inside
        # the fold).  Set has_val=False and zero out val tensors so the
        # palminteri training functions take their no-val code path.
        if FOLD is not None:
            has_val = False
            xin_val = None
            xin_enc_val = None
            choice_one_hot_val = None
            c_val = None

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)



    seed_value = args.seed
    latent = args.latent
    #seeds = [42, 56, 85, 100, 162]
    random.seed(seed_value)           
    np.random.seed(seed_value) 

    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


    # directories for model checkpoints and RSA's
    # Support custom run directory via environment variable (for hyperparameter search)
    HP_RUN_DIR = os.environ.get("HP_RUN_DIR")
    if HP_RUN_DIR:
        run_dir = HP_RUN_DIR
    else:
        # Build runs directory with optional DGP prefix
        # Human data (sloutsky/palminteri) doesn't use dataset IDs
        _suffix = f"_{args.run_suffix}" if args.run_suffix else ""
        if is_human_data:
            BASE_DIR = f"runs_{DGP}{_suffix}" if latent else f"runs_vanilla_{DGP}{_suffix}"
        elif DGP:
            BASE_DIR = (f"runs_{DGP}_dataset{DATASET_ID}{_suffix}" if latent
                        else f"runs_vanilla_{DGP}_dataset{DATASET_ID}{_suffix}")
        else:
            BASE_DIR = (f"runs_dataset{DATASET_ID}{_suffix}" if latent
                        else f"runs_vanilla_dataset{DATASET_ID}{_suffix}")
        # For outer CV folds, nest under fold{k}/
        if FOLD is not None:
            run_dir = os.path.join(BASE_DIR, f"fold{FOLD}", f"seed_{seed_value}")
        else:
            run_dir = os.path.join(BASE_DIR, f"seed_{seed_value}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    rsa_dir  = os.path.join(run_dir, "rsa")
    lossdir = os.path.join(run_dir, "loss")
    frozen_decoder_dir = os.path.join(run_dir, "frozen_decoder")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(rsa_dir, exist_ok=True)
    os.makedirs(lossdir, exist_ok=True)
    os.makedirs(frozen_decoder_dir, exist_ok=True)

    # HYPERPARAMS
    z_dim = args.z
    lamda = args.lmbd
    # Detect number of actions from data (sloutsky has 4, synthetic has 2)
    A = choice_one_hot_train.shape[-1]
    print(f"Detected A={A} actions from data")
    hidden = args.hidden
    enc_hidden = args.enc_hidden if args.enc_hidden is not None else hidden
    epochs = args.epochs

    # Dimensions already set during data loading above


    # Ensure any previous wandb run is properly closed
    try:
        wandb.finish()
    except Exception:
        pass

    wandb.init(project=wandb_name, config=args)
    # --- Create a config dict ---
    config_dict = vars(args).copy()  # start with CLI arguments

    config_dict.update({
        "epochs": epochs,
        "step1_epochs": args.step1_epochs,
        "hidden": hidden,
        "enc_hidden": enc_hidden,
        "z_dim": z_dim,
        "A": A,
        "sim_nametag": sim_nametag,
        "n_fit_iter": n_fit_iter,
        "in_dim": in_dim
    })

    # Add model architecture config for easy loading
    # Reinitialise the decoder from z2h0(z) at each block boundary so that z
    # must carry participant identity (prevents posterior collapse in multi-block
    # settings where the GRU could otherwise track identity via its hidden state).
    reinit_decoder_per_block = DGP in ("thalmann", "dezfouli")

    config_dict["model_config"] = {
        "in_dim": in_dim,
        "dec_in_dim": dec_in_dim,
        "enc_in_dim": enc_in_dim if is_human_data else in_dim,
        "z_dim": z_dim,
        "hidden": hidden,
        "enc_hidden": enc_hidden,
        "A": A,
        "block_structure": False,
        "n_participants_train": B,
        "n_participants_test": B_test,
        "model_type": "IDRNN" if latent else "Vanilla",
        "task_emb_dim": task_emb_dim,
        "n_tasks": n_tasks,
        "reinit_decoder_per_block": reinit_decoder_per_block,
        "continuous_encoder": args.continuous_encoder,
    }

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    if latent:

        if args.joint:
            # === JOINT TRAINING: encoder + decoder trained together from scratch ===
            print(f"[JOINT TRAINING] beta={args.beta}")

            encoder_IDRNN = IDRNN(in_dim=in_dim, z_dim=z_dim, hid=enc_hidden,
                                  n_tasks=n_tasks, task_emb_dim=task_emb_dim,
                                  continuous_encoder=args.continuous_encoder)
            decoder = Decoder(in_dim=in_dim, z_dim=z_dim, hid=hidden, A=A)
            # Both encoder and decoder are trainable
            joint_model = LatentRNN_secondstep(
                encoder=encoder_IDRNN, hid=hidden, z_dim=z_dim,
                in_dim=in_dim, A=A, decoder=decoder,
                reinit_decoder_per_block=reinit_decoder_per_block
            )
            joint_model.name = "IDRNN_joint"
            joint_model.to(device)

            # For block-structured data (dezfouli) xin is already 4-D; others need unsqueeze
            _needs_block_dim = (xin_train.dim() == 3)
            xenc_train = xin_train.unsqueeze(1)      if _needs_block_dim else xin_train
            y_train    = _y_train_labels.unsqueeze(1) if _needs_block_dim else _y_train_labels
            xenc_test  = xin_test.unsqueeze(1)       if _needs_block_dim else xin_test
            y_test     = _y_test_labels.unsqueeze(1)  if _needs_block_dim else _y_test_labels

            model, best_mu, best_lv, train_elbos, val_elbos, training_dict, pA_rnn_dict = train_IDRNN_joint(
                model=joint_model, xenc=xenc_train, blocks=xenc_train, y=y_train,
                xenc_val=xenc_test, y_val=y_test, p_target=pA, device=device,
                ctest=c_test, ctrain=c_train, alpha_values_test=params,
                checkpoint_dir=ckpt_dir, rsa_dir=rsa_dir, loss_dir=lossdir,
                epochs=epochs, lr=1e-3, beta=args.beta
            )
        else:
            # === TWO-STEP TRAINING: original approach ===
            ids = torch.arange(B)
            idstest = torch.arange(B_test)
            encoder = LookupEncoderZ(n_participants=B, z_dim=z_dim)
            # For thalmann: decoder sees base_in_dim + task_emb_dim per timestep.
            # The task embedding is concatenated inside LatentRNNz/LatentRNN_secondstep.
            decoder = Decoder(in_dim=dec_in_dim, z_dim=z_dim, hid=hidden, A=A)
            _block_structure = (xin_train.dim() == 4)   # True for dezfouli/thalmann, False for sloutsky
            model = LatentRNNz(encoder=encoder, decoder=decoder, hid=hidden, z_dim=z_dim, in_dim=dec_in_dim, A=A,
                               block_structure=_block_structure,
                               n_tasks=n_tasks, task_emb_dim=task_emb_dim).to(device)
            if task_ids_tensor is not None:
                model.set_task_ids(task_ids_tensor)

            # Step 1: Train decoder with lookup embeddings
            step1_epochs = args.step1_epochs
            print(f"[Step 1] Training decoder for {step1_epochs} epochs...")
            if use_inner_cv:
                if has_val:
                    B_val,_,_ = xin_val.shape
                    val_ids = torch.arange(B_val)
                else:
                    val_ids = None
                model, train_loss, val_loss, pA_dict, training_dict = train_latentrnn_noblocks_palminteri(model=model, ids_train=ids, X_train=xin_train,
                y_onehot=c_train, ids_val=val_ids, X_val=xin_val, y_val_onehot=c_val, epoch_nr=step1_epochs, lr=1e-3, weight_decay=1e-4, device=device)
            else:
                model, train_loss, kl_vals, pA_dict, training_dict = train_latentrnn_noblocks(model=model, ids_train=ids, ids_test=idstest, X_train=xin_train,
                y_onehot=choice_one_hot_train, X_test=xin_test, y_test_onehot=choice_one_hot_test,train_alpha_values= params_train, p_target=pA, p_test_target=pA_test, epoch_nr=step1_epochs, lr=1e-3, weight_decay=1e-4, device=device)
            z_lookup = training_dict["z"]

            # Step-1 specificity (matched vs mismatched lookup-z NLL on training
            # subjects) + raw lookup-z dump.  Used downstream to rank seeds when
            # pooling outer-CV NLL across multiple training seeds.  Also used by
            # nested_cv.select_canonical to pick the canonical IDRNN seed
            # (after the no-fold retrain on the full training set).
            if True:
                try:
                    _step1_n_mismatch = 50
                    model.eval()
                    # ids was created on CPU (torch.arange(B)) but the encoder
                    # lookup table lives on `device`; index_select needs both
                    # on the same device.
                    _ids_dev = ids.to(device)
                    with torch.no_grad():
                        _logits_m, _, _ = model(_ids_dev, xin_train)
                        _mask  = (c_train.reshape(-1) >= 0).float()
                        _denom = _mask.sum().clamp(min=1)
                        _log_p = torch.log_softmax(
                            _logits_m.reshape(-1, A), dim=-1)
                        _chosen = _log_p.gather(
                            -1,
                            c_train.reshape(-1).clamp(min=0).long().unsqueeze(-1)
                        ).squeeze(-1)
                        _matched_nll = float(-(_chosen * _mask).sum() / _denom)

                        _n_subj = z_lookup.shape[0]
                        _rng = np.random.default_rng(seed_value)
                        _mis_vals = []
                        for _ in range(_step1_n_mismatch):
                            _perm = _rng.permutation(_n_subj)
                            _same = (_perm == np.arange(_n_subj))
                            while _same.any():
                                _perm[_same] = (_perm[_same] + 1) % _n_subj
                                _same = (_perm == np.arange(_n_subj))
                            _z_mis = z_lookup[
                                torch.as_tensor(_perm,
                                                device=z_lookup.device,
                                                dtype=torch.long)]
                            if xin_train.dim() == 4:
                                _bl = []
                                for _b in range(xin_train.size(1)):
                                    _inp_b = model._append_task_emb(
                                        xin_train[:, _b], _b)
                                    _out, _ = model.decoder(_inp_b, _z_mis)
                                    _bl.append(_out)
                                _logits_mis = torch.stack(_bl, dim=1)
                            else:
                                _inp = model._append_task_emb(xin_train, 0)
                                _logits_mis, _ = model.decoder(_inp, _z_mis)
                            _log_p_mis = torch.log_softmax(
                                _logits_mis.reshape(-1, A), dim=-1)
                            _chosen_mis = _log_p_mis.gather(
                                -1,
                                c_train.reshape(-1).clamp(min=0).long().unsqueeze(-1)
                            ).squeeze(-1)
                            _mis_vals.append(
                                float(-(_chosen_mis * _mask).sum() / _denom))
                        _mismatched_nll = float(np.mean(_mis_vals))
                    _spec = _mismatched_nll - _matched_nll
                    config_dict["step1_specificity"]    = _spec
                    config_dict["step1_matched_nll"]    = _matched_nll
                    config_dict["step1_mismatched_nll"] = _mismatched_nll
                    np.save(os.path.join(run_dir, "step1_z_lookup.npy"),
                            z_lookup.detach().cpu().numpy())
                    with open(config_path, "w") as f:
                        json.dump(config_dict, f, indent=2)
                    print(f"[Step 1] specificity = {_spec:+.4f}  "
                          f"(matched={_matched_nll:.4f}, "
                          f"mismatched={_mismatched_nll:.4f})")
                except Exception as _e:
                    print(f"[Step 1] specificity computation failed: {_e}")

            wandb.finish()
            wandb.init(project=wandb_name, config=args)

            #second step
            # Encoder uses enc_in_dim (raw features only; task embedding added to decoder path)
            encoder_IDRNN = IDRNN(in_dim=enc_in_dim, z_dim=z_dim, hid=enc_hidden,
                                  n_tasks=n_tasks, task_emb_dim=task_emb_dim,
                                  continuous_encoder=args.continuous_encoder)
            frozen_decoder = copy.deepcopy(model.decoder)
            frozen_decoder_path = os.path.join(frozen_decoder_dir, f"policy_model.pt")
            torch.save(model.state_dict(), frozen_decoder_path)
            for p in frozen_decoder.parameters():
                p.requires_grad = False
            # n_tasks=None so __init__ does NOT create a fresh random task_embedding;
            # we copy the Step-1 embedding below and freeze it with the decoder.
            latent_secondstep = LatentRNN_secondstep(encoder=encoder_IDRNN, hid=hidden, z_dim=z_dim,
                                                     in_dim=dec_in_dim, A=A, decoder=frozen_decoder,
                                                     reinit_decoder_per_block=reinit_decoder_per_block)
            latent_secondstep.name = "GRU" # "LatentDistillation"
            # Carry over the task embedding trained in Step 1 and freeze it alongside
            # the decoder.  The decoder weights were optimised with these specific
            # task vectors, so Step 2 must use the same ones.
            if task_ids_tensor is not None and getattr(model, 'task_embedding', None) is not None:
                frozen_task_emb = copy.deepcopy(model.task_embedding)
                for p in frozen_task_emb.parameters():
                    p.requires_grad = False
                latent_secondstep.task_embedding = frozen_task_emb
                latent_secondstep.set_task_ids(task_ids_tensor)
            latent_secondstep.to(device)
         
            # Encoder input: participant-specific dims; decoder (blocks) gets full input
            _needs_block_dim = (xin_train.dim() == 3)
            xenc_train   = xin_enc_train.unsqueeze(1)  if _needs_block_dim else xin_enc_train   # (B, [1,] T, enc_in_dim)
            blocks_train = xin_train.unsqueeze(1)      if _needs_block_dim else xin_train       # (B, [n_blocks,] T, in_dim)
            y_train      = _y_train_labels.unsqueeze(1) if _needs_block_dim else _y_train_labels # (B, [n_blocks,] T)

            lookup_z   = z_lookup.to(device)
            #y_test = torch.argmax(choice_one_hot_test, dim=-1).unsqueeze(1)

            if use_inner_cv:
                if has_val:
                    z_val_lookup = training_dict["z_val"]
                    lookup_z_val = z_val_lookup.to(device)
                    xenc_val_step2 = xin_enc_val.unsqueeze(1)
                    blocks_val_step2 = xin_val.unsqueeze(1)
                    y_val_step2 = torch.argmax(choice_one_hot_val, dim=-1).unsqueeze(1)
                else:
                    lookup_z_val = None
                    xenc_val_step2 = None
                    blocks_val_step2 = None
                    y_val_step2 = None
                # For thalmann: wrap make_model to inject the Step-1 task embedding
                # (frozen) and task_ids into each CV fold model.  The frozen decoder
                # was co-trained with these task vectors, so all Step-2 models must
                # use the same embedding — just like the frozen decoder weights.
                if DGP == "thalmann" and task_emb_dim > 0:
                    _task_ids_cv       = task_ids_tensor
                    _frozen_task_emb   = copy.deepcopy(model.task_embedding)
                    for p in _frozen_task_emb.parameters():
                        p.requires_grad = False
                    _n_tasks_cv        = n_tasks
                    _task_emb_dim_cv   = task_emb_dim

                    def _make_model_thalmann(wrapper_model, encoder, hidden, z_dim, in_dim, A,
                                             decoder, enc_in_dim, enc_hidden, device):
                        m = make_model(wrapper_model, encoder, hidden, z_dim, in_dim, A,
                                       decoder, enc_in_dim, enc_hidden, device,
                                       task_embedding=_frozen_task_emb,
                                       task_ids=_task_ids_cv,
                                       n_tasks=_n_tasks_cv,
                                       task_emb_dim=_task_emb_dim_cv)
                        return m

                    _make_model_fn = _make_model_thalmann
                else:
                    _make_model_fn = make_model

                # For thalmann: weight the decoder NLL so only the restless bandit
                # block (task_id=1) contributes to the policy loss.  The 30 two-armed
                # blocks (task_id=0) are too short (10 trials) to carry reliable
                # individual-difference signal and add noise to the encoder gradient.
                # Weight any non-2-armed block (task_id>0: restless, and horizon
                # in the 3-task model) equally at 1; the short 2-armed task stays
                # at 0.  (task_ids>0).float() == task_ids.float() for the 2-task
                # case, so existing runs are unchanged.
                if DGP == "thalmann" and task_ids_tensor is not None:
                    if args.block_weight_mode == "uniform":
                        _block_weights = torch.ones_like(task_ids_tensor, dtype=torch.float)
                    else:
                        _block_weights = (task_ids_tensor > 0).float()
                else:
                    _block_weights = None

                cv_summary = train_latentrnn_IDRNN_palminteri_CV(LatentRNN_secondstep, IDRNN, hidden, z_dim, dec_in_dim,
                                                                A, frozen_decoder, enc_in_dim, enc_hidden, _make_model_fn, xenc_train,
                                                                blocks_train, y_train, lookup_z, device, xenc_val=xenc_val_step2,
                                                                blocks_val=blocks_val_step2, y_val=y_val_step2,
                                                                z_val_lookup=lookup_z_val, epoch_nr=epochs, patience=600,
                                                                lr=1e-3, checkpoint_dir=ckpt_dir, loss_dir=lossdir, lmbd=lamda,
                                                                nr_splits=3, block_weights=_block_weights,
                                                                unif_weight=args.unif_weight)
                n_epochs_after_cv = cv_summary["selected_epoch"]
                cv_val_loss = float(cv_summary["mean_val_elbos"][n_epochs_after_cv - 1])
                # Record selected epoch and val loss in config so downstream scripts know it was CV-trained
                config_dict["cv_selected_epoch"] = n_epochs_after_cv
                config_dict["cv_val_loss"] = cv_val_loss
                if "mean_val_nll_flats" in cv_summary:
                    config_dict["cv_val_nll"] = float(
                        cv_summary["mean_val_nll_flats"][n_epochs_after_cv - 1]
                    )
                if "mean_z_std_per_epoch" in cv_summary:
                    config_dict["cv_z_std_final"] = float(
                        cv_summary["mean_z_std_per_epoch"][n_epochs_after_cv - 1]
                    )
                    np.save(os.path.join(run_dir, "z_std_curve.npy"),
                            cv_summary["mean_z_std_per_epoch"])
                with open(config_path, "w") as f:
                    json.dump(config_dict, f, indent=2)
                model, best_mu, best_lv, train_elbos, val_elbos, training_dict, pA_rnn_dict = train_final_model_after_cv(LatentRNN_secondstep, IDRNN, frozen_decoder,
                                                                                                                         _make_model_fn, hidden, z_dim, dec_in_dim, A, enc_in_dim,
                                                                                                                         enc_hidden, xenc_train, blocks_train, y_train,
                                                                                                                         lookup_z, n_epochs_after_cv, device, lr=1e-3,
                                                                                                                         checkpoint_dir=ckpt_dir, loss_dir=lossdir, lmbd=lamda,
                                                                                                                         block_weights=_block_weights,
                                                                                                                         unif_weight=args.unif_weight)
                # Always save the CV-selected checkpoint (overwrites stale checkpoints from old architectures)
                final_ckpt_path = os.path.join(ckpt_dir, f"epoch{n_epochs_after_cv:04d}.pt")
                torch.save(training_dict["weights"], final_ckpt_path)

                # ── Step 2.5: brief decoder unfreeze with per-timestep z ─────
                # Enabled when --step2_5_epochs > 0.  The cv-selected
                # checkpoint above is kept as `epoch{N:04d}_pre_step25.pt`;
                # the post-step-2.5 weights overwrite the cv-selected file
                # so testing_script.py picks them up automatically.
                if args.step2_5_epochs > 0:
                    print(f"\n[Step 2.5] brief unfreeze + per-timestep z "
                          f"({args.step2_5_epochs} ep, lr={args.step2_5_lr})")
                    pre_path = os.path.join(
                        ckpt_dir, f"epoch{n_epochs_after_cv:04d}_pre_step25.pt")
                    torch.save(training_dict["weights"], pre_path)
                    model, _step25_losses = train_idrnn_step2_5(
                        model, xenc_train, blocks_train, y_train,
                        n_epochs=args.step2_5_epochs,
                        lr=args.step2_5_lr,
                        block_weights=_block_weights,
                    )
                    torch.save(model.state_dict(), final_ckpt_path)
                    config_dict["step2_5_epochs"]    = args.step2_5_epochs
                    config_dict["step2_5_lr"]        = args.step2_5_lr
                    config_dict["step2_5_final_nll"] = float(_step25_losses[-1])
                    with open(config_path, "w") as f:
                        json.dump(config_dict, f, indent=2)


            else:
                #think about whether I want to implement test data here too
                xenc_test = xin_test.unsqueeze(1)
                y_test    = torch.argmax(choice_one_hot_test, dim=-1).unsqueeze(1)
                print(f"targets: {y_test}")
                model, best_mu, best_lv, train_elbos, val_elbos, training_dict, pA_rnn_dict = train_latentrnn_IDRNN(model=latent_secondstep, xenc=xenc_train,
                blocks=xenc_train, y=y_train, lookup_z=lookup_z, xenc_val=xenc_test, y_val=y_test, p_target= pA, device=device,ctest=c_test, ctrain=c_train,alpha_values_test=params, checkpoint_dir=ckpt_dir,rsa_dir=rsa_dir, loss_dir = lossdir, epochs=epochs, patience=epochs, lr=1e-3, lmbd = lamda)

    else:
        
        _block_structure = (xin_train.dim() == 4)
        model = AblatedRNN(hid=hidden, in_dim=dec_in_dim, A=A, block_structure=_block_structure,
                           n_tasks=n_tasks, task_emb_dim=task_emb_dim).to(device)
        if task_ids_tensor is not None:
            model.set_task_ids(task_ids_tensor)
        if use_inner_cv:
            y_train = _y_train_labels
            # CV to select the optimal number of training epochs
            cv_summary_vanilla = train_ablated_noblocks_palminteri_CV(
                hidden, in_dim, A, xin_train, y_train, device,
                epoch_nr=epochs, lr=1e-3, nr_splits=3,
                dec_in_dim=dec_in_dim, n_tasks=n_tasks,
                task_emb_dim=task_emb_dim, task_ids=task_ids_tensor,
            )
            n_epochs_vanilla_cv = cv_summary_vanilla["selected_epoch"]
            cv_val_loss = float(cv_summary_vanilla["mean_val_elbos"][n_epochs_vanilla_cv - 1])
            # Record selected epoch and val loss in config
            config_dict["cv_selected_epoch"] = n_epochs_vanilla_cv
            config_dict["cv_val_loss"] = cv_val_loss
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            # Train final model on all data for the CV-selected number of epochs
            model, train_loss, val_loss, kl_loss, pA_rnn_dict, training_dict = \
                train_final_ablated_model_after_cv(
                    hidden, in_dim, A, xin_train, y_train, device,
                    n_epochs=n_epochs_vanilla_cv, lr=1e-3,
                    checkpoint_dir=ckpt_dir, loss_dir=lossdir,
                    dec_in_dim=dec_in_dim, n_tasks=n_tasks,
                    task_emb_dim=task_emb_dim, task_ids=task_ids_tensor,
                )
            # Always save the CV-selected checkpoint (overwrites stale checkpoints from old architectures)
            final_ckpt_path = os.path.join(ckpt_dir, f"epoch{n_epochs_vanilla_cv:04d}.pt")
            torch.save(model.state_dict(), final_ckpt_path)
        else:
            print(f"xin test shape: {xin_test.shape}")
            print(f"choice one hot test shape: {choice_one_hot_test.shape}")
            model, train_loss, val_loss, kl_loss, pA_rnn_dict, training_dict = train_ablated_noblocks(
            model, xin_train, choice_one_hot_train, xin_test, choice_one_hot_test, device=device, ctest=c_test, ctrain=c_train, alpha_values_test=params, checkpoint_dir=ckpt_dir, rsa_dir=rsa_dir, loss_dir=lossdir, p_target=pA, p_test=pA_test, epoch_nr = epochs, lr = 1e-3
            )



    pA_rnn_dict = {k: v.detach().cpu().numpy() for k, v in pA_rnn_dict.items()}


    wandb.finish()

    """# Save model configuration for reloading
    model_config = {"in_dim": in_dim, "hid": hidden, "block_structure": False}

    # Save the relevant parts

    if sloutsky or palminteri:
        save_dict = {
        "weights": training_dict["weights"],
        "best_epoch": training_dict["best_epoch"],
        "model_config": model_config
    }
    else:
        save_dict = {
            "weights": training_dict["weights"],
            "best_epoch": training_dict["best_epoch"],
            "best_kl": training_dict["best_kl"],
            "model_config": model_config
        }"""
