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
palminteri = False
sloutsky = False
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

elif sloutsky:
    df_train = pd.read_csv("data/train_df_sloutsky.csv")
    df_test = pd.read_csv("data/test_df_sloutsky.csv")
    df_val = pd.read_csv("data/val_df_sloutsky.csv")
    df_train = df_train.rename(columns={'subid': 'session'})
    df_test = df_test.rename(columns={'subid': 'session'})
    df_val = df_val.rename(columns={'subid': 'session'})
    # ---- load training arrays ----
    xin_train = np.load("data/train_xin_sloutsky.npy", allow_pickle=True)
    choice_one_hot_train = np.load("data/train_choice_one_hot_sloutsky.npy", allow_pickle=True)
    c_train = np.load("data/c_train_sloutsky.npy", allow_pickle=True)

    # ---- load validation arrays ----
    xin_val = np.load("data/val_xin_sloutsky.npy", allow_pickle=True)
    choice_one_hot_val = np.load("data/val_choice_one_hot_sloutsky.npy", allow_pickle=True)
    c_val = np.load("data/c_val_sloutsky.npy", allow_pickle=True)

    # ---- load test arrays ----
    xin_test = np.load("data/test_xin_sloutsky.npy", allow_pickle=True)
    choice_one_hot_test = np.load("data/test_choice_one_hot_sloutsky.npy", allow_pickle=True)
    c_test = np.load("data/c_test_sloutsky.npy", allow_pickle=True)
    pA = np.load("data/pA_train.npy") #not relevant here, but just so that we don't have to adapt script too much
    pA_test = np.load("data/pA_test.npy") #not relevant here, but just so that we don't have to adapt script too much
    c_train = torch.from_numpy(c_train).float().to(device)
    c_val = torch.from_numpy(c_val).float().to(device)
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
    parser.add_argument('--z', type=int, help="dimension of latent space", default=1)
    parser.add_argument('--seed', type=int, help="random seed", default=42)
    parser.add_argument('--latent', type=lambda x: x.lower() == 'true', help="latent or vanilla modeling", default=False)
    parser.add_argument('--dataset_id', type=int, help="dataset ID for multi-dataset experiments", default=0)
    parser.add_argument('--joint', type=lambda x: x.lower() == 'true', help="joint training (True) vs two-step (False)", default=False)
    parser.add_argument('--beta', type=float, help="KL weight for joint training (beta-VAE style)", default=0.1)
    parser.add_argument('--epochs', type=int, help="number of training epochs", default=10000)
    args = parser.parse_args()

    DATASET_ID = args.dataset_id
    DATA_DIR = f"data_dataset{DATASET_ID}"
    PLOT_DIR = f"plots_dataset{DATASET_ID}"

    wandb_name = "RNNIndDiffs"

    # Load data with dataset_id
    if not (palminteri or sloutsky):
        # --- Load CSVs ---
        df_train = pd.read_csv(f"{DATA_DIR}/df_train.csv")
        df_test = pd.read_csv(f"{DATA_DIR}/df_test.csv")
        session_ll_df_test = pd.read_csv(f"{DATA_DIR}/session_ll_df_test.csv")
        test_parameter_df = pd.read_csv(f"{DATA_DIR}/true_test_parameter_values.csv")
        train_parameter_df = pd.read_csv(f"{DATA_DIR}/true_parameter_values.csv")

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
    if HP_RUN_DIR and latent:
        run_dir = HP_RUN_DIR
    else:
        BASE_DIR = f"runs_dataset{DATASET_ID}" if latent else f"runs_vanilla_dataset{DATASET_ID}"
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
    A = 2
    hidden = 10
    epochs = args.epochs

    # Get dimensions from loaded data
    if not (palminteri or sloutsky):
        in_dim = xin_train.shape[2]
        B = xin_train.shape[0]
        B_test = xin_test.shape[0]
        T = xin_train.shape[1]
        T_test = xin_test.shape[1]
        in_dim_test = xin_test.shape[2]


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
        "hidden": hidden,
        "z_dim": z_dim,
        "A": A,
        "sim_nametag": sim_nametag,
        "n_fit_iter": n_fit_iter,
        "in_dim": in_dim
    })

    # Add model architecture config for easy loading
    config_dict["model_config"] = {
        "in_dim": in_dim,
        "z_dim": z_dim,
        "hidden": hidden,
        "A": A,
        "block_structure": False,
        "n_participants_train": B,
        "n_participants_test": B_test,
        "model_type": "IDRNN" if latent else "Vanilla"
    }

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    if latent:

        if args.joint:
            # === JOINT TRAINING: encoder + decoder trained together from scratch ===
            print(f"[JOINT TRAINING] beta={args.beta}")

            encoder_IDRNN = IDRNN(in_dim=in_dim, z_dim=z_dim, hid=hidden)
            decoder = Decoder(in_dim=in_dim, z_dim=z_dim, hid=hidden)
            # Both encoder and decoder are trainable
            joint_model = LatentRNN_secondstep(
                encoder=encoder_IDRNN, hid=hidden, z_dim=z_dim,
                in_dim=in_dim, A=A, decoder=decoder
            )
            joint_model.name = "IDRNN_joint"
            joint_model.to(device)

            xenc_train = xin_train.unsqueeze(1)
            y_train = torch.argmax(choice_one_hot_train, dim=-1).unsqueeze(1)
            xenc_test = xin_test.unsqueeze(1)
            y_test = torch.argmax(choice_one_hot_test, dim=-1).unsqueeze(1)

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
            decoder = Decoder(in_dim=in_dim, z_dim=z_dim, hid=hidden)
            model = LatentRNNz(encoder=encoder, decoder = decoder, hid=hidden, z_dim=z_dim, in_dim=in_dim, A=A, block_structure=False).to(device)

            if palminteri or sloutsky:
                B_val,_,_ = xin_val.shape
                val_ids = torch.arange(B_val)
                model, train_loss, val_loss, pA_dict, training_dict = train_latentrnn_noblocks_palminteri(model=model, ids_train=ids, X_train=xin_train,
                y_onehot=c_train, ids_val=val_ids, X_val=xin_val, y_val_onehot=c_val, epochs=epochs, lr=1e-3, weight_decay=1e-4, device=device)
            else:
                model, train_loss, kl_vals, pA_dict, training_dict = train_latentrnn_noblocks(model=model, ids_train=ids, ids_test=idstest, X_train=xin_train,
                y_onehot=choice_one_hot_train, X_test=xin_test, y_test_onehot=choice_one_hot_test,train_alpha_values= params_train, p_target=pA, p_test_target=pA_test, epochs=epochs, lr=1e-3, weight_decay=1e-4, device=device)
            z_lookup = training_dict["z"]
            wandb.finish()
            wandb.init(project=wandb_name, config=args)

            #second step
            encoder_IDRNN = IDRNN(in_dim=in_dim, z_dim=z_dim, hid=10)
            frozen_decoder = copy.deepcopy(model.decoder)
            frozen_decoder_path = os.path.join(frozen_decoder_dir, f"policy_model.pt")
            torch.save(model.state_dict(), frozen_decoder_path)
            for p in frozen_decoder.parameters():
                p.requires_grad = False
            latent_secondstep = LatentRNN_secondstep(encoder=encoder_IDRNN, hid=hidden, z_dim=z_dim,in_dim=in_dim, A=A,
            decoder=frozen_decoder)
            latent_secondstep.name = "GRU" # "LatentDistillation"
            latent_secondstep.to(device)
            xenc_train = xin_train.unsqueeze(1)                                # (B, 1, T, 4)
            y_train    = torch.argmax(choice_one_hot_train, dim=-1).unsqueeze(1)  # (B, 1, T)

            lookup_z   = z_lookup.to(device)
            #y_test = torch.argmax(choice_one_hot_test, dim=-1).unsqueeze(1)

            if palminteri or sloutsky:
                z_val_lookup = training_dict["z_val"]
                lookup_z_val = z_val_lookup.to(device)
                xenc_val = xin_val.unsqueeze(1)
                y_val = torch.argmax(choice_one_hot_val, dim=-1).unsqueeze(1)
                model, best_mu, best_lv, train_elbos, val_elbos, training_dict, pA_rnn_dict = train_latentrnn_IDRNN_palminteri(model=latent_secondstep, xenc=xenc_train,
                blocks=xenc_train, y=y_train, lookup_z=lookup_z, xenc_val=xenc_val, y_val=y_val, z_val_lookup=lookup_z_val, epochs=5000, patience=600, lr=1e-3)
            else:
                #think about whether I want to implement test data here too
                xenc_test = xin_test.unsqueeze(1)
                y_test    = torch.argmax(choice_one_hot_test, dim=-1).unsqueeze(1)
                print(f"targets: {y_test}")
                model, best_mu, best_lv, train_elbos, val_elbos, training_dict, pA_rnn_dict = train_latentrnn_IDRNN(model=latent_secondstep, xenc=xenc_train,
                blocks=xenc_train, y=y_train, lookup_z=lookup_z, xenc_val=xenc_test, y_val=y_test, p_target= pA, device=device,ctest=c_test, ctrain=c_train,alpha_values_test=params, checkpoint_dir=ckpt_dir,rsa_dir=rsa_dir, loss_dir = lossdir, epochs=epochs, patience=epochs, lr=1e-3, lmbd = lamda)

    else:
        
        model = AblatedRNN(hid=hidden, in_dim=in_dim, A=A, block_structure=False).to(device)
        if palminteri or sloutsky:
            y_train    = torch.argmax(choice_one_hot_train, dim=-1)
            y_val = torch.argmax(choice_one_hot_val, dim=-1)
            model, train_loss, val_loss, kl_loss, pA_rnn_dict, training_dict = train_ablated_noblocks_palminteri(
            model, xin_train, y_train, xin_val, y_val, epochs = epochs, lr = 1e-3
            )
        else:
            print(f"xin test shape: {xin_test.shape}")
            print(f"choice one hot test shape: {choice_one_hot_test.shape}")
            model, train_loss, val_loss, kl_loss, pA_rnn_dict, training_dict = train_ablated_noblocks(
            model, xin_train, choice_one_hot_train, xin_test, choice_one_hot_test, device=device, ctest=c_test, ctrain=c_train, alpha_values_test=params, checkpoint_dir=ckpt_dir, rsa_dir=rsa_dir, loss_dir=lossdir, p_target=pA, p_test=pA_test, epochs = epochs, lr = 1e-3
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
