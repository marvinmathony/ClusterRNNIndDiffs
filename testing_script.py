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
args = parser.parse_args()

############################
#### LOAD CORRECT MODEL ####
############################
latent = args.latent
DATASET_ID = args.dataset_id
model_fitting = args.model_fitting
palminteri = False
sloutsky = False

DATA_DIR = f"data_dataset{DATASET_ID}"
n_fit_iter = 5
vanilla_nametag = "vanilla"
latent_nametag = "latentmodel"

# Load data
df_train = pd.read_csv(f"{DATA_DIR}/df_train.csv")
df_test = pd.read_csv(f"{DATA_DIR}/df_test.csv")
session_ll_df_test = pd.read_csv(f"{DATA_DIR}/session_ll_df_test.csv")

xin_train = np.load(f"{DATA_DIR}/xin_train.npy")
xin_test = np.load(f"{DATA_DIR}/xin_test.npy")
c_test = np.load(f"{DATA_DIR}/c_test.npy")
c_train = np.load(f"{DATA_DIR}/c_train.npy")

xin_train = torch.from_numpy(xin_train).float().to(device)
xin_test = torch.from_numpy(xin_test).float().to(device)
c_test = torch.from_numpy(c_test).float().to(device)
c_train = torch.from_numpy(c_train).float().to(device)


BASE_DIR = f"runs_dataset{DATASET_ID}" if latent else f"runs_vanilla_dataset{DATASET_ID}"

# Load best epoch
with open(os.path.join(BASE_DIR, "best_epoch_by_specificity.json"), "r") as f:
    meta = json.load(f)

best_epoch = meta["best_epoch"]
SEED_FOR_ANALYSIS = meta["best_seed"]
print(f"Using epoch {best_epoch:04d} from seed {SEED_FOR_ANALYSIS} for analysis.")

#### LOAD MODEL PARAMETERS ####
run_dir = f"{BASE_DIR}/seed_{SEED_FOR_ANALYSIS}"  # example
config_path = os.path.join(run_dir, "config.json")

with open(config_path, "r") as f:
    cfg = json.load(f)


#### LOAD CORRECT MODEL ARCHITECTURE AND MODEL WEIGHTS ####
if latent:
    encoder_IDRNN = IDRNN(in_dim=cfg["in_dim"], z_dim=cfg["z_dim"], hid=cfg["hidden"])
    frozen_decoder = Decoder(in_dim=cfg["in_dim"], z_dim=cfg["z_dim"], hid=cfg["hidden"])
    for p in frozen_decoder.parameters():
        p.requires_grad = False
    model = LatentRNN_secondstep(encoder=encoder_IDRNN, hid=cfg["hidden"], z_dim=cfg["z_dim"],in_dim=cfg["in_dim"], A=cfg["A"],
    decoder=frozen_decoder)
    model.to(device)
else:
    model = AblatedRNN(hid=cfg["hidden"], in_dim=cfg["in_dim"], A=cfg["A"], block_structure=False).to(device)

ckpt_path = os.path.join(
    BASE_DIR,
    f"seed_{SEED_FOR_ANALYSIS}",
    "checkpoints",
    f"epoch{best_epoch:04d}.pt"
)
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

####################


rnn_df, latent_tensor, latent_tensor_train, geometric_mean_prob_rnn = compute_rnn_likelihoods_torch(test_latentrnn_secondstep_causal_posterior_weighting, model, xin_test, c_test, xin_train, latent=latent, choice_train=c_train, id=True)
if latent:
    rnn_no_id, _, _, geometric_mean_prob_common_process  = compute_rnn_likelihoods_torch(test_latentrnn_secondstep_causal_posterior_weighting, model, xin_test, c_test, xin_train, latent=latent, choice_train=c_train, id=False)
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


# --- Append to CSV or create it if missing ---
if not os.path.exists(out_path):
    df.to_csv(out_path, index=False)
else:
    df.to_csv(out_path, mode='a', header=False, index=False)

print(f"Saved results in {out_path}")


# --- Fit Cognitive Models ---
model_configs = {
    "Q": {"asymmetric_alpha": False, "forgetting_type": "none", "choice_trace": False},
    "FQ": {"asymmetric_alpha": False, "forgetting_type": "fixed", "choice_trace": False}
}
if model_fitting:

    model_eval_Q_df, params_dict, p1_common_dict, p1_ML_dict, p1_MAP_dict, q_common_dict = fit.fit_all_models(
        model_configs, df_train, df_test, n_iter=n_fit_iter, fit_ML=False
    )
    

    with open(f"{DATA_DIR}/params_dict.pkl", "wb") as f:
        pickle.dump(params_dict, f)

    df = pd.DataFrame({
    "model": list(params_dict.keys()),
    "params": [list(v) for v in params_dict.values()]})

    df.to_csv(f"{DATA_DIR}/params_dict.csv", index=False)

    np.savez(f"{DATA_DIR}/q_common_dict.npz", **q_common_dict)
    print(f"q value dict saved under {DATA_DIR}/q_common_dict.npz ✅")

    print(f"model_eval_Q_df: {model_eval_Q_df}")
    print(f"session_ll_df_test: {session_ll_df_test}")
    print(f"rnn_df: {rnn_df}")
    #print(f"rnn_df: {rnn_no_id}")
    ### run ID and noID and append both here. This can then be used for plotting
    if latent:

        model_eval_df = pd.concat([model_eval_Q_df, session_ll_df_test, rnn_df, rnn_no_id], ignore_index=True)
    else:
        model_eval_df = pd.concat([model_eval_Q_df, session_ll_df_test, rnn_df], ignore_index=True)


    # Save p1 values (for later plotting)
    np.savez(f"{DATA_DIR}/p1_common_dict.npz", **p1_common_dict)

if latent:
    rnn_df.to_csv(f"{DATA_DIR}/rnn_results{latent_nametag}.csv", index=False)
    rnn_no_id.to_csv(f"{DATA_DIR}/rnn_results_common_process.csv", index=False)
    if palminteri:
        torch.save(latent_tensor, f"{DATA_DIR}/latents_tensor{latent_nametag}_palminteri.pt")
        torch.save(latent_tensor_train, f"{DATA_DIR}/latents_tensor{latent_nametag}_palminteri_traindata.pt")
        print(f"latent tensor with dim {latent_tensor.shape} saved under {DATA_DIR}/latents_tensor_palminteri.pt")
        model_eval_df.to_csv(f"{DATA_DIR}/model_eval_df{latent_nametag}.csv", index=False)
    elif sloutsky:
        torch.save(latent_tensor, f"{DATA_DIR}/latents_tensor{latent_nametag}_sloutsky.pt")
        torch.save(latent_tensor_train, f"{DATA_DIR}/latents_tensor{latent_nametag}_sloutsky_traindata.pt")
        print(f"latent tensor with dim {latent_tensor.shape} saved under {DATA_DIR}/latents_tensor_sloutsky.pt")
    else:
        torch.save(latent_tensor, f"{DATA_DIR}/latents_tensor{latent_nametag}{SEED_FOR_ANALYSIS}.pt")
        torch.save(latent_tensor_train, f"{DATA_DIR}/latents_tensor{latent_nametag}_traindata.pt")
        if model_fitting:

            model_eval_df.to_csv(f"{DATA_DIR}/model_eval_df{latent_nametag}.csv", index=False)

else:
    rnn_df.to_csv(f"{DATA_DIR}/rnn_results{vanilla_nametag}.csv", index=False)
    if palminteri:

        torch.save(latent_tensor, f"{DATA_DIR}/latents_tensor_palminteri.pt")
        model_eval_df.to_csv(f"{DATA_DIR}/model_eval_df.csv", index=False)
    elif sloutsky:
        torch.save(latent_tensor, f"{DATA_DIR}/latents_tensor{vanilla_nametag}_sloutsky.pt")
        torch.save(latent_tensor_train, f"{DATA_DIR}/latents_tensor{vanilla_nametag}_sloutsky_traindata.pt")
    else:
        torch.save(latent_tensor, f"{DATA_DIR}/latents_tensor{vanilla_nametag}{SEED_FOR_ANALYSIS}.pt")
        print(f"latent saved under {DATA_DIR}/latents_tensor{vanilla_nametag}{SEED_FOR_ANALYSIS}.pt")
        if model_fitting:

            model_eval_df.to_csv(f"{DATA_DIR}/model_eval_df{vanilla_nametag}.csv", index=False)

    ### insted of the below, do this during training for the vanilla model ###
    #torch.save(save_dict, "checkpoints/ablated_rnn_best.pt")
    #torch.save(model.state_dict(), "checkpoints/ablated_rnn.pt")
    #np.savez("data/pA_rnn_dict.npz", **pA_rnn_dict)
    #np.save("data/training_dict.npy", training_dict)




print("✅ Training and fitting done. Results saved to data/ and checkpoints/")




# Now do your analysis with this model:
# - compute latents
# - RSA vs behavior
# - hidden state visualizations, etc.
