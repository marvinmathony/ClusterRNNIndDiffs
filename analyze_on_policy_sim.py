import os, numpy as np, pandas as pd
import torch, seaborn as sns, matplotlib.pyplot as plt
import torch.nn as nn
from scipy.optimize import minimize, differential_evolution
from kalman_simulation import estimate_shared_model, negative_log_likelihood_IDtest, negative_log_likelihood, negative_log_likelihood_IDtest_onlybeta, negative_log_likelihood_onlybeta


sim_nametag_load = "vanillaRNN"
sim_nametag_save = "test6"
# Assuming df_simulation is your combined CSV data
df = pd.read_csv(f"data/IDtest{sim_nametag_load}.csv")
#df = pd.read_csv("data/simulated_data_withID.csv")
input_threshold = 50

beta_true = df["beta_true"]

plt.hist(beta_true, bins=20, color="salmon", edgecolor="black")
plt.xlabel("γ (gamma)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("True Beta distribution", fontsize=14)
plt.tight_layout()
plt.show()
plt.savefig(f"plots/histogram_true_beta_{sim_nametag_save}.png")
print("Plotted true beta distribution")

"""class DifferentiableKalmanProbit(nn.Module):
    def __init__(self, noise_var=1.0, initial_var=10.0, trials_per_block=200):
        super().__init__()
        self.noise_var = noise_var
        self.initial_var = initial_var
        self.trials_per_block = trials_per_block

    def forward(self, actions, rewards, beta, gamma):
        
        
        actions: (B, T)
        rewards: (B, T)
        beta, gamma: (B, 1) or (B,) parameters
        
        
        B, T = actions.shape
        device = actions.device
        #print(f"Inside model forward: actions shape {actions.shape}, rewards shape {rewards.shape}, beta shape {beta.shape}, gamma shape {gamma.shape}")

        # Initialize Kalman variables
        mean = torch.zeros(B, 2, T + 1, device=device)  # (B, 2, T+1)
        var  = torch.full((B, 2, T + 1), self.initial_var, device=device)  # (B, 2, T+1)

        log_probs = []

        for t in range(T):
            # Reset beliefs at block start
            if t % self.trials_per_block == 0:
                mean[:, :, t] = 0.0
                var[:, :, t] = self.initial_var

            mu0 = mean[:, 0, t]
            mu1 = mean[:, 1, t]
            v0  = var[:, 0, t]
            v1  = var[:, 1, t]

            V_t = mu0 - mu1
            TU = torch.sqrt(v0 + v1 + 1e-8)
            RU = (torch.sqrt(v0 + 1e-8) - torch.sqrt(v1 + 1e-8))
            beta = beta.squeeze(-1)   # (B,)
            gamma = gamma.squeeze(-1) # (B,)

            z = beta * (V_t / TU) + gamma * RU
            #print(f"t={t}, V_t shape: {V_t.shape}, TU shape: {TU.shape}, RU shape: {RU.shape}, z shape: {z.shape}")
            p0 = torch.distributions.Normal(0, 1).cdf(z)
            p = torch.stack([p0, 1 - p0], dim=1)  # (B, 2)
            a_t = actions[:, t]  # (B,)
            #print(f"shape of p: {p.shape}, shape of a_t: {a_t.shape}")
            log_p = torch.log(torch.gather(p, 1, a_t.unsqueeze(1)) + 1e-8).squeeze(1)
            log_probs.append(log_p)

            # Kalman update
            r_t = rewards[:, t]
            chosen = actions[:, t]  # 0 or 1
            unchosen = 1 - chosen

            v_chosen = var[torch.arange(B), chosen, t]
            m_chosen = mean[torch.arange(B), chosen, t]
            k_t = v_chosen / (v_chosen + self.noise_var)
            new_mean = m_chosen + k_t * (r_t - m_chosen)
            new_var  = (1 - k_t) * v_chosen

            # Update means and vars for next timestep
            mean[torch.arange(B), chosen, t + 1] = new_mean
            mean[torch.arange(B), unchosen, t + 1] = mean[torch.arange(B), unchosen, t]
            var[torch.arange(B), chosen, t + 1] = torch.clamp(new_var, min=0.1)
            var[torch.arange(B), unchosen, t + 1] = var[torch.arange(B), unchosen, t]

        return torch.stack(log_probs, dim=1)  # (B, T)


def fit_participant_params(actions, rewards, trials_per_block=200, mask=None,
                           lr=0.3, steps=2000, weight_decay=0.0, verbose=True):
     
   actions, rewards: tensors (P,T) for P participants

   mask: optional (P,T) bool/0-1 to ignore warm-up trials (e.g. Trial < input_threshold)

   returns beta_hat, gamma_hat: (P,) tensors (on same device)
   
    

    device = actions.device
    P, T = actions.shape

    model = DifferentiableKalmanProbit(trials_per_block=trials_per_block).to(device)

    # softplus params to keep positivity
    sp = nn.Softplus()
    invsp = lambda x: torch.log(torch.exp(x) - 1.0)

    # init near reasonable values (e.g., 1.0)
    beta_raw  = nn.Parameter(invsp(torch.full((P,1), 2.5, device=device)))
    #gamma_raw = nn.Parameter(invsp(torch.full((P,1), 2.5, device=device)))
    gamma = torch.full((P,1), 2.5, device=device)

    optim = torch.optim.Adam([beta_raw], lr=lr, weight_decay=weight_decay)

    if mask is None:
        mask = torch.ones_like(actions, dtype=torch.float32, device=device)
    else:
        mask = mask.float().to(device)

    for step in range(steps):
        optim.zero_grad()
        beta  = sp(beta_raw)   # (P,1)
        #gamma = sp(gamma_raw)  # (P,1)

        log_probs = model(actions, rewards, beta, gamma)  # (P,T)
        # negative log-likelihood with masking (exclude warm-up)
        nll = -(log_probs * mask).sum() / mask.sum().clamp_min(1.0)

        nll.backward()
        optim.step()

        if verbose and (step % 100 == 0 or step == steps-1):
            with torch.no_grad():
                b_mean = beta.mean().item(); g_mean = gamma.mean().item()
                print(f"step {step:4d}  NLL={nll.item():.4f}   mean β={b_mean:.3f}  mean γ={g_mean:.3f}")

    with torch.no_grad():
        beta_hat  = sp(beta_raw).squeeze(-1)
        gamma_hat = gamma #sp(gamma_raw).squeeze(-1)
    return beta_hat, gamma_hat


# keep only model-generated trials
df = df[df.Trial >= input_threshold].reset_index(drop=True)


P = df["participant"].nunique()
T = df.groupby("participant").size().max()
#print(f"Number of participants: {P}, max trials per participant: {T}")

# build (P,T) arrays (assumes complete data; if ragged, pad and build a mask)
actions = torch.full((P, T), 0, dtype=torch.long)
rewards = torch.zeros((P, T))
for pid, g in df.groupby("participant"):
    g = g.sort_values(["Block","Trial"])
    a = torch.tensor(g["Action"].to_numpy(), dtype=torch.long)
    r = torch.tensor(g["Reward"].to_numpy(), dtype=torch.float32)
    actions[pid,:len(a)] = a
    rewards[pid,:len(r)] = r
mask = torch.ones_like(actions, dtype=torch.float32)  # already filtered; else zero out warm-up cols

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actions = actions.to(device); rewards = rewards.to(device); mask = mask.to(device)

beta_hat, gamma_hat = fit_participant_params(actions, rewards,
                                             trials_per_block=200,
                                             mask=mask,
                                             lr=0.05, steps=300, verbose=True)

print(beta_hat.cpu().numpy()[:10], gamma_hat.cpu().numpy()[:10])

# Assuming beta_hat, gamma_hat are torch tensors (P,)
beta_np  = beta_hat.detach().cpu().numpy()
gamma_np = gamma_hat.detach().cpu().numpy()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(beta_np, bins=20, color="skyblue", edgecolor="black")
plt.xlabel("β (beta)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Recovered Beta per Participant", fontsize=14)

plt.subplot(1, 2, 2)
plt.hist(gamma_np, bins=20, color="salmon", edgecolor="black")
plt.xlabel("γ (gamma)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Recovered Gamma per Participant", fontsize=14)

plt.tight_layout()
plt.show()
plt.savefig(f"plots/histogram_recovered_gammaandbeta_{sim_nametag_save}.png")




"""











participant_params = []

### FITTING BOTH BETA AND GAMMA ###
"""def fit_participant(df_p, n_starts=3):
    best_result = None
    best_nll = np.inf
    
    for _ in range(n_starts):
        print("Optimization start...")
        x0 = np.random.uniform(low=[0.1, 0.1], high=[12, 12])  # random start in plausible range
        global_result = differential_evolution(
        func=lambda x: negative_log_likelihood_IDtest(x, df_p),
        bounds=[(0.1, 12), (0.1, 12)],
        maxiter=1000
        )
        local_result = minimize(
        negative_log_likelihood_IDtest,
        x0=x0,
        args=(df_p,),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-6, "gtol": 1e-6}
        )
        result = minimize(
            negative_log_likelihood_IDtest,
            x0=x0,
            args=(df_p,),
            method="Powell",
            bounds=[(0.1, 12), (0.1, 12)],  # bounds help keep parameters in sane range
            options={"maxiter": 2000}
        )
        if local_result.fun < best_nll:
            best_result = local_result
            best_nll = local_result.fun
    print(f"Best optimization result for participant {df_p['participant'].iloc[0]}: x={best_result.x}, fun={best_result.fun}")

    return best_result.x, best_nll
    #print(f"Local optimization result: x={local_result.x}, fun={local_result.fun}")
    #return local_result.x, local_result.fun

for pid, df_p in df.groupby("participant"):
    #df_p = df_p[df_p.Trial >= input_threshold]
    print(df_p['beta_true'].iloc[0], df_p['gamma_true'].iloc[0])
    print(f"likelihood test: {negative_log_likelihood_IDtest([2,2], df_p)}")
    result = fit_participant(df_p, n_starts=1)
    beta_hat, gamma_hat = result[0]
    participant_params.append({"participant": pid, "beta_hat": beta_hat, "gamma_hat": gamma_hat})

true_params = df.groupby('participant')[['beta_true', 'gamma_true']].first().reset_index()
df_fits = pd.DataFrame(participant_params)
df_combined = df_fits.merge(true_params, on='participant')
print(df_fits.head())"""

### FITTING ONLY BETA ###
def fit_participant(df_p, n_starts=3):
    best_result = None
    best_nll = np.inf
    
    for _ in range(n_starts):
        print("Optimization start...")
        x0 = np.random.uniform(low=0.1, high=6)  # random start in plausible range
        global_result = differential_evolution(
        func=lambda x: negative_log_likelihood_IDtest_onlybeta(x, df_p),
        bounds=[(0.5, 6)],
        maxiter=1000
        )
        print(f"global result: {global_result.x}")
        local_result = minimize(
        negative_log_likelihood_IDtest_onlybeta,
        x0=global_result.x,
        args=(df_p,),
        method='SLSQP',
        bounds = [(0.5, 5.0)],
        tol=1e-4
        )
        """result = minimize(
            negative_log_likelihood_IDtest,
            x0=x0,
            args=(df_p,),
            method="Powell",
            bounds=[(0.1, 12), (0.1, 12)],  # bounds help keep parameters in sane range
            options={"maxiter": 2000}
        )"""
        if local_result.fun < best_nll:
            best_result = local_result
            best_nll = local_result.fun
    print(f"Best optimization result for participant {df_p['participant'].iloc[0]}: x={best_result.x}, fun={best_result.fun}")

    return best_result.x, best_nll
    #print(f"Local optimization result: x={local_result.x}, fun={local_result.fun}")
    #return local_result.x, local_result.fun

for pid, df_p in df.groupby("participant"):
    #df_p = df_p[df_p.Trial >= input_threshold]
    print(df_p['beta_true'].iloc[0], df_p['gamma_true'].iloc[0])
    print(f"likelihood test: {negative_log_likelihood_IDtest_onlybeta(2, df_p)}")
    result = fit_participant(df_p, n_starts=10)
    beta_hat = result[0]
    participant_params.append({"participant": pid, "beta_hat": beta_hat})

true_params = df.groupby('participant')['beta_true'].first().reset_index()
df_fits = pd.DataFrame(participant_params)
df_combined = df_fits.merge(true_params, on='participant')
print(df_fits.head())


# Create side-by-side histograms
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Beta comparison
axes[0, 0].hist(df_combined['beta_true'], bins=20, color='lightcoral', alpha=0.7, edgecolor='black', label='True')
axes[0, 0].set_xlabel('β (beta)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('True Beta Distribution')
axes[0, 0].legend()

axes[0, 1].hist(df_combined['beta_hat'], bins=20, color='skyblue', alpha=0.7, edgecolor='black', label='Recovered')
axes[0, 1].set_xlabel('β (beta)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Recovered Beta Distribution')
axes[0, 1].legend()

# Gamma comparison
"""axes[1, 0].hist(df_combined['gamma_true'], bins=20, color='lightcoral', alpha=0.7, edgecolor='black', label='True')
axes[1, 0].set_xlabel('γ (gamma)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('True Gamma Distribution')
axes[1, 0].legend()

axes[1, 1].hist(df_combined['gamma_hat'], bins=20, color='salmon', alpha=0.7, edgecolor='black', label='Recovered')
axes[1, 1].set_xlabel('γ (gamma)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Recovered Gamma Distribution')
axes[1, 1].legend()"""

plt.tight_layout()
plt.savefig(f"plots/true_vs_recovered_params_{sim_nametag_save}.png")
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df_fits["beta_hat"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("β (beta)")
plt.ylabel("Frequency")
plt.title("Recovered Beta per Participant")
#plt.savefig(f"plots/histogram_recovered_beta_{sim_nametag}.png")

"""plt.subplot(1, 2, 2)
plt.hist(df_fits["gamma_hat"], bins=20, color="salmon", edgecolor="black")
plt.xlabel("γ (gamma)")
plt.ylabel("Frequency")
plt.title("Recovered Gamma per Participant")
plt.tight_layout()
plt.show()
plt.savefig(f"plots/histogram_recovered_gamma_{sim_nametag_save}.png")"""
