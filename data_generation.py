import sim_Q_data as sim
import random
import numpy as np
import pandas as pd

# parameters and variables
seed_base = 1
nTrial = 200
nSession = 200


# --- Data Generation ---
rewardsTrain = sim.gen_reward_seq(seed=seed_base, T=nTrial, pHigh=0.7, pLow=0.3, interval=50)
rewardsTest = sim.gen_reward_seq(seed=seed_base + 1, T=nTrial, pHigh=0.7, pLow=0.3, interval=50)
np.save("data/rewards_train.npy", rewardsTrain)

true_param = sim.generate_parameter_lists(true_model='FQ', ind_diff_type='uniform',
                                          Delta_alpha=0.8, nSession=nSession)

true_param_df = pd.DataFrame(true_param)
true_param_df.to_csv("data/true_parameter_values.csv", index=False)

print("simulating training data")
c, r, pA, Q, CT, df_train, xin_train, choice_one_hot_train, _, _, _ = sim.simulate_Qlearning(
    rewards=rewardsTrain, seed=seed_base + 2, n_sessions=nSession, n_trials=nTrial,
    alphaP_list=true_param['alphaP_list'],
    alphaN_list=true_param['alphaN_list'],
    beta=true_param['beta_list'],
    alphaF_list=true_param['alphaF_list'],
    phi_list=true_param['phi_list'],
    tau_list=true_param['tau_list']
)

print("simulating test data")
c_test, r_test, pA_test, _, _, df_test, xin_test, choice_one_hot_test, normalized_LL_test, session_ll_df_test, _ = sim.simulate_Qlearning(
    rewards=rewardsTest, seed=seed_base + 10, n_sessions=nSession, n_trials=nTrial,
    alphaP_list=true_param['alphaP_list'],
    alphaN_list=true_param['alphaN_list'],
    beta=true_param['beta_list'],
    alphaF_list=true_param['alphaF_list'],
    phi_list=true_param['phi_list'],
    tau_list=true_param['tau_list']
)
print(f"xin_train shape: {xin_train.shape}")
print(f"choice_one_hot_test shape: {choice_one_hot_test.shape}")
print(f"choice_one_hot_test: {choice_one_hot_test}")

#save data externally
df_train.to_csv("data/df_train.csv", index=False)
df_test.to_csv("data/df_test.csv", index=False)
session_ll_df_test.to_csv("data/session_ll_df_test.csv", index=False)
np.save('data/xin_train.npy', xin_train)
np.save('data/xin_test.npy', xin_test)
np.save('data/choice_one_hot_train.npy', choice_one_hot_train)
np.save('data/choice_one_hot_test.npy', choice_one_hot_test)
np.save('data/c_test.npy', c_test)
np.save('data/pA_train.npy', pA)
np.save('data/pA_test.npy', pA_test)
np.save('data/c_train.npy', c)