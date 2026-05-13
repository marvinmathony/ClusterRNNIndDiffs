import os, json, numpy as np, pandas as pd, torch
from modelsandtraining import IDRNN

DGP = 'thalmann'; FOLD = 0; T_TASK0_LAST = 9
task_ids_global = torch.tensor(np.load('data_thalmann/task_ids_per_block.npy'), dtype=torch.long)
xin_test = np.load('data_thalmann/fold0/xin_test.npy')
subids = pd.read_csv('data_thalmann/fold0/df_test.csv')['subid'].values
quest = pd.read_csv('data/finalQuestionnaireDataSession1.csv').set_index('ID')
quest['PHQ'] = quest[['PHQ_9_%d' % i for i in range(10)]].mean(1)
phq = quest.reindex(subids)['PHQ'].values.astype(float)
valid = ~np.isnan(phq)
xin_v, phq_v = xin_test[valid], phq[valid]
xin_t = torch.tensor(xin_v, dtype=torch.float32)

combos = [
    'uw05_lmbd005_eh5_h5_z10',
    'uw05_lmbd01_eh5_h5_z10',
    'uw05_lmbd02_eh5_h5_z10',
    'uw00_lmbd02_eh5_h5_z10',
    'uw01_lmbd02_eh5_h5_z10',
]

for combo in combos:
    run_base = 'runs_thalmann_hp_v2_%s/fold0' % combo
    if not os.path.exists(run_base):
        print('%s  MISSING' % combo); continue
    sd0 = sorted(d for d in os.listdir(run_base) if d.startswith('seed_'))[0]
    cfg_p = os.path.join(run_base, sd0, 'config.json')
    if not os.path.exists(cfg_p):
        print('%s  no config' % combo); continue
    with open(cfg_p) as f: cfg = json.load(f)
    mc = cfg['model_config']
    ckpt = os.path.join(run_base, sd0, 'checkpoints', 'epoch%04d.pt' % cfg['cv_selected_epoch'])
    if not os.path.exists(ckpt):
        print('%s  no ckpt' % combo); continue
    enc = IDRNN(in_dim=mc['enc_in_dim'], z_dim=mc['z_dim'], hid=mc['enc_hidden'],
                n_tasks=mc['n_tasks'], task_emb_dim=mc['task_emb_dim'])
    enc.load_state_dict({k[8:]: v for k, v in torch.load(ckpt, map_location='cpu').items()
                         if k.startswith('encoder.')})
    enc.eval(); enc.set_task_ids(task_ids_global)
    with torch.no_grad():
        mu, logvar = enc(xin_t)
    z = mu[:, 0, T_TASK0_LAST, :].numpy()
    sig = np.exp(0.5 * logvar[:, 0, T_TASK0_LAST, :].numpy())
    idx_l, idx_h = int(np.argmin(phq_v)), int(np.argmax(phq_v))
    snr = float((z.std(0) / (sig.mean(0) + 1e-8)).mean())
    max_diff = float(np.abs((z[idx_h] - z[idx_l]) / (z.std(0) + 1e-8)).max())
    print('%s  z_std=%.4f  sigma=%.4f  SNR=%.3f  max_diff/std=%.3f  lmbd=%.3f  uw=%.2f' % (
        combo, z.std(0).mean(), sig.mean(), snr, max_diff,
        cfg['lmbd'], cfg['unif_weight']))
