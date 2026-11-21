"""Unvectorised baseline: lifelib's CashValue_SE (the same model, one policy at a time)."""
import sys, os, time, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..'))
import modelx as mx, model as M
N = 300
m = M.load_original(); mp, scen = M.portfolio_sample(m, N, 1); Z = M.normals(1)
P = M.configure(m, mp, scen, Z); t = time.perf_counter(); pv_me = P.result_pv(); t_me = time.perf_counter() - t
se = mx.read_model(os.path.join(os.path.dirname(M.source_path()), 'CashValue_SE')); S = se.Projection
tab = m.Projection.model_point_10000.iloc[:N].copy(); tab['accum_prem_init_pp'] = 0
S.model_point_table = tab; S.std_norm_rand = Z; S.scen_id = 0
ids = list(tab.index); t = time.perf_counter(); rows = [S[i].result_pv() for i in ids]; t_se = time.perf_counter() - t
cols = [c for c in rows[0].columns if c in pv_me.columns]
a = np.array([r[cols].to_numpy()[0] for r in rows]); b = pv_me[cols].to_numpy()
months = int(sum(S[i].proj_len() for i in ids))
rel = np.abs(a - b) / np.maximum(np.abs(b), 1.0)
out = {'policies': N, 'policy_months': months, 'seconds_one_at_a_time': t_se, 'seconds_vectorised': t_me,
       'columns_compared': cols, 'identical_share': float((a == b).mean()), 'max_rel_diff': float(rel.max())}
print(out); json.dump(out, open(os.path.join(HERE, 'results', 'baseline.json'), 'w'), indent=1)
