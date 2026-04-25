"""
IAPW-D Experiment Suite v3 — NeurIPS 2026
4 high-impact figures, K=16 APs, M=50 devices, T=10000 rounds.
Baselines: PW, Optimistic MW, FLL, IAPW-D fixed, IAPW-D adaptive.
Includes Markov-modulated (bursty) congestion scenario.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os, time

rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'legend.fontsize': 7, 'figure.dpi': 300, 'savefig.bbox': 'tight',
    'text.usetex': False})
FIG_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Helpers ──
def latency_vec(rho, alphas, betas):
    return alphas * rho + betas

def generate_markov_alphas(T, K, alpha_base, factor=3.0,
                           p_cong=0.02, p_norm=0.05, rng=None):
    """2-state Markov chain per AP: normal/congested."""
    if rng is None:
        rng = np.random.RandomState()
    states = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        r = rng.random(K)
        go_cong = (states[t-1] == 0) & (r < p_cong)
        go_norm = (states[t-1] == 1) & (r < p_norm)
        states[t] = states[t-1].copy()
        states[t][go_cong] = 1
        states[t][go_norm] = 0
    return np.where(states == 0, alpha_base[None, :], alpha_base[None, :] * factor)

# ═══════════════════════════════════════════════════════════════
#  Core: Vectorized Multi-Device Game
# ═══════════════════════════════════════════════════════════════
def run_game(M, K, T, alphas_t, betas, W_p, sigma0, c, tau,
             lam, eta, algo='iapwd', adaptive_lambda=False):
    """
    Run multi-device game. W_p is (M, K) priority matrix.
    alphas_t: (T, K) time-varying alpha values.
    algo: 'iapwd' | 'pw' | 'omw' | 'fll'
    """
    weights = np.ones((M, K))
    a_cur = np.random.randint(0, K, M)
    deltas = np.zeros(M, dtype=int)
    prev_L = np.zeros((M, K))
    arange_M = np.arange(M)

    cum_surr = np.zeros(M)
    cum_phys = np.zeros(M)
    cum_sw = np.zeros(M, dtype=int)
    cum_sc = np.zeros(M)
    cum_gD = np.zeros(M)
    cum_ap = np.zeros((M, K))  # per-AP cumulative loss

    hist_surr = np.zeros((M, T))
    hist_phys = np.zeros((M, T))
    hist_sw = np.zeros((M, T), dtype=int)
    hist_sc = np.zeros((M, T))
    hist_gD = np.zeros((M, T))

    # FLL state
    if algo == 'fll':
        fll_cum = np.zeros((M, K))

    for t in range(T):
        al = alphas_t[t]

        if algo == 'fll':
            # Compute load from deterministic assignments
            rho = np.bincount(a_cur, minlength=K).astype(float)
            bl = latency_vec(rho, al, betas)
            L = bl[None, :] / W_p
            fll_cum += L
            cum_ap += L
            # Decide switches
            threshold = sigma0 + c * tau
            best_ap = np.argmin(fll_cum, axis=1)
            gain = fll_cum[arange_M, a_cur] - fll_cum[arange_M, best_ap]
            do_switch = (best_ap != a_cur) & (gain > threshold)
            sig_t = sigma0 + c * np.minimum(deltas, tau).astype(float)
            cum_sw += do_switch.astype(int)
            cum_sc += do_switch * sig_t
            deltas = np.where(do_switch, 0, deltas + 1)
            a_cur = np.where(do_switch, best_ap, a_cur)
            cur_loss = L[arange_M, a_cur]
            cum_surr += cur_loss
            cum_phys += cur_loss
        else:
            # Compute base distribution
            if algo == 'omw' and t > 0:
                w_opt = weights * np.maximum(1 - eta * prev_L, 1e-15)
                Wd = w_opt.sum(axis=1, keepdims=True)
                P = w_opt / Wd
            else:
                Wd = weights.sum(axis=1, keepdims=True)
                P = weights / Wd

            rho = P.sum(axis=0)
            bl = latency_vec(rho, al, betas)
            L = bl[None, :] / W_p
            cum_ap += L
            F = (P * L).sum(axis=1)

            if algo in ('pw', 'omw'):
                P_sample = P
                cum_surr += F
                cum_phys += F
            else:  # iapwd
                phi = P[arange_M, a_cur]
                L_cur = L[arange_M, a_cur]
                sig_t = sigma0 + c * np.minimum(deltas, tau).astype(float)
                lt = lam / np.sqrt(t + 1) if adaptive_lambda else lam
                gt = lt * sig_t
                clip = (L_cur <= F)
                ge = gt * clip
                den = 1 + ge * phi
                Pt = P / den[:, None]
                Pt[arange_M, a_cur] = (1 + ge) * phi / den
                Ft = (Pt * L).sum(axis=1)
                cum_gD += (Ft - F)
                cum_surr += F
                rho_p = Pt.sum(axis=0)
                pl = latency_vec(rho_p, al, betas)
                Lp = pl[None, :] / W_p
                cum_phys += (Pt * Lp).sum(axis=1)
                P_sample = Pt

            # Sample
            cp = P_sample.cumsum(axis=1)
            cp[:, -1] = 1.0  # numerical safety
            u = np.random.uniform(0, 1, M)
            a_new = (cp < u[:, None]).sum(axis=1)
            a_new = np.minimum(a_new, K - 1)

            switched = (a_new != a_cur)
            cum_sw += switched.astype(int)
            if algo == 'iapwd':
                sig_now = sigma0 + c * np.minimum(deltas, tau).astype(float)
                cum_sc += switched * sig_now
            deltas = np.where(switched, 0, deltas + 1)
            a_cur = a_new
            prev_L = L.copy()
            weights *= np.maximum(1 - eta * L, 1e-15)

        hist_surr[:, t] = cum_surr
        hist_phys[:, t] = cum_phys
        hist_sw[:, t] = cum_sw
        hist_sc[:, t] = cum_sc
        hist_gD[:, t] = cum_gD

    return {'surrogate_loss': hist_surr, 'physical_loss': hist_phys,
            'switches': hist_sw, 'switch_cost': hist_sc, 'gamma_D': hist_gD,
            'best_fixed': cum_ap.min(axis=1)}


# ═══════════════════════════════════════════════════════════════
#  FIGURE 1: Regret + Switch Comparison (Markov-modulated)
# ═══════════════════════════════════════════════════════════════
def fig1_regret_switches():
    print("Figure 1: Regret + Switch Comparison (Markov-modulated)...")
    t0 = time.time()
    T, K, M, N = 10000, 16, 50, 40
    sigma0, c, tau = 0.05, 0.002, 100
    eta = min(np.sqrt(np.log(K) / T), 0.5)
    alpha_base = np.linspace(0.03, 0.12, K)
    betas = np.linspace(0.01, 0.04, K)
    W_p = np.ones((M, K))

    configs = [
        ('IAPW-D (fixed $\lambda\!=\!20$)', 'iapwd', 20, False),
        ('IAPW-D (adaptive $\lambda_0\!=\!20$)', 'iapwd', 20, True),
        ('PW (no inertia)', 'pw', 0, False),
        ('Optimistic MW', 'omw', 0, False),
        ('Follow-the-Lazy-Leader', 'fll', 0, False),
    ]
    colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']
    styles = ['-', '--', '-', '-.', ':']

    all_reg = {n: [] for n, *_ in configs}
    all_sw = {n: [] for n, *_ in configs}
    ts_arr = np.arange(1, T + 1)

    for trial in range(N):
        rng = np.random.RandomState(trial * 7)
        alphas_t = generate_markov_alphas(T, K, alpha_base, factor=3.0,
                                          p_cong=0.02, p_norm=0.05, rng=rng)
        for name, algo, lam, adapt in configs:
            np.random.seed(trial * 1000 + hash(name) % 500)
            h = run_game(M, K, T, alphas_t, betas, W_p, sigma0, c, tau,
                         lam, eta, algo=algo, adaptive_lambda=adapt)
            
            bf = h['best_fixed'].mean()
            avg_surr = h['surrogate_loss'].mean(axis=0)
            regret_curve = avg_surr - (bf / T) * ts_arr
            avg_sw = h['switches'].mean(axis=0).astype(float)
            all_reg[name].append(regret_curve)
            all_sw[name].append(avg_sw)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ts = ts_arr

    for i, (name, *_) in enumerate(configs):
        arr = np.array(all_reg[name])
        mu, sd = arr.mean(0), arr.std(0)
        ax1.plot(ts, mu, color=colors[i], ls=styles[i], label=name, lw=1.2)
        ax1.fill_between(ts, mu - sd, mu + sd, color=colors[i], alpha=0.1)

    ax1.set_xlabel('Round $t$'); ax1.set_ylabel('Avg. Regret')
    ax1.set_title(f'(a) Surrogate Regret (Markov, $K$={K}, $M$={M})')
    ax1.legend(fontsize=6); ax1.grid(True, alpha=0.3)

    for i, (name, *_) in enumerate(configs):
        arr = np.array(all_sw[name])
        mu = arr.mean(0)
        ax2.plot(ts, mu, color=colors[i], ls=styles[i], label=name, lw=1.2)
    ax2.set_xlabel('Round $t$'); ax2.set_ylabel('Avg. Cumulative Switches')
    ax2.set_title('(b) Switch Count')
    ax2.legend(fontsize=6); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_regret_switches.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig1_regret_switches.png'))
    plt.close()
    for name, *_ in configs:
        sw_final = np.array(all_sw[name])[:, -1].mean()
        print(f"    {name}: switches={sw_final:.0f}")
    print(f"  Saved fig1 ({time.time()-t0:.0f}s)")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 2: Physical vs Surrogate Regret + Gap
# ═══════════════════════════════════════════════════════════════
def fig2_physical_regret():
    print("Figure 2: Physical vs Surrogate Regret...")
    t0 = time.time()
    T, K, M, N = 10000, 16, 50, 25
    sigma0, c, tau = 0.05, 0.002, 100
    eta = min(np.sqrt(np.log(K) / T), 0.5)
    alphas = np.linspace(0.03, 0.12, K)
    betas = np.linspace(0.01, 0.04, K)
    alphas_t = np.tile(alphas, (T, 1))
    W_p = np.ones((M, K))

    configs = [
        ('Adaptive $\lambda_0\!=\!5$', 5, True),
        ('Fixed $\lambda\!=\!5$', 5, False),
        ('Fixed $\lambda\!=\!50$', 50, False),
    ]
    colors = ['#4CAF50', '#2196F3', '#FF5722']

    res = {n: {'surr': np.zeros(T), 'phys': np.zeros(T)} for n, *_ in configs}

    for trial in range(N):
        for name, lam, adapt in configs:
            np.random.seed(trial * 200 + hash(name) % 200)
            h = run_game(M, K, T, alphas_t, betas, W_p, sigma0, c, tau,
                         lam, eta, algo='iapwd', adaptive_lambda=adapt)
            res[name]['surr'] += h['surrogate_loss'].mean(axis=0) / N
            res[name]['phys'] += h['physical_loss'].mean(axis=0) / N

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ts = np.arange(1, T + 1)

    for i, (name, *_) in enumerate(configs):
        r = res[name]
        ax1.plot(ts, r['surr'], color=colors[i], ls='--', label=f'{name} (surr)', lw=1.2)
        ax1.plot(ts, r['phys'], color=colors[i], ls='-', label=f'{name} (phys)', lw=1.0, alpha=0.8)
    ax1.set_xlabel('Round $t$'); ax1.set_ylabel('Avg. Cumulative Loss')
    ax1.set_title(f'(a) Surrogate vs Physical Loss ($K$={K}, $M$={M})')
    ax1.legend(fontsize=5.5); ax1.grid(True, alpha=0.3)

    for i, (name, *_) in enumerate(configs):
        gap = res[name]['phys'] - res[name]['surr']
        ax2.plot(ts, gap, color=colors[i], label=name, lw=1.2)
    ax2.plot(ts, 0.3 * np.sqrt(ts), 'k--', label=r'$O(\sqrt{T})$', lw=0.8, alpha=0.5)
    ax2.set_xlabel('Round $t$'); ax2.set_ylabel('Physical $-$ Surrogate Gap')
    ax2.set_title('(b) Surrogate-to-Physical Gap')
    ax2.legend(fontsize=6); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_physical_regret.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig2_physical_regret.png'))
    plt.close()
    print(f"  Saved fig2 ({time.time()-t0:.0f}s)")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 3: Theory Verification (Γ_D, switch cost, ε_dyn rate)
# ═══════════════════════════════════════════════════════════════
def fig3_theory_verification():
    print("Figure 3: Theory Verification...")
    t0 = time.time()
    K, M = 16, 50
    sigma0, c, tau = 0.05, 0.002, 100
    sigma_max = sigma0 + c * tau
    alphas = np.linspace(0.03, 0.12, K)
    betas = np.linspace(0.01, 0.04, K)
    W_p = np.ones((M, K))
    lam = 5.0
    N_hist = 80

    T_v = 5000
    eta_v = min(np.sqrt(np.log(K) / T_v), 0.5)
    alphas_t = np.tile(alphas, (T_v, 1))
    all_gD, all_sc = [], []

    for trial in range(N_hist):
        np.random.seed(trial + 7000)
        h = run_game(M, K, T_v, alphas_t, betas, W_p, sigma0, c, tau,
                     lam, eta_v, algo='iapwd')
        all_gD.extend(h['gamma_D'][:, -1].tolist())
        for d in range(M):
            all_sc.append(h['switch_cost'][d, -1] / T_v)

    switch_bound = sigma_max * (K - 1) / (K + lam * sigma_max)

    Ts = [500, 1000, 2000, 5000, 10000, 25000, 50000]
    emp_eps, theory_eps = [], []
    N_conv = 20

    for Tc in Ts:
        eta_c = min(np.sqrt(np.log(K) / Tc), 0.5)
        lam_star = max((K - 1) / 2 * np.sqrt(Tc / np.log(K)) - K / sigma_max, 0.1)
        alphas_tc = np.tile(alphas, (Tc, 1))
        eps_t = []
        for trial in range(N_conv):
            np.random.seed(trial + 9000)
            h = run_game(M, K, Tc, alphas_tc, betas, W_p, sigma0, c, tau,
                         lam_star, eta_c, algo='iapwd')
            for d in range(M):
                avg_l = h['surrogate_loss'][d, -1] / Tc
                avg_sc = h['switch_cost'][d, -1] / Tc
                best_avg = h['best_fixed'][d] / Tc
                eps_t.append(max(avg_l + avg_sc - best_avg, 1e-10))
        emp_eps.append(np.mean(eps_t))
        theory_eps.append(4 * np.sqrt(np.log(K) / Tc))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    axes[0].hist(all_gD, bins=50, color='#2196F3', alpha=0.8, edgecolor='white')
    axes[0].axvline(0, color='red', ls='--', lw=1.5, label='$\\Gamma_D=0$')
    axes[0].set_xlabel('$\\Gamma_D(T)$'); axes[0].set_ylabel('Count')
    axes[0].set_title(f'(a) Stickiness Overhead ($K$={K}, $M$={M})')
    axes[0].legend()

    axes[1].hist(all_sc, bins=50, color='#4CAF50', alpha=0.8, edgecolor='white')
    axes[1].axvline(switch_bound, color='red', ls='--', lw=1.5,
                    label=f'Bound={switch_bound:.4f}')
    axes[1].set_xlabel('Avg. switching cost/round')
    axes[1].set_title('(b) Switching Cost')
    axes[1].legend()

    axes[2].loglog(Ts, emp_eps, 'o-', color='#2196F3', label='Empirical', ms=6)
    axes[2].loglog(Ts, theory_eps, 's--', color='#FF5722',
                   label='$4\\sqrt{\\ln K/T}$', ms=5)
    log_T = np.log(Ts); log_e = np.log(emp_eps)
    slope = np.polyfit(log_T, log_e, 1)[0]
    axes[2].set_xlabel('Horizon $T$')
    axes[2].set_ylabel('$\\varepsilon_{\\mathrm{dyn}}$')
    axes[2].set_title(f'(c) Convergence (slope={slope:.2f})')
    axes[2].legend(); axes[2].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_theory_verification.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig3_theory_verification.png'))
    plt.close()
    pct = sum(g <= 1e-10 for g in all_gD)
    print(f"  Γ_D ≤ 0: {pct}/{len(all_gD)}")
    print(f"  Mean switch cost: {np.mean(all_sc):.5f} (bound: {switch_bound:.5f})")
    print(f"  Conv slope: {slope:.2f}")
    print(f"  Saved fig3 ({time.time()-t0:.0f}s)")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 4: Heterogeneous Weights + Pareto Frontier
# ═══════════════════════════════════════════════════════════════
def fig4_hetero_pareto():
    print("Figure 4: Heterogeneous Weights + Pareto Frontier...")
    t0 = time.time()
    T, K, M = 5000, 16, 50
    sigma0, c, tau = 0.05, 0.002, 100
    eta = min(np.sqrt(np.log(K) / T), 0.5)
    alphas = np.linspace(0.03, 0.12, K)
    betas = np.linspace(0.01, 0.04, K)
    alphas_t = np.tile(alphas, (T, 1))
    N = 30

    categories = ['Voice']*10 + ['Video']*10 + ['BestEffort']*15 + ['Background']*15
    cat_base = {'Voice': 7, 'Video': 4, 'BestEffort': 2, 'Background': 1}
    avg_by_cat = {c: [] for c in ['Voice', 'Video', 'BestEffort', 'Background']}

    for trial in range(N):
        np.random.seed(trial + 3000)
        W_hetero = np.ones((M, K))
        rng_w = np.random.RandomState(trial + 6000)
        for d in range(M):
            base = cat_base[categories[d]]
            W_hetero[d, :] = base + rng_w.uniform(-0.3 * base, 0.3 * base, K)
            W_hetero[d, :] = np.maximum(W_hetero[d, :], 0.5)

        h = run_game(M, K, T, alphas_t, betas, W_hetero, sigma0, c, tau,
                     5.0, eta, algo='iapwd', adaptive_lambda=True)
        for d in range(M):
            avg_by_cat[categories[d]].append(h['surrogate_loss'][d, -1] / T)

    W_uni = np.ones((M, K))
    lambdas = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
    mr, msc = [], []

    for lam_v in lambdas:
        rt, st = [], []
        for trial in range(N):
            np.random.seed(trial + 4000)
            h = run_game(M, K, T, alphas_t, betas, W_uni, sigma0, c, tau,
                         lam_v, eta, algo='iapwd')
            bf = h['best_fixed'].mean()
            reg = h['surrogate_loss'].mean(axis=0)[-1] - bf
            rt.append(reg)
            st.append(h['switch_cost'].mean(axis=0)[-1])
        mr.append(np.mean(rt)); msc.append(np.mean(st))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    cat_names = ['Voice\n($w$=7)', 'Video\n($w$=4)', 'BestEff.\n($w$=2)', 'Backgr.\n($w$=1)']
    cat_data = [avg_by_cat['Voice'], avg_by_cat['Video'],
                avg_by_cat['BestEffort'], avg_by_cat['Background']]
    bp = ax1.boxplot(cat_data, tick_labels=cat_names, patch_artist=True,
                     medianprops=dict(color='#D32F2F', linewidth=2))
    box_colors = ['#1565C0', '#2196F3', '#64B5F6', '#BBDEFB']
    for patch, col in zip(bp['boxes'], box_colors):
        patch.set_facecolor(col)
    ax1.set_ylabel('Avg. Loss per Round')
    ax1.set_title(f'(a) Loss by Priority Class ($K$={K}, $M$={M})')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.plot(msc, mr, 'o-', color='#2196F3', markersize=6, lw=1.2)
    for i, lv in enumerate(lambdas):
        ax2.annotate(f'$\\lambda$={lv}', (msc[i], mr[i]),
                     fontsize=6, ha='left', va='bottom',
                     xytext=(3, 3), textcoords='offset points')
    ax2.set_xlabel('Total Switching Cost')
    ax2.set_ylabel('Surrogate Regret')
    ax2.set_title(f'(b) Regret--Switching Cost Pareto ($K$={K})')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_hetero_pareto.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig4_hetero_pareto.png'))
    plt.close()
    for cat in ['Voice', 'Video', 'BestEffort', 'Background']:
        print(f"    {cat}: mean={np.mean(avg_by_cat[cat]):.4f} std={np.std(avg_by_cat[cat]):.4f}")
    print(f"  Saved fig4 ({time.time()-t0:.0f}s)")


# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("IAPW-D NeurIPS Experiment Suite v3")
    print(f"K=16 APs, M=50 devices")
    print("=" * 60)
    fig1_regret_switches()
    fig2_physical_regret()
    fig3_theory_verification()
    fig4_hetero_pareto()
    print("=" * 60)
    print("All figures saved to:", FIG_DIR)
