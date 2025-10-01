import numpy as np
rng = np.random.default_rng(42)

# ---------------------------
# 1) Problem definitions
# ---------------------------
# Task A: f1(x) = -(x-2)(x-6),  0 < x < 9  (maximize)
# Task B: f2(x) = -(x-200)(x-600),  0 < x < 900  (maximize)

def f1(x):  # scalar
    return - (x - 2.0) * (x - 6.0)

def f2(x):  # scalar
    return - (x - 200.0) * (x - 600.0)

bounds = np.array([[0.0, 9.0],
                   [0.0, 900.0]], dtype=float)  # per-dimension bounds [min,max]
n_tasks = 2  # we treat each dimension as one task

# ---------------------------
# 2) Helper: evaluation, ranks, scalar fitness (MFO)
# ---------------------------
def evaluate_population(P):
    """
    P: (N, 2) array; P[:,0] is candidate for Task A, P[:,1] for Task B.
    Returns:
      F: (N, 2) objective values (higher is better)
      ranks: (N, 2) rank per task (1 = best)
      scalar_fit: (N,) scalar fitness = 1/min_rank
      skill: (N,) skill factor in {0,1}
    """
    N = P.shape[0]
    F = np.empty((N, 2), dtype=float)
    F[:, 0] = f1(P[:, 0])
    F[:, 1] = f2(P[:, 1])

    # ranks per task (1 is best). argsort returns ascending; we want descending by F
    ranks = np.empty_like(F, dtype=int)
    for k in range(2):
        order = np.argsort(-F[:, k])  # indices sorted by decreasing fitness
        ranks[order, k] = np.arange(1, N + 1)

    best_rank = np.min(ranks, axis=1)
    scalar_fit = 1.0 / best_rank
    skill = np.argmin(ranks, axis=1)  # task where the candidate ranks best
    return F, ranks, scalar_fit, skill

# ---------------------------
# 3) CCC-style operators (NO GA/PSO)
# ---------------------------
def clip_to_bounds(P):
    # per-dimension clipping
    for d in range(P.shape[1]):
        P[:, d] = np.clip(P[:, d], bounds[d, 0], bounds[d, 1])
    return P

def cooperation_step(P, skill, p_xfer=0.35, beta=0.5, omega=None):
    """
    With probability p_xfer, each individual borrows a step toward a helper,
    possibly from the *other* task (inter-task transfer). Dimension chosen at random.
    """
    N, D = P.shape
    for i in range(N):
        if rng.random() < p_xfer:
            # choose a helper uniformly from population (can be same or other task)
            h = rng.integers(0, N)
            # choose which dimension to transfer on (0 or 1)
            d = rng.integers(0, D)
            step = beta * (omega[i] if omega is not None else 1.0) * (P[h, d] - P[i, d])
            P[i, d] += step
    return P

def elite_discussion(P, F, take_top=0.2):
    """
    Pair top individuals and apply 'take-the-better' per dimension.
    Robust to odd-sized elite sets.
    """
    N = P.shape[0]
    if N < 2:
        return P

    # Score for choosing elites (normalized sum across tasks)
    denom0 = np.abs(F[:, 0]).max() + 1e-12
    denom1 = np.abs(F[:, 1]).max() + 1e-12
    scores = (F[:, 0] / denom0) + (F[:, 1] / denom1)

    # Pick top-k; ensure k is EVEN and >= 2
    n_top = max(2, int(np.ceil(take_top * N)))
    if n_top % 2 == 1:  # make it even
        n_top -= 1
    if n_top < 2:
        return P

    elite_idx = np.argsort(-scores)[:n_top]
    rng.shuffle(elite_idx)

    # Pair sequentially
    for j in range(0, n_top, 2):
        a, b = elite_idx[j], elite_idx[j + 1]

        # Dimension-wise keep the better coordinate by its own task’s objective
        if f1(P[a, 0]) < f1(P[b, 0]):
            P[a, 0] = P[b, 0]
        else:
            P[b, 0] = P[a, 0]

        if f2(P[a, 1]) < f2(P[b, 1]):
            P[a, 1] = P[b, 1]
        else:
            P[b, 1] = P[a, 1]

    return P

def competition_and_reseed(P, F, reseed_frac=0.1, crowd_eps=(1.0, 20.0)):
    """
    Light diversity maintenance:
    - If two individuals are too close in a dimension, nudge the *better* one slightly backward.
    - Reseed worst few uniformly at random within bounds.
    """
    N, D = P.shape
    # Crowding nudges
    pairs = rng.integers(0, N, size=(N, 2))
    for a, b in pairs:
        if a == b:
            continue
        # closeness thresholds per dimension
        close0 = abs(P[a, 0] - P[b, 0]) < crowd_eps[0]
        close1 = abs(P[a, 1] - P[b, 1]) < crowd_eps[1]
        if close0 or close1:
            # Better by combined score
            sa = f1(P[a, 0]) + f2(P[a, 1])
            sb = f1(P[b, 0]) + f2(P[b, 1])
            better = a if sa >= sb else b
            # small backward nudge (noise)
            noise = rng.normal(0.0, [0.3, 30.0], size=2)
            P[better] -= 0.05 * noise

    # Reseed worst few by total score
    scores = f1(P[:, 0]) + f2(P[:, 1])
    n_bad = max(1, int(reseed_frac * N))
    worst = np.argsort(scores)[:n_bad]
    for i in worst:
        for d in range(D):
            P[i, d] = rng.uniform(bounds[d, 0], bounds[d, 1])
    return P

# ---------------------------
# 4) MF-CCCA main loop
# ---------------------------
def mf_ccca(pop_size=60, iters=200, rmp=0.3, alpha=0.7, beta=0.5, print_every=1):
    """
    Multifactorial CCC for two tasks (1D + 1D).
    Returns ((xA, fA), (xB, fB), history)
    - print_every: int or None/0. If 1 prints every iteration; if k prints every k iters.
    """
    # Initialize population uniformly within each dimension's bounds
    P = np.column_stack([
        rng.uniform(bounds[0, 0], bounds[0, 1], size=pop_size),
        rng.uniform(bounds[1, 0], bounds[1, 1], size=pop_size),
    ])
    omega = rng.uniform(0.1, 0.5, size=pop_size)  # learning ability, will adapt slightly

    best_hist = {"A": [], "B": [], "fA": [], "fB": []}

    # Optional header
    if print_every and print_every > 0:
        print(f"{'Iter':>5} | {'xA_best':>10} {'f1_best':>12} || {'xB_best':>10} {'f2_best':>12}")

    for t in range(1, iters + 1):
        P = clip_to_bounds(P)
        F, ranks, scalar_fit, skill = evaluate_population(P)

        # Best per task (values, not indices)
        bestA_idx = np.argmax(F[:, 0])
        bestB_idx = np.argmax(F[:, 1])
        bestA_x = P[bestA_idx, 0]
        bestB_x = P[bestB_idx, 1]
        bestA_f = F[bestA_idx, 0]
        bestB_f = F[bestB_idx, 1]

        # record history
        best_hist["A"].append(bestA_x)
        best_hist["B"].append(bestB_x)
        best_hist["fA"].append(bestA_f)
        best_hist["fB"].append(bestB_f)

        # progress print
        if print_every and print_every > 0 and (t % print_every == 0 or t == 1 or t == iters):
            print(f"{t:5d} | {bestA_x:10.6f} {bestA_f:12.6f} || {bestB_x:10.6f} {bestB_f:12.6f}")

        # --- CCCA Self-study (toward task-best in skilled dimension)
        for i in range(pop_size):
            d = skill[i]  # 0 for Task A, 1 for Task B
            best_val = bestA_x if d == 0 else bestB_x
            step = alpha * omega[i] * (best_val - P[i, d])
            P[i, d] += step

            # mild exploration on the non-skilled dimension (tiny step toward that task's best)
            other = 1 - d
            other_best = bestB_x if other == 1 else bestA_x
            P[i, other] += 0.15 * alpha * omega[i] * (other_best - P[i, other])

        # --- Cooperation (inter-/intra-task assistance)
        P = cooperation_step(P, skill, p_xfer=rmp, beta=beta, omega=omega)

        # --- Elite discussion (dimension-wise keep-better among elites)
        P = elite_discussion(P, F, take_top=0.25)

        # --- Competition & reseed for diversity
        P = competition_and_reseed(P, F, reseed_frac=0.08, crowd_eps=(0.4, 15.0))

        # --- Adaptive learning ability (increase slowly, capped)
        omega = np.minimum(1.0, omega + 0.01)

    # Final evaluation
    P = clip_to_bounds(P)
    F, ranks, scalar_fit, skill = evaluate_population(P)
    bestA_idx = np.argmax(F[:, 0]); bestB_idx = np.argmax(F[:, 1])
    solA = P[bestA_idx, 0]; valA = F[bestA_idx, 0]
    solB = P[bestB_idx, 1]; valB = F[bestB_idx, 1]

    return (solA, valA), (solB, valB), best_hist

if __name__ == "__main__":
    # Change print_every to:
    #   1  -> print every iteration
    #   10 -> print every 10 iterations
    #   0/None -> no per-iteration prints
    (xA, fA), (xB, fB), hist = mf_ccca(
        pop_size=60,
        iters=200,
        rmp=0.35,
        alpha=0.7,
        beta=0.5,
        print_every=10
    )

    print("\nFinal Results")
    print("Task A (maximize -(x-2)(x-6), 0<x<9)")
    print(f"  Best x ≈ {xA:.6f},  f(x) ≈ {fA:.6f}   (analytic optimum at x=4, f=4)")
    print("Task B (maximize -(x-200)(x-600), 0<x<900)")
    print(f"  Best x ≈ {xB:.6f},  f(x) ≈ {fB:.6f}   (analytic optimum at x=400, f=40000)")
