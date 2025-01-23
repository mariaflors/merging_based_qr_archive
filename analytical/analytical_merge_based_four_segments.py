"""
Script to compute the average waiting time of the entanglement-based repeater over four segments without a patching
limitation.

"""
# Derivation
# ----------
#
# First, we define two auxiliary functions:


def A(p, pg):
    """The average time it takes to produce two Bell pairs in parallel,
    followed by fusing them into a 3-qubit cluster state.

    Parameters
    ----------
    p : fusion success probability
    pg : elementary-link generation success probability
    """
    maximum_of_two_iid_geometric_random_vars = (3 - 2 * pg) / (pg * (2 - pg))
    return maximum_of_two_iid_geometric_random_vars * (1 / p)


def B(p, pg):
    """
    The average time it takes to produce 3-qubit cluster states in parallel.
    Found as eq. (64) in https://arxiv.org/pdf/1710.06214, but with a factor p_fuse (called `a` in that article) omitted
    in the denominator because eq. (64) also includes restarting 1/p_fuse times on average until the
    fuse/entanglement-swap operation is successful.

    Parameters
    ----------
    p : float
        Fusion success probability.
    pg : float
        Success probability of generating an elementary link.
    """
    numerator_1 = 2 * (p * p) * (pg ** 4) * (pg - 1) * (2 * pg - 3)
    numerator_2 = -1 * p * (20 * (pg ** 5) - 72 * (pg ** 4) + 93 * (pg ** 3) - 53 * (pg ** 2) + 10 * pg + 4)
    numerator_3 = 3 * ((3 - 2 * pg) ** 2 * (2 * pg * pg - 3 * pg + 2))
    numerator = numerator_1 + numerator_2 + numerator_3
    denominator_1 = p * pg  # omitting a factor p_fuse here
    denominator_2 = 2 - pg
    denominator_3 = p * (pg * pg) - (p + 2) * pg + 3
    denominator_4 = -1 * p * (pg ** 3) + 4 * (pg * pg) - 6 * pg + 4
    denominator = denominator_1 * denominator_2 * denominator_3 * denominator_4
    return numerator / denominator


# In the continuation of our derivation, we will assume that the fuse operation is instantaneous.
# 
# Denote by X our final goal: the average time it takes to generate a cluster state over four segments, starting from
# the situation where no entanglement at all is present. Then note that
# 
# X = B + (1 - p) * Z
#
# where B is the function defined above and Z is the time it takes to produce a cluster state from the
# "hole-in-the-middle situation", i.e., a Bell pair exists on both segments 1 and 4 but no entanglement over segments
# 2 and 3. The reason for this is, that if the fuse operation fails (which happens with probability p), then we end up
# with a hole in the middle.
#
# Continuing, we find that
#
# Z = A + 2 * p * (1 - p) * Y + (1 - p)^2 * X
#
# where A is the function defined above and Y is the time it takes to produce a cluster state, starting from the
# "hole-on-the-side situation", i.e., there is only a 3-qubit cluster state over segments 1 and 2 and no entanglement
# over 3 and 4 (by symmetry, the situation of entanglement-over-3-and-4 and no-entanglement-over-1-and-2 has the same
# waiting time).
# The derivation is similar: A is the time it takes to produce the patch, which then glues to only one side with
# probability 2 * p * (1 - p)  (the factor 2 is there because the hole-on-the-left has the same waiting time as the
# hole-on-the-right) and with probability (1-p)^2 both fuses fail and all entanglement is gone.
#
# Finally,
#
# Y = A + (1 - p) * Z
#
# Rephrased, the linear system of equations in the three unknowns X, Y, Z is now:
# 
# (1)  1 * X + 0 * Y + (p - 1) * Z = B
# (2)  0 * X + 1 * Y + (p - 1) * Z = A
# (3)  (1 - p)^2 * X + 2p(1-p) * Y - 1 * Z = -A
# 
# Solving is done in the following steps:
#
# first, eq. (1) multiplied by (1-p)^2 gives:
#
# (1')  (1-p)^2 * X + (1-p)^2 * (p-1) * Z = (1-p)^2 * B
# 
# next, eq. (2) multiplied by 2p(1-p) yields
#
# (2')  2p(1-p) * Y - 2p * (1-p)^2 * Z = 2p(1-p) * A
#
# Subtracting (1') from (3) yields
#
# (3') = (3) - (1'):  2p(1-p) * Y + (-1 + (1-p)^2 * (1-p)) * Z = -A - (1-p)^2 * B
#
# Next, subtracting (2') from (3') yields
#
# (3'') = (3') - (2'):  (-1 + (1-p)^3 + 2p * (1-p)^2) * Z = (-1 - 2p(1-p)) * A - (1 - p)^2 * B
#
# so we arrive at:

def Z(p, pg):
    numerator = (-1 - 2 * p * (1 - p)) * A(p=p, pg=pg) - (1 - p) * (1 - p) * B(p=p, pg=pg)
    denominator = -1 + (1 - p) * (1 - p) * (1 - p) + 2 * p * (1 - p) * (1 - p)
    return numerator / denominator


# Now we use eq (1) to find:

def X(p, pg):
    return B(p=p, pg=pg) + (1 - p) * Z(p=p, pg=pg)


if __name__ == "__main__":
    # Run this code to get a plot of the verification of our Monte Carlo sampling
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    from monte_carlo.merge_based import wt
    # Numeric and the analytic results for some values of p_gen and p to verify our Monte Carlo simulation
    sys.setrecursionlimit(100000)
    num_of_samples = 1000
    pgs = np.linspace(0.1, 0.9)
    ps = [0.3, 0.5, 0.7, 0.9]
    colors = ["blue", "orange", "red", "green"]
    numerics = [[] for p in ps]
    d_numerics = [[] for p in ps]
    analytics = [[] for p in ps]
    for i, p in enumerate(ps):
        for pg in pgs:
            data = []
            for j in range(num_of_samples):
                data.append(wt(s=4, p_gen=pg, p=p, growth_limit=0, patch_limit=0)[0])
            numerics[i].append(np.mean(data))
            d_numerics[i].append(np.std(data) / np.sqrt(num_of_samples))
            analytics[i].append(X(p=p, pg=pg))
    for i, p in enumerate(ps):
        plt.plot(pgs, numerics[i], label=fr"Numerical $p=${p}", color=colors[i], linestyle="None", marker='o', markersize=4)
        plt.plot(pgs, analytics[i], label=fr"Analytical $p=${p}", color=colors[i])
    plt.yscale("log")
    plt.tick_params(axis="both", which="both", direction="in")
    plt.xlabel(r"$p_{\text{gen}}$")
    plt.ylabel("# time-steps to achieve an end-to-end link")
    plt.title("Four-segment merging-based repeater chain without patching limitation")
    plt.legend()
    plt.show()

