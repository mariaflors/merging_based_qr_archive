"""
This script plots the secret key rate in terms of the probability of success of the merging operation for the
merging-based protocol with and without the patching limitation from the generated data.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skr_tools import secret_key_rate

# Let us define some parameters for the plotting function.
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cm"
plt.rcParams["font.size"] = 12  # Change to 24 for small images
line_width = 1.5  # Change to 3 for small images
marker_size = 7


def plot_data(ps, dephasing_time, total_distance, ks, growths, patches, num=50, save=False, path=''):
    """This function plots the averaged data of the secret ket rate, given a set of success probabilities of the merging
    operation for the merging-based protocol with the specified growth and patch limit.

    Parameters
    ---------
    ps : list
        List of success probabilities of fusing two cluster states into one.
    dephasing_time : float
        Dephasing time of the quantum memories in seconds.
    total_distance : float
        Total distance in meters.
    ks : list of int
        List of number of steps, such that the number of segments is 2 ** k.
    growths : list of int
        List of the limits to how much a gap can grow.
    patches: list of int
        List of the limits to at what size of the cluster can the patching be attempted.
    num : int
        Number of points per data series. Default is 50.
    save : True or False
        If True, the outcomes are saved in a .npy file. Default is False.
    path : str
        Specification on where to save the .npy file. Default is ''.
    """
    fig, ax = plt.subplots(figsize=[6.4, 4.8])
    # Series of different k.
    for j, k in enumerate(ks):
        pf_eb = [[[] for __ in patches] for __ in growths]
        wt_eb = [[[] for __ in patches] for __ in growths]
        std_wt_eb = [[[] for __ in patches] for __ in growths]
        ez_eb = [[[] for __ in patches] for __ in growths]
        std_ez_eb = [[[] for __ in patches] for __ in growths]
        ex_eb = [[[] for __ in patches] for __ in growths]
        std_ex_eb = [[[] for __ in patches] for __ in growths]
        for i, growth in enumerate(growths):
            for l, patch in enumerate(patches):
                data_eb = np.load(
                    f"results/skr_pf_mb_k={k}_T={dephasing_time}_g={growth}_plim={patch}_td={total_distance}_pf={ps[-1]}.npy")
                pf_eb[i][l] = data_eb[0]
                wt_eb[i][l] = data_eb[1]
                std_wt_eb[i][l] = data_eb[2]
                ez_eb[i][l] = data_eb[3]
                std_ez_eb[i][l] = data_eb[4]
                ex_eb[i][l] = data_eb[5]
                std_ex_eb[i][l] = data_eb[6]

        eb = [[[] for __ in patches] for __ in growths]
        d_eb = [[[] for __ in patches] for __ in growths]
        colors = ["#C3D278", "#DA7446", "#2C65A9"]
        linestyles = ['--', '-']

        for i in range(num):
            # Plot secret key rate
            for n in range(len(growths)):
                for m in range(len(patches)):
                    result_eb = secret_key_rate(wt=wt_eb[n][m][i], d_wt=std_wt_eb[n][m][i], e_z=ez_eb[n][m][i],
                                                e_x=ex_eb[n][m][i], d_e_z=std_ez_eb[n][m][i], d_e_x=std_ex_eb[n][m][i],
                                                distance=pf_eb[n][m][i] / 2 ** k)
                    eb[n][m].append(result_eb[0])
                    d_eb[n][m].append(result_eb[1])

        for n, growth in enumerate(growths):
            for m, patch in enumerate(patches):
                linestyle_index = (n * len(patches) + m) % len(linestyles)
                ax.plot(pf_eb[n][m], eb[n][m], linestyle=linestyles[linestyle_index], color=colors[j])

    ax.set_xlabel(r"$p$")
    ax.set_ylabel("Secret key rate [Hz]")
    ax.set_ylim(bottom=3e5)
    ax.set_xlim(left=0.05, right=0.95)
    ax.set_yscale("log")
    # Add manual legend.
    LIM = Line2D([0], [0], linestyle='-', label=r'Limited', color='black', linewidth=line_width)
    NLIM = Line2D([0], [0], linestyle='--', label=r'Unlimited', color='black', linewidth=line_width)
    col_lines = [
        Line2D([0], [0], marker='s', markersize=marker_size, linestyle='', label=fr'$k={k}$', markeredgecolor='black',
               markerfacecolor=color, linewidth=line_width) for k, color in zip(ks, colors)]
    plt.legend(handles=[LIM, NLIM] + col_lines, handlelength=1.7, loc=4)
    # Edit ticks.
    plt.tick_params(axis="both", which="both", direction="in")
    plt.tick_params(axis="both", which="major", length=4)  # Change to length 8 for small plots
    plt.tick_params(axis="x", which="both", pad=6)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    # Save or show the plot.
    if save is True:
        fig.savefig(path + f"LIMIT.pdf", bbox_inches="tight")
    else:
        plt.show()
    return


if __name__ == "__main__":
    # PARAMETERS
    num_points = 50
    total_distance = 20000  # meters
    ps = np.linspace(0.05, 1, num=50)
    ks = [2, 3, 4]
    growths = [0]
    patches = [0, 4]

    # PLOT DATA
    plot_data(ps=ps, dephasing_time=10, ks=ks, total_distance=total_distance, growths=growths, patches=patches,
              num=num_points, save=False, path='results/')
