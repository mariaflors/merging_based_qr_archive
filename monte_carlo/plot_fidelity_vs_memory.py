"""
This script plots the fidelity in terms of the dephasing time for both the merging- and the swapping-based protocols
from the generated data.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skr_tools import secret_key_fraction, secret_key_rate, raw_rate

# Let us define some parameters for the plotting function.
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cm"
plt.rcParams["font.size"] = 12  # Change to 24 for small images
line_width = 1.5  # Change to 3 for small images
marker_size = 7


def plot_data(p, dephasing_times, total_distance, ks, growths, patches, num=50, save=False, path=''):
    """This function plots the averaged data of the fidelity, given a set of total distances for both the merging-based
    and the swapping-based protocols.

    Parameters
    ---------
    p : float
        Success probability of merging and swapping operations.
    dephasing_times : List of float
        List of dephasing time of the quantum memories in seconds.
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
        If True, the outcomes are saved in a .pdf file. Default is False.
    path : str
        Specification on where to save the .pdf file. Default is ''.
    """
    fig, ax = plt.subplots(figsize=[6.4, 4.8])
    for j, k in enumerate(ks):
        # Series of different `k`.
        data_sb = np.load(
            f"results/skr_fid_T_sb_k={k}_p={p}_td={total_distance}_Ts=[{dephasing_times[0]}, {dephasing_times[-1]}].npy")
        dt_sb = data_sb[0]
        fid_sb = data_sb[7]
        std_fid_sb = data_sb[8]
        dt_eb = [[[] for __ in patches] for __ in growths]
        fid_eb = [[[] for __ in patches] for __ in growths]
        std_fid_eb = [[[] for __ in patches] for __ in growths]
        for i, growth in enumerate(growths):
            for l, patch in enumerate(patches):
                data_eb = np.load(
                    f"results/skr_fid_T_mb_k={k}_p={p}_td={total_distance}_g={growth}_plim={patch}_Ts=[{dephasing_times[0]}, {dephasing_times[-1]}].npy")
                dt_eb[i][l] = data_eb[0]
                fid_eb[i][l] = data_eb[7]
                std_fid_eb[i][l] = data_eb[8]

        eb = [[[] for __ in patches] for __ in growths]
        sb = []
        d_eb = [[[] for __ in patches] for __ in growths]
        d_sb = []
        colors = ["#C3D278", "#DA7446", "#B44559", "#2C65A9"]
        line_styles = ['--', '-']

        for i in range(num):
            # Plot fidelity
            for n in range(len(growths)):
                for m in range(len(patches)):
                    eb[n][m].append(fid_eb[n][m][i])
                    d_eb[n][m].append(std_fid_eb[n][m][i])
            sb.append(fid_sb[i])
            d_sb.append(std_fid_sb[i])
        for n, growth in enumerate(growths):
            for m, patch in enumerate(patches):
                line_style_index = (n * len(patches) + m) % len(line_styles)
                ax.plot(dt_eb[n][m], eb[n][m], linestyle=line_styles[line_style_index], color=colors[j])
        ax.plot(dt_sb, sb, linestyle=':', color=colors[j])
    ax.set_ylabel("Fidelity")
    ax.set_xscale("log")
    ax.set_xlabel("Dephasing time [s]")
    # Add manual legend.
    SB = Line2D([0], [0], linestyle=':', label='SB', color='black', linewidth=line_width)
    MB1 = Line2D([0], [0], linestyle='--', label=r'MB $g_l=1$', color='black', linewidth=line_width)
    MB2 = Line2D([0], [0], linestyle='-', label=r'MB $g_l=2$', color='black', linewidth=line_width)
    col_lines = [
        Line2D([0], [0], marker='s', markersize=marker_size, linestyle='', label=fr'$k={k}$', markeredgecolor='black',
               markerfacecolor=color, linewidth=line_width) for k, color in zip(ks, colors)]
    plt.legend(handles=[SB, MB1, MB2] + col_lines, handlelength=1.7)
    # Edit ticks.
    plt.tick_params(axis="both", which="both", direction="in")
    plt.tick_params(axis="both", which="major", length=4)  # Change to length 8 for small plots
    plt.tick_params(axis="x", which="both", pad=6)
    # Set limits.
    ax.set_xlim(left=0.9, right=110)
    # Save or show the plot.
    if save is True:
        fig.savefig(path + "Fidelity_Memory.pdf", bbox_inches="tight")
    else:
        plt.show()
    return


if __name__ == "__main__":
    # PARAMETERS
    num_points = 50
    total_distance = 500000  # meters
    ks = [6, 7, 8, 9]
    dephasing_times = list(np.logspace(0, 2, num=num_points))  # seconds
    growths = [1, 2]
    patches = [4]

    # PLOT DATA
    plot_data(p=0.5, dephasing_times=dephasing_times, ks=ks, total_distance=total_distance, growths=growths,
              patches=patches, num=num_points, save=False, path='results/')
