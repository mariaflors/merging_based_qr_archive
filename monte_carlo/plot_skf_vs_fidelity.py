"""
This script plots the secret key fraction in terms of the fidelity of the merging- and the swapping-based protocols from
the generated data.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skr_tools import secret_key_fraction

# Let us define some parameters for the plotting function.
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cm"
plt.rcParams["font.size"] = 12
line_width = 1.5
marker_size = 7


def plot_data(p=0.5, k=6, dephasing_time=10, total_distances=np.linspace(1e3, 2200000, num=50), growth=1, patch=4,
              num=50, save=False, path=''):

    """This function plots the secret key fraction in terms of the fidelity of the merging- and the swapping-based
    protocols.

    Parameters
    ---------
    p : float
        Success probability of merging and swapping operations. Default is 0.5.
    k : int
        Number of steps, such that the number of segments is 2 ** k. Default is 6.
    dephasing_time : float
        Dephasing time of the quantum memories in seconds. Default is 10.
    total_distances : List of float
        List of total distances in meters. Default is 'np.linspace(1e3, 2200000, num=50)'.
    growth : int
        Limit to how much a gap can grow. Default is 1.
    patch: int
        Limit to at what size of the cluster can the patching be attempted. Default is 4.
    num : int
        Number of points per data series. Default is 50.
    save : True or False
        If True, the outcomes are saved in a .pdf file. Default is False.
    path : str
        Specification on where to save the .pdf file. Default is ''.
    """
    fig, ax = plt.subplots(figsize=[6.4, 4.8])
    data_sb = np.load(f"results/skr_fid_td_sb_k={k}_p={p}_T={dephasing_time}_td={total_distances[-1]}.npy")
    ez_sb = data_sb[3]
    std_ez_sb = data_sb[4]
    ex_sb = data_sb[5]
    std_ex_sb = data_sb[6]
    fid_sb = data_sb[7]
    data_eb = np.load(
        f"results/skr_fid_td_mb_k={k}_p={p}_T={dephasing_time}_g={growth}_plim={patch}_td={total_distances[-1]}.npy")
    ez_eb = data_eb[3]
    std_ez_eb = data_eb[4]
    ex_eb = data_eb[5]
    std_ex_eb = data_eb[6]
    fid_eb = data_eb[7]

    eb = []
    sb = []
    d_eb = []
    d_sb = []
    for i in range(num):
        # Plot secret key fraction.
        result_eb = secret_key_fraction(e_z=ez_eb[i], e_x=ex_eb[i], d_e_z=std_ez_eb[i], d_e_x=std_ex_eb[i])
        eb.append(result_eb[0])
        d_eb.append(result_eb[1])

        result_sb = secret_key_fraction(e_z=ez_sb[i], e_x=ex_sb[i], d_e_z=std_ez_sb[i], d_e_x=std_ex_sb[i])
        sb.append(result_sb[0])
        d_sb.append(result_sb[1])

    ax.plot(fid_eb, eb, color="black")
    ax.plot(fid_sb, sb, linestyle=':', color="black")

    ax.set_xlabel("Fidelity")
    ax.set_ylabel('Secret key fraction')
    # Edit ticks.
    plt.tick_params(axis="both", which="both", direction="in")
    plt.tick_params(axis="both", which="major", length=4)
    plt.tick_params(axis="x", which="both", pad=6)
    # Add manual legend.
    SB = Line2D([0], [0], linestyle=':', label='SB', color='black', linewidth=line_width)
    MB = Line2D([0], [0], linestyle='-', label='MB', color='black', linewidth=line_width)
    plt.legend(handles=[SB, MB], handlelength=1.7)
    # Set limits.
    ax.set_xlim(left=0.745,right=1.005)
    # Save or show the plot.
    if save is True:
        fig.savefig(path + "SKF_FID.pdf", bbox_inches="tight")
    else:
        plt.show()
    return


if __name__ == "__main__":
    # PLOT DATA
    plot_data(save=False, path='results/')
