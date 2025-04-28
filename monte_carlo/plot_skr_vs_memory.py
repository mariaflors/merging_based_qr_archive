"""
This script plots the increment ratio of secret key rate in terms of the dephasing time of the merging-based
protocol over the swapping-based protocol from the generated data.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skr_tools import secret_key_rate

# Let us define some parameters for the plotting function.
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cm"
plt.rcParams["font.size"] = 12
line_width = 1.5
marker_size = 7


def plot_data(p, dephasing_times, total_distance, ks, growths, patches, rate="skr", num=50, save=False, path=''):
    """This function plots the averaged data of the chosen rate (secret ket rate, secret key fraction or raw rate),
    given a set of total distances for both the merging-based and the swap-based protocols.

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
    rate : str
        Parameter to be plotted, can take two values:
        "skr" --> secret key rate,
        "fid" --> fidelity.
        Default is "skr".
    num : int
        Number of points per data series. Default is 50.
    save : True or False
        If True, the outcomes are saved in a .pdf file. Default is False.
    path : str
        Specification on where to save the .pdf file. Default is ''.
    """
    if rate != "skr" and rate != "fid":
        raise ValueError("The entered rate is not correct")
    fig, ax = plt.subplots(figsize=[6.4, 4.8])
    # Series of different `k`.
    for j, k in enumerate(ks):
        data_sb = np.load(
            f"results/skr_fid_T_sb_k={k}_p={p}_td={total_distance}_Ts=[{dephasing_times[0]}, {dephasing_times[-1]}].npy")
        dt_sb = data_sb[0]
        wt_sb = data_sb[1]
        std_wt_sb = data_sb[2]
        ez_sb = data_sb[3]
        std_ez_sb = data_sb[4]
        ex_sb = data_sb[5]
        std_ex_sb = data_sb[6]
        fid_sb = data_sb[7]
        std_fid_sb = data_sb[8]
        dt_eb = [[[] for __ in patches] for __ in growths]
        wt_eb = [[[] for __ in patches] for __ in growths]
        std_wt_eb = [[[] for __ in patches] for __ in growths]
        ez_eb = [[[] for __ in patches] for __ in growths]
        std_ez_eb = [[[] for __ in patches] for __ in growths]
        ex_eb = [[[] for __ in patches] for __ in growths]
        std_ex_eb = [[[] for __ in patches] for __ in growths]
        fid_eb = [[[] for __ in patches] for __ in growths]
        std_fid_eb = [[[] for __ in patches] for __ in growths]
        for i, growth in enumerate(growths):
            for l, patch in enumerate(patches):
                data_eb = np.load(
                    f"results/skr_fid_T_mb_k={k}_p={p}_td={total_distance}_g={growth}_plim={patch}_Ts=[{dephasing_times[0]}, {dephasing_times[-1]}].npy")
                dt_eb[i][l] = data_eb[0]
                wt_eb[i][l] = data_eb[1]
                std_wt_eb[i][l] = data_eb[2]
                ez_eb[i][l] = data_eb[3]
                std_ez_eb[i][l] = data_eb[4]
                ex_eb[i][l] = data_eb[5]
                std_ex_eb[i][l] = data_eb[6]
                fid_eb[i][l] = data_eb[7]
                std_fid_eb[i][l] = data_eb[8]

        ratio = [[[] for __ in patches] for __ in growths]
        d_ratio = [[[] for __ in patches] for __ in growths]
        colors = ["#C3D278", "#DA7446", "#B44559", "#2C65A9"]
        line_styles = ['--', '-']

        for i in range(num):
            if rate == "skr":
                # Plot ratio of secret key rate.
                result_sb = secret_key_rate(wt=wt_sb[i], d_wt=std_wt_sb[i], e_z=ez_sb[i], e_x=ex_sb[i],
                                            d_e_z=std_ez_sb[i], d_e_x=std_ex_sb[i], distance=total_distance / 2 ** k)
                for n in range(len(growths)):
                    for m in range(len(patches)):
                        result_eb = secret_key_rate(wt=wt_eb[n][m][i], d_wt=std_wt_eb[n][m][i], e_z=ez_eb[n][m][i],
                                                    e_x=ex_eb[n][m][i], d_e_z=std_ez_eb[n][m][i], d_e_x=std_ex_eb[n][m][i],
                                                    distance=total_distance / 2 ** k)
                        r = result_eb[0] / result_sb[0]
                        ratio[n][m].append(r)
                        d_ratio[n][m].append(
                            r * np.sqrt((result_eb[0] * result_sb[1]) ** 2 + (result_sb[0] * result_eb[1]) ** 2))
            else:
                # Plot ratio of fidelity.
                for n in range(len(growths)):
                    for m in range(len(patches)):
                        r = fid_eb[n][m][i] / fid_sb[i]
                        ratio[n][m].append(r)
                        d_ratio[n][m].append(
                            r * np.sqrt((fid_eb[n][m][i] * std_fid_sb[i]) ** 2 + (fid_sb[i] * std_fid_eb[n][m][i]) ** 2))

        for n, growth in enumerate(growths):
            for m, patch in enumerate(patches):
                line_style_index = (n * len(patches) + m) % len(line_styles)
                ax.plot(dt_eb[n][m], ratio[n][m], linestyle=line_styles[line_style_index], color=colors[j])
    ax.set_xscale("log")
    # Make the x-axis in km.
    ax.set_xlabel("Dephasing time [s]")
    if rate == "skr":
        ax.set_ylabel(r'S$_{\mathrm{MB}}$ / S$_{\mathrm{SB}}$')
        name = "SKR"
    else:
        ax.set_ylabel(r'F$_{\mathrm{MB}}$ / F$_{\mathrm{SB}}$')
        name = "FID"
    # Set title.
    # bbox = dict(facecolor="white", edgecolor="black", boxstyle='round,pad=0.3')
    # ax.set_title(fr"$p={p_fuse}$, Total distance$={total_distance}$ m", loc="center", y=1.0, pad=-20, bbox=bbox)
    # Edit ticks.
    plt.tick_params(axis="both", which="both", direction="in")
    plt.tick_params(axis="both", which="major", length=4)
    plt.tick_params(axis="x", which="both", pad=6)
    # Add manual legend.
    EB1 = Line2D([0], [0], linestyle='--', label=r'$g_l=1$', color='black', linewidth=line_width)
    EB2 = Line2D([0], [0], linestyle='-', label=r'$g_l=2$', color='black', linewidth=line_width)
    col_lines = [
        Line2D([0], [0], marker='s', markersize=marker_size, linestyle='', label=fr'$k={k}$', markeredgecolor='black',
               markerfacecolor=color, linewidth=line_width) for k, color in zip(ks, colors)]
    plt.legend(handles=[EB1, EB2] + col_lines, handlelength=1.7)
    # Set limits.
    ax.set_xlim(left=0.9, right=110)
    # Save or show the plot.
    if save is True:
        fig.savefig(path + f"Ratio_{name}.pdf", bbox_inches="tight")
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
