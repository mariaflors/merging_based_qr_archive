"""
This script plots the secret key rate/raw rate/secret key fraction/fidelity  in terms of the total distance for both the
merging- and the swapping-based protocols from the generated data.
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


def plot_data(p, dephasing_time, total_distances, ks, growths, patches, rate="skr", num=50, save=False, path=''):
    """This function plots the averaged data of the chosen rate (secret ket rate, secret key fraction, raw rate, or
    fidelity), given a set of total distances for both the merging-based and the swapping-based protocols.

    Parameters
    ---------
    p : float
        Success probability of merging and swapping operations.
    dephasing_time : float
        Dephasing time of the quantum memories in seconds.
    total_distances : List of float
        List of total distances in meters.
    ks : list of int
        List of number of steps, such that the number of segments is 2 ** k.
    growths : list of int
        List of the limits to how much a gap can grow.
    patches: list of int
        List of the limits to at what size of the cluster can the patching be attempted.
    rate : str
        Parameter to be plotted, can take four values:
        "skr" --> secret key rate,
        "skf" --> secret key fraction,
        "rr"  --> raw rate,
        "fid" --> fidelity.
        Default is "skr".
    num : int
        Number of points per data series. Default is 50.
    save : True or False
        If True, the outcomes are saved in a .pdf file. Default is False.
    path : str
        Specification on where to save the .pdf file. Default is ''.
    """
    if rate != "skr" and rate != "skf" and rate != "rr" and rate != "fid":
        raise ValueError("The entered rate is not correct")
    fig, ax = plt.subplots(figsize=[6.4, 4.8])
    # Series of different `k`.
    for j, k in enumerate(ks):
        data_sb = np.load(f"results/skr_fid_td_sb_k={k}_p={p}_T={dephasing_time}_td={total_distances[-1]}.npy")
        td_sb = data_sb[0]
        wt_sb = data_sb[1]
        std_wt_sb = data_sb[2]
        ez_sb = data_sb[3]
        std_ez_sb = data_sb[4]
        ex_sb = data_sb[5]
        std_ex_sb = data_sb[6]
        fid_sb = data_sb[7]
        std_fid_sb = data_sb[8]
        td_eb = [[[] for __ in patches] for __ in growths]
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
                    f"results/skr_fid_td_mb_k={k}_p={p}_T={dephasing_time}_g={growth}_plim={patch}_td={total_distances[-1]}.npy")
                td_eb[i][l] = data_eb[0]
                wt_eb[i][l] = data_eb[1]
                std_wt_eb[i][l] = data_eb[2]
                ez_eb[i][l] = data_eb[3]
                std_ez_eb[i][l] = data_eb[4]
                ex_eb[i][l] = data_eb[5]
                std_ex_eb[i][l] = data_eb[6]
                fid_eb[i][l] = data_eb[7]
                std_fid_eb[i][l] = data_eb[8]

        eb = [[[] for __ in patches] for __ in growths]
        sb = []
        d_eb = [[[] for __ in patches] for __ in growths]
        d_sb = []
        colors = ["#C3D278", "#DA7446", "#B44559", "#2C65A9"]
        line_styles = ['--', '-']

        for i in range(num):
            if rate == "skr":
                # Plot secret key rate.
                for n in range(len(growths)):
                    for m in range(len(patches)):
                        result_eb = secret_key_rate(wt=wt_eb[n][m][i], d_wt=std_wt_eb[n][m][i], e_z=ez_eb[n][m][i],
                                                    e_x=ex_eb[n][m][i], d_e_z=std_ez_eb[n][m][i],
                                                    d_e_x=std_ex_eb[n][m][i], distance=td_eb[n][m][i] / 2 ** k)
                        eb[n][m].append(result_eb[0])
                        d_eb[n][m].append(result_eb[1])

                result_sb = secret_key_rate(wt=wt_sb[i], d_wt=std_wt_sb[i], e_z=ez_sb[i], e_x=ex_sb[i],
                                            d_e_z=std_ez_sb[i], d_e_x=std_ex_sb[i], distance=td_sb[i] / 2 ** k)
                sb.append(result_sb[0])
                d_sb.append(result_sb[1])
            elif rate == "skf":
                # Plot secret key fraction.
                for n in range(len(growths)):
                    for m in range(len(patches)):
                        result_eb = secret_key_fraction(e_z=ez_eb[n][m][i], e_x=ex_eb[n][m][i],
                                                        d_e_z=std_ez_eb[n][m][i], d_e_x=std_ex_eb[n][m][i])
                        eb[n][m].append(result_eb[0])
                        d_eb[n][m].append(result_eb[1])

                result_sb = secret_key_fraction(e_z=ez_sb[i], e_x=ex_sb[i], d_e_z=std_ez_sb[i], d_e_x=std_ex_sb[i])
                sb.append(result_sb[0])
                d_sb.append(result_sb[1])
            elif rate == "rr":
                # Plot raw rate.
                for n in range(len(growths)):
                    for m in range(len(patches)):
                        result_eb = raw_rate(wt=wt_eb[n][m][i], d_wt=std_wt_eb[n][m][i],
                                             distance=td_eb[n][m][i] / 2 ** k)
                        eb[n][m].append(result_eb[0])
                        d_eb[n][m].append(result_eb[1])

                result_sb = raw_rate(wt=wt_sb[i], d_wt=std_wt_sb[i], distance=td_sb[i] / 2 ** k)
                sb.append(result_sb[0])
                d_sb.append(result_sb[1])
            elif rate == "fid":
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
                ax.plot(td_eb[n][m], eb[n][m], linestyle=line_styles[line_style_index], color=colors[j])
        ax.plot(td_sb, sb, linestyle=':', color=colors[j])
    # Make the x-axis in km.
    m2km = lambda x, _: f'{x / 1000: g}'
    ax.xaxis.set_major_formatter(m2km)
    ax.set_xlabel("Total distance [km]")
    if rate == "skr":
        ax.set_ylabel("Secret key rate [Hz]")
        ax.set_ylim(bottom=0.99, top=5e5)
        ax.set_yscale("log")
        name = "SKR"
    elif rate == "skf":
        ax.set_ylabel("Secret key fraction")
        name = "SKF"
    elif rate == "rr":
        ax.set_ylabel("Raw rate [Hz]")
        ax.set_ylim(bottom=0.99, top=5e5)
        ax.set_yscale("log")
        name = "RR"
    else:
        ax.set_ylabel("Fidelity")
        name = "FID"
    ax.set_xlim(left=-80000, right=2200000)
    # Set title.
    # bbox = dict(facecolor="white", edgecolor="black", boxstyle='round,pad=0.3')
    # ax.set_title(fr"$p={p_fuse}$, $T={dephasing_time}$ s", loc="center", y=1.0, pad=-20, bbox=bbox)
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
    # Save or show the plot.
    if save is True:
        fig.savefig(path + f"{name}.pdf", bbox_inches="tight")
    else:
        plt.show()
    return


if __name__ == "__main__":
    # PARAMETERS
    num_points = 50
    total_distances = np.linspace(1e3, 2200000, num=num_points)  # meters
    ks = [6, 7, 8, 9]
    dephasing_times = [10]  # seconds
    growths = [1, 2]
    patches = [4]

    # PLOT DATA
    plot_data(p=0.5, dephasing_time=10, ks=ks, total_distances=total_distances, growths=growths, patches=patches,
              rate="fid", num=num_points, save=False, path='results/')
