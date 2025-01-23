"""
This script generates the data to plot the increment ratio of secret key rate in terms of the dephasing time of the
merging-based protocol over the swapping-based protocol.
To obtain the data we parallelize using Pool from multiprocessing.
"""
import numpy as np
import sys
from swap_based import sample_swapbased
from merge_based import wt
import os
from multiprocessing import Pool
from noisy_graph_states.tools.strategies_1d import side_to_side, every_second
from skr_tools import final_bell_pair, q_bers


def mb_run_it(total_distance, k, p, dephasing_time, growth, patch, num_samples_per_point, cs=2e8):
    """This function generates a point of data to compute the secret key rate for a given total distance, a
    probability of the merging operation and a number of segments. It computes the merging-based with the specified
    growth and patch limit.
    It returns the waiting time and both QBERs, and the corresponding standard deviation of the mean of each parameter.

    Parameters
    ---------
    total_distance : float
        Total distance in meters.
    k : int
        Number of steps, such that the number of segments is 2 ** k.
    p : float
        Success probability of merging two cluster states into one.
    dephasing_time : float
        Dephasing time of the quantum memories in seconds.
    growth : int
        Limit to how much a gap can grow.
    patch : int
        Limit to at what size of the cluster can the patching be attempted.
    num_samples_per_point : int
        Number of samples to compute a single point.
    cs : float
        Communication speed in m/s. Default is 2e8 (speed of light in optical fibre).

    Returns
    -------
    list :  First element corresponds to the mean waiting time in time-steps.
            Second element is the standard deviation of the mean of the waiting time
            Third element is the mean QBER in the Z basis.
            Fourth element is the standard deviation of the mean of the QBER in the Z basis.
            Fifth element is the mean QBER in the X basis.
            Sixth element is the standard deviation of the mean of the QBER in the X basis.
    """
    # Get the internode distance.
    distance = total_distance / 2 ** k
    # Determine the probability of generation for the given distance.
    p_gen = np.exp(-distance / 22e3)
    # Create the strategy to manipulate the 1D cluster.
    sts = side_to_side(2 ** k + 1)
    # Generate average data for merging-based case.
    w_times = []
    q_times = []
    for __ in range(num_samples_per_point):
        result = wt(s=2 ** k, p_gen=p_gen, p=p, growth_limit=growth, patch_limit=patch)
        w_times.append(result[0])
        q_times.append(result[1])
    # Compute averages of the waiting time.
    wt_eb = np.mean(w_times)
    std_wt_eb = np.std(w_times) / np.sqrt(num_samples_per_point)
    # Compute averages of the QBERs.
    e_z = []
    e_x = []
    for qt in q_times:
        qt = [element * 2 * distance / cs for element in qt]
        state = final_bell_pair(time=qt, strategy=sts, dephasing_time=dephasing_time)
        errors = q_bers(state=state)
        e_z.append(errors[0])
        e_x.append(errors[1])
    ez_eb = np.mean(e_z)
    std_ez_eb = np.std(e_z) / np.sqrt(num_samples_per_point)
    ex_eb = np.mean(e_x)
    std_ex_eb = np.std(e_x) / np.sqrt(num_samples_per_point)
    return [dephasing_time, wt_eb, std_wt_eb, ez_eb, std_ez_eb, ex_eb, std_ex_eb]


def sb_run_it(total_distance, k, p, dephasing_time, num_samples_per_point, cs=2e8):
    """This function generates a point of data to compute the secret key rate for a given internode distance, a
    probability of the swapping operation and a number of segments. It computes the swapping-based protocol.
    It returns the waiting time and both QBERs, and the corresponding standard deviation of the mean of each parameter.

    Parameters
    ---------
    total_distance : float
        Total distance in meters.
    k : int
        Number of steps, such that the number of segments is 2 ** k.
    p : float
        Success probability of swapping two Bell pairs.
    dephasing_time : float
        Dephasing time of the quantum memories in seconds.
    num_samples_per_point : int
        Number of samples to compute a single point.
    cs : float
        Communication speed in m/s. Default is 2e8 (speed of light in optical fibre).

    Returns
    -------
    list :  First element corresponds to the mean waiting time in time-steps.
            Second element is the standard deviation of the mean of the waiting time
            Third element is the mean QBER in the Z basis.
            Fourth element is the standard deviation of the mean of the QBER in the Z basis.
            Fifth element is the mean QBER in the X basis.
            Sixth element is the standard deviation of the mean of the QBER in the X basis.
    """
    # Get the internode distance.
    distance = total_distance / 2 ** k
    # Determine the probability of generation for the given distance.
    p_gen = np.exp(-distance / 22e3)
    # Create the strategy to manipulate the 1D cluster.
    esq = every_second(2 ** k + 1)
    # Generate average data for swapping-based case.
    w_times = []
    q_times = []
    for __ in range(num_samples_per_point):
        result = sample_swapbased(k=k, p_gen=p_gen, p=p)
        w_times.append(result[0])
        q_times.append(result[1])
    # Compute averages of the waiting time.
    wt_sb = np.mean(w_times)
    std_wt_sb = np.std(w_times) / np.sqrt(num_samples_per_point)
    # Compute averages of the QBERs.
    e_z = []
    e_x = []
    for qt in q_times:
        qt = [element * 2 * distance / cs for element in qt]
        state = final_bell_pair(time=qt, strategy=esq, dephasing_time=dephasing_time)
        errors = q_bers(state=state)
        e_z.append(errors[0])
        e_x.append(errors[1])
    ez_sb = np.mean(e_z)
    std_ez_sb = np.std(e_z) / np.sqrt(num_samples_per_point)
    ex_sb = np.mean(e_x)
    std_ex_sb = np.std(e_x) / np.sqrt(num_samples_per_point)
    return [dephasing_time, wt_sb, std_wt_sb, ez_sb, std_ez_sb, ex_sb, std_ex_sb]


def generate_data(ks, ps, dephasing_times, growths, patches, total_distances, num_samples_per_point, pool_size=4):
    """This function generates the averaged data given a set of internode distances
    for both the merging-based and the swapping-based protocols.

    Parameters
    ---------
    ks : list of int
        List of number of steps, such that the number of segments is 2 ** k.
    ps : list of float
        List of the success probabilities of merging and swapping operations.
    dephasing_times : list of float
        List of the dephasing times of the quantum memories in seconds.
    growths : list of int
        List of the limits to how much a gap can grow.
    patches : list of int
        List of the limits to at what size of the cluster can the patching be attempted.
    total_distances : List of float
        List of total distances in meters.
    num_samples_per_point : int
        Number of samples to compute a single point.
    pool_size : int
        Number of pools. Default is 4.
    """
    # Generate a folder to save data.
    if not os.path.exists("results"):
        os.makedirs("results")
    # Create the pool.
    pool = Pool(pool_size)
    res_sb = {}
    res_eb = {}
    for total_distance in total_distances:
        for k in ks:
            for p in ps:
                star_args_sb = [(total_distance, k, p, dephasing_time, num_samples_per_point) for dephasing_time in
                                dephasing_times]
                res_sb[(k, p, total_distance)] = pool.starmap_async(sb_run_it, star_args_sb, chunksize=1)
                for growth in growths:
                    for patch in patches:
                        star_args_eb = [
                            (total_distance, k, p, dephasing_time, growth, patch, num_samples_per_point) for
                            dephasing_time in dephasing_times]
                        res_eb[(k, p, total_distance, growth, patch)] = pool.starmap_async(mb_run_it, star_args_eb,
                                                                                             chunksize=1)

    pool.close()
    # Unpack data.
    for total_distance in total_distances:
        for k in ks:
            for p in ps:
                current_res_sb = res_sb[(k, p, total_distance)].get()
                res_unpacked_sb = list(zip(*current_res_sb))
                dt_sb = res_unpacked_sb[0]
                wt_sb = res_unpacked_sb[1]
                std_wt_sb = res_unpacked_sb[2]
                ez_sb = res_unpacked_sb[3]
                std_ez_sb = res_unpacked_sb[4]
                ex_sb = res_unpacked_sb[5]
                std_ex_sb = res_unpacked_sb[6]
                # Save the data of the swapping-based.
                np.save(
                    f"results/skr_T_sb_k={k}_p={p}_td={total_distance}_Ts=[{dephasing_times[0]}, {dephasing_times[-1]}]",
                    (dt_sb, wt_sb, std_wt_sb, ez_sb, std_ez_sb, ex_sb, std_ex_sb))
                for growth in growths:
                    for patch in patches:
                        current_res_eb = res_eb[(k, p, total_distance, growth, patch)].get()
                        res_unpacked_eb = list(zip(*current_res_eb))
                        dt_eb = res_unpacked_eb[0]
                        wt_eb = res_unpacked_eb[1]
                        std_wt_eb = res_unpacked_eb[2]
                        ez_eb = res_unpacked_eb[3]
                        std_ez_eb = res_unpacked_eb[4]
                        ex_eb = res_unpacked_eb[5]
                        std_ex_eb = res_unpacked_eb[6]
                        # Save the data of the merging-based approach for a certain growth.
                        np.save(
                            f"results/skr_T_mb_k={k}_p={p}_td={total_distance}_g={growth}_plim={patch}_Ts=[{dephasing_times[0]}, {dephasing_times[-1]}]",
                            (dt_eb, wt_eb, std_wt_eb, ez_eb, std_ez_eb, ex_eb, std_ex_eb))
    return


if __name__ == "__main__":

    sys.setrecursionlimit(100000)

    # PARAMETERS
    num_samples_per_point = 100000
    num_points = 50
    total_distances = [500000]  # meters
    ps = [0.5]
    ks = [6, 7, 8, 9]
    dephasing_times = list(np.logspace(0, 2, num=num_points))  # seconds
    growths = [1, 2]
    patches = [4]
    pool_size = 20

    # GENERATE DATA
    generate_data(ks=ks, ps=ps, dephasing_times=dephasing_times, growths=growths, patches=patches,
                  total_distances=total_distances, num_samples_per_point=num_samples_per_point, pool_size=pool_size)
