"""
A script for Monte Carlo sampling the waiting time from the Bell-state based doubling protocol.
For more details on the noise analysis see Appendix A from paper `Merging-Based Quantum Repeater`.
"""
import numpy as np


def sample_swapbased(k, p_gen, p):
    """Function that samples the waiting time of the swapping-based repeater approach with the double distance protocol.
    Parameters
    ----------
    k : int
        Number of steps, such that the number of segments is 2 ** k.
    p_gen : float
        Success probability of generating an elementary link.
    p : float
        Success probability of swapping two Bell pairs.

    Returns
    -------
    float : waiting time to create an end-to-end Bell pair over 2 ** k segments using the swapping-based protocol.
    list : list of length s + 1 that contains the waiting times of the qubits.
    """
    if k == 0:
        # Sample from geometric distribution with bias p_gen.
        return np.random.geometric(p=p_gen), [0, 0]
    else:
        # We split the chain in two, and we compute the waiting time of the chain and the waiting time of each qubit.
        t_left, q_left = sample_swapbased(k=k - 1, p_gen=p_gen, p=p)
        t_right, q_right = sample_swapbased(k=k - 1, p_gen=p_gen, p=p)
        # The total time is the maximum time between the two sides.
        t = max(t_left, t_right)
        # Update the times of the qubits since, the qubits from one side have to wait for the other side.
        if t_left != t_right:
            diff = abs(t_left - t_right)
            if t_left < t_right:
                q_left[0] += diff
                q_left[-1] += diff
            else:
                q_right[0] += diff
                q_right[-1] += diff
        # Attempt the swapping of the two Bell pairs
        if np.random.random() < p:
            # Successful swapping operation.
            q = q_left[:-1] + [q_left[-1] + q_right[0]] + q_right[1:]
            return t, q
        else:
            # Failed swapping operation. Protocol restarts.
            retry = sample_swapbased(k=k, p_gen=p_gen, p=p)
            return t + retry[0], retry[1]
