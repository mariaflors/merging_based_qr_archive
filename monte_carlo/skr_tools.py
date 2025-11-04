"""
Here there are the functions needed to compute the secret key rate given the average waiting time, its standard
deviation of the mean, and the waiting time for each memory qubit in the protocol.
All functions compute the propagation of uncertainties.
"""
import numpy as np
import math
import noisy_graph_states as ngs
import graphepp as gg
import noisy_graph_states.libs.matrix as mat
import noisy_graph_states.libs.graph as gt


def time_noise_coefficient(time_interval, dephasing_time):
    """This function computes the noise probability of dephasing noise of a qubit with a memory of a certain dephasing
    time which has waited a certain time interval.

    Parameters
    ----------
    time_interval : scalar
        Time in that the memory has been waiting in seconds.
    dephasing_time : scalar
        Dephasing time of the memory in seconds.
    """
    return (1 - np.exp(-time_interval / dephasing_time)) / 2


def final_bell_pair(time, strategy, dephasing_time):
    """This function computes the final noisy state of a Bell pair between the two end point of a 1D cluster, where the
    qubits in between are measured using a certain `strategy`.
    The qubits are subject to time-dependent single-qubit dephasing noise.

    Parameters
    ---------
    time : list
        List where each element represents how long each qubit has waited in seconds.
    strategy : ngs.Strategy
        Chosen strategy to measure the qubits in between the target ones.
    dephasing_time : float
        Dephasing time of the quantum memories in seconds.

    Returns
    ------
    state : ngs.State
        Final state of a Bell pair between the ends of the chain.
    """
    # Number of qubits in the final 1D cluster.
    qubits = len(time)
    # Create the graph.
    graph = gg.Graph(N=qubits, E=[(i, i + 1) for i in range(0, qubits - 1)])
    # Create the noiseless state.
    state = ngs.State(graph=graph, maps=[])

    # Add the noise.
    for i, t in enumerate(time):
        state = ngs.z_noise(state, [i], time_noise_coefficient(t, dephasing_time))

    # Perform the measurements in the Y basis.
    state = strategy(state)
    return state


def q_bers(state):
    """This function computes the Quantum Bit Error Rates (QBERs) of a noisy Bell pair.

    Parameters
    ---------
    state : ngs.State
        State of a Bell pair between the ends of the chain.

    Returns
    ------
    e_z[0][0] : float
        QBER in the Z basis.
    e_x[0][0] : float
        QBER in the X basis.
    """
    # Number of qubits in the initial state (initial Bell pairs).
    qubits = state.graph.N
    # Reduced density matrix of the qubits at the end nodes (the rest have been previously measured).
    dm = ngs.noisy_bp_dm(state, [0, qubits - 1])
    # Bring the state from the graph state basis to the computational basis.
    dm = np.dot(np.kron(mat.I(2), mat.Ha), np.dot(dm, np.kron(mat.I(2), mat.Ha)))
    # Define the projectors.
    z0z1 = np.kron(mat.z0, mat.z1)
    z1z0 = np.kron(mat.z1, mat.z0)
    x0x1 = np.kron(mat.x0, mat.x1)
    x1x0 = np.kron(mat.x1, mat.x0)
    # Compute the quantum bit error rates.
    e_z = np.dot(mat.H(z0z1), np.dot(dm, z0z1)) + np.dot(mat.H(z1z0), np.dot(dm, z1z0))
    e_x = np.dot(mat.H(x0x1), np.dot(dm, x0x1)) + np.dot(mat.H(x1x0), np.dot(dm, x1x0))
    return e_z[0][0], e_x[0][0]


def binary_entropy(x, d_x):
    """This function computes the binary entropy.

    Parameters
    ----------
    x : float
        Takes values from 0 to 1.
    d_x : float
        Uncertainty of `x`.

    Returns
    -------
    entropy: float
        Ranges from 0 to 1.
    d_entropy : float
        Uncertainty of `entropy`.
    """
    if x == 0.0:
        entropy = - (1 - x) * math.log2(1 - x)
        d_entropy = (math.log2(1 - x) + 1 / math.log(2)) * d_x
    elif 1 - x == 0.0:
        entropy = - x * math.log2(x)
        d_entropy = - (math.log2(x) + 1 / math.log(2)) * d_x
    else:
        entropy = - x * math.log2(x) - (1 - x) * math.log2(1 - x)
        d_entropy = (math.log2(1 - x) - math.log2(x)) * d_x
    return entropy, d_entropy


def secret_key_fraction(e_z, e_x, d_e_z, d_e_x):
    """This function computes the secret key fraction given the QBERs.

    Parameters
    ----------
    e_z : float
        QBER in the Z basis.
    d_e_z : float
        Uncertainty of `e_z`.
    e_x : float
        QBER in the X basis.
    d_e_x : float
        Uncertainty of `e_x`.

    Returns
    -------
    r : float
        Secret key fraction.
    d_r : float
        Uncertainty of `r`.
    """
    r_z, d_r_z = binary_entropy(np.real_if_close(e_z), d_e_z)
    r_x, d_r_x = binary_entropy(np.real_if_close(e_x), d_e_x)
    r = 1 - r_z - r_x
    d_r = np.abs(r) * np.sqrt(d_r_z ** 2 + d_r_x ** 2)
    # Take only positive values
    r = max([r, 0.0])
    d_r = 0.0 if r == 0.0 else d_r
    return r, d_r


def raw_rate(wt, d_wt, distance, cs=2e8):
    """This function computes the raw rate given the mean time until 1st end-end link in time-steps (duration of an
    attempt of the generation of an elementary link).

    Parameters
    ----------
    wt : float
        Waiting time in time-steps.
    d_wt : float
        Uncertainty of `wt`.
    distance : float
        Internode distance in meters.
    cs : float
        Communication speed in m/s. Default is 2e8 (speed of light in optical fibre).

    Returns
    -------
    raw_rate : float
        Raw rate.
    d_raw_rate : float
        Uncertainty of `raw_rate`.
    """
    ct = 2 * distance / cs  # Communication time corresponding to a time-step
    rr = 1 / (wt * ct)
    d_rr = rr * d_wt / (ct * wt ** 2)
    return rr, d_rr


def secret_key_rate(wt, d_wt, e_z, e_x, d_e_z, d_e_x, distance, cs=2e8):
    """This function computes the secret key rate.

    Parameters
    ----------
    wt : float
        Waiting time in time-steps.
    d_wt : float
        Uncertainty of `wt`.
    e_z : float
        QBER in the Z basis.
    d_e_z : float
        Uncertainty of `e_z`.
    e_x : float
        QBER in the X basis.
    d_e_x : float
        Uncertainty of `e_x`.
    distance : float
        Internode distance in meters.
    cs : float
        Communication speed in m/s. Default is 2e8 (speed of light in optical fibre).

    Returns
    -------
    skr : float
        Secret key fraction.
    d_skr : float
        Uncertainty of `r`.
    """
    rr, d_rr = raw_rate(wt, d_wt, distance, cs)
    skf, d_skf = secret_key_fraction(e_z, e_x, d_e_z, d_e_x)
    skr = rr * skf
    d_skr = skr * np.sqrt((skf * d_rr) ** 2 + (rr * d_skf) ** 2)
    return skr, d_skr


def bell_pair_fidelity(state):
    """Compute the fidelity of the noisy Bell pair in the graph basis.

    Parameters
    ----------
    state : ngs.State
        State of a Bell pair between the ends of the chain.

    Returns
    -------
    fidelity : int
        Fidelity, takes values [0, 1]
    """
    # Get the ket of a noiseless Bell pair in the graph basis
    noiseless_ket = gt.bell_pair_ket
    # Number of qubits in the initial state (initial Bell pairs).
    qubits = state.graph.N
    # Reduced noisy density matrix of the qubits at the end nodes (the rest have been previously measured).
    noisy_dm = ngs.noisy_bp_dm(state, [0, qubits - 1])
    return mat.H(noiseless_ket) @ noisy_dm @ noiseless_ket
