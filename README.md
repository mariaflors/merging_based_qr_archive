# Merging-Based Quantum Repeater

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14824315.svg)](https://doi.org/10.5281/zenodo.14824315)

This code corresponds to the scenarios and settings described in

> Merging-Based Quantum Repeater <br>
> Maria Flors Mor-Ruiz, Jorge Miguel-Ramiro, Julius Wallnöfer, Tim Coopmans, and Wolfgang Dür <br>
> Preprint: [arXiv:2502.04450 [quant-ph]](https://doi.org/10.48550/arXiv.2502.04450);


## Requirements
This repository uses numpy, matplotlib, [graphepp](https://github.com/jwallnoefer/graphepp), and [noisy_graph_states](https://github.com/jwallnoefer/noisy_graph_states). We recommend installing the exact versions in `requirements.txt`, to do so, use the following command:
```
pip install -r requirements.txt
```

## Structure of the code
There are two main folders where the code is stored, `analytical`, where there is the file that has the analytic analysis of the merging-based approach with the double distance, 
and `monte_carlo`, where there are seven files that are used to perform the Monte Carlo simulation of both the swapping and merging based approaches of the double distance protocol:

1. `swap_based.py`: Sampling function of the swapping-based approach.
2. `merge_based.py`: Sampling functions of the merging-based approach. Described in detail in Appendix B.
3. `skr_tools.py`: Functions needed to analyze the data of the sampling functions to get the secret key rate, secret key fraction, and raw rate.
4. `skr_vs_total_distance.py`: Functions to generate (with parallelization) and save the data of the secret key rate and fidelity in terms of the total distance of the repeater chain. This uses the sampling functions of the swapping and merging-based approaches.
5. `plot_skr_vs_total_distance.py`: Function to plot the secret key rate and fidelity in terms of the total distance of the repeater chain for both the merging and swapping based approaches. This uses the data generated with file `skr_vs_total_distance.py`.
6. `skr_vs_memory.py`: Functions to generate (with parallelization) and save the data of the secret key rate and fidelity in terms of the dephasing time of the quantum memories. This uses the sampling functions of the swapping and merging-based approaches.
7. `plot_skr_vs_total_distance.py`: Function to plot the ratio of improvement on the secret key rate of the merging-based over the swapping-based protocol in terms of the dephasing time of the quantum memories. This uses the data generated with file `skr_vs_memory.py`.
8. `plot_fidelity_vs_memory.py`: Function to plot the fidelity in terms of the dephasing time of the quantum memories for both the merging and swapping based approaches. This uses the data generated with file `skr_vs_memory.py`.
9. `skr_vs_p_merge.py`: Functions to generate (with parallelization) and save the data of the secret key rate in terms of the success probability of the merging operation of the merging-based approach with and without patching limitation. This uses the sampling function of the merging-based approach.
10. `plot_skr_vs_p_merge.py`: Function to plot the secret key rate in terms of the success probability of the merging operation of the merging-based approach with and without patching limitation. This uses the data generated with file `skr_vs_p_merge.py`.
11. `plot_skf_vs_fidelity.py`: Function to plot the secret key fraction in terms of the fidelity of the final Bell pair. This uses the data generated with file `skr_vs_total_distance.py`.

## Usage of the code
Here a brief instruction on how to use the functions to replicate the results of the paper is presented.

For the results presented in the main text of the paper, one must run the file `skr_vs_total_distance.py` to generate the data. 
In the `if __name__ == "__main__"` part of the code one can edit the parameters in order to make the simulation compatible with the machine that is being used. Note that for higher `k` the computational time is longer.
Then the file `plot_skr_vs_total_distance.py` can be run with the current parameters to plot the generated data (corresponding to Fig. 3).

For the additional results presented in Appendix D:
- To replicate Fig. 5, one must change the plotting parameter in file `plot_skr_vs_total_distance.py` to `"rr"` and to `"skf"` to plot the raw rate and the secret key fraction separately. Note that the data for this plot is generated together with the data of the plot of the main text.
- To replicate Fig. 6, one must run the file `skr_vs_memory.py` to generate the data. In the `if __name__ == "__main__"` part of the code one can edit the parameters in order to make the simulation compatible with the machine that is being used. Note that for higher `k` the computational time is longer.
Then the file `plot_skr_vs_memory.py` can be run with the current parameters to plot the generated data.
- To replicate Fig. 7, one must run the file `skr_vs_p_merge.py` to generate the data. In the `if __name__ == "__main__"` part of the code one can edit the parameters in order to make the simulation compatible with the machine that is being used. Note that for higher `k` the computational time is longer. Then the file `plot_skr_vs_p_merge.py` can be run with the current parameters to plot the generated data. 

For the additional results presented in Appendix E:
- To replicate Fig. 8, one must change the plotting parameter in file `plot_skr_vs_total_distance.py` to `"fid"` to plot the fidelity. Note that the data for this plot is generated together with the data of the plot of the main text.
- To replicate Fig. 9, one must run the file `skr_vs_memory.py` to generate the data. In the `if __name__ == "__main__"` part of the code one can edit the parameters in order to make the simulation compatible with the machine that is being used. Note that for higher `k` the computational time is longer.
Then the file `plot_fidelity_vs_memory.py` can be run with the current parameters to plot the generated data.
- To replicate Fig. 10, one must run `plot_skf_vs_fidelity.py` with the current parameters. Note that the data for this plot is generated together with the data of the plot of the main text.

To check the verification of the simulation using the analytic analysis, in the `if __name__ == "__main__"` part of the file `analytical/analytical_merge_based_four_segments.py` there is a code that produces a plot that verifies the behaviour of the Monte Carlo sampling.
