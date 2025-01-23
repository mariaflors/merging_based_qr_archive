"""
This script simulates the merging-based repeater protocol.
That is, it performs a Monte Carlo simulation to find the time until
the end nodes of the repeater chain are entangled with a cluster state,
when the chain starts with no entanglement whatsoever (= waiting time).
It also returns a list of the waiting time for each qubit in the final 1D
cluster that is created before achieving an end-to-end Bell pair.

More specifically, the script:
    - contains the function `wt` (waiting time) which samples from the waiting time
      distribution on a desired number of repeater segments
    - in addition, the functions `wt_and_merge_on_one_side` and `wt_and_merge_on_two_sides`
      compute the waiting time of producing fresh entanglement *and* merging the freshly
      generated entanglement with existing entanglement. The three functions call each
      other recursively
    - *Important assumption:* if there are two 'holes' in the cluster state that need
      to be patched, the protocol considers that the entanglement left in the middle is
      discarded and needs to be generated anew. So, we recover the situation where there
      is entanglement left on both sides, but now the gap is larger.
      The reason for this is ease of analysis: if we also wanted to take the two-hole
      situation into account, we would also need a function `wt_two_patches_and_merge`
      etc. The Monte Carlo sample that `wt` returns is thus an upper bound to the time
      of the protocol that does take such multi-patches into account. (One could also
      view this analysis as a first-order approximation.)
    - The waiting times of each qubit are tracked taking into account the following property.
      When two qubits are fused the operation is a CNOT between a qubit that acts like source
      and another that acts like target. Then the target qubit is measured in the Z basis.
      Since we consider time-dependent dephasing noise, using the Noisy Stabilizer Formalism,
      one can see that the noise map of the source and the target after the fusion have the
      same structure, and if we consider that both have memories of the same quality (same
      dephasing time), then this is equivalent to considering that the time that the source
      qubit is affected by is its own timing and the time of the target qubit until it has
      been measured.
      See Appendix A in paper `Merging-Based Quantum Repeater` for more details.

Note: throughout the code we will use the following assumption:

ASSUMPTION: if a merging operation of two cluster state fails, then the merged qubits are traced out.
So if a merging operation attempt is performed on two long-range cluster states
(each spanning at least > 2 segments), then a 'hole' of at least 2 segments is left.

LIMIT 1: On the second step of the protocol the patching is not attempted, if `patch_limit` is set to 4.

LIMIT 2: The patching of a grown gap is only attempted if s > 2 ** (growth + 2).
         See more details in Appendix C in paper `Merging-Based Quantum Repeater`.
"""
import numpy as np


def wt(s, p_gen, p, growth_limit=0, patch_limit=4):
    """
    Parameters
    ----------
    s : int
        Number of repeater segments.
    p_gen : float
        Success probability of generating an elementary link.
    p : float
        Success probability of merging two cluster states into one.
    growth_limit : int
        Limit to how much a gap can grow. Default set to 0.
    patch_limit : int
        Limit to at what size of the cluster can the patching be attempted. Default set to 4.

    Returns
    -------
    float : waiting time to create a cluster state over `s` segments
            using the merging-based repeater protocol.
    list : list of length s + 1 that contains the waiting times for
            each of the qubits in the final 1D cluster.
    """
    # Two edge cases:
    # - when s = 0, i.e., no entanglement needs to be generated, and no qubits wait.
    if s == 0:
        return 0, []
    # - s = 1, i.e., entanglement generation over an elementary link and the memories of the two qubits are initialized.
    if s == 1:
        # Sample from geometric distribution with bias p_gen.
        return np.random.geometric(p=p_gen), [0, 0]

    # Finding the waiting time for generating two short-range cluster states.
    # We define by `left` and `right` the distances (in number of segments) that the two short-range states span.
    left = s // 2 + s % 2
    right = s // 2
    # We split the chain in two, and we compute the waiting time of the chain and the waiting time of each qubit.
    t_left, q_left = wt(s=left, p_gen=p_gen, p=p, growth_limit=growth_limit, patch_limit=patch_limit)
    t_right, q_right = wt(s=right, p_gen=p_gen, p=p, growth_limit=growth_limit, patch_limit=patch_limit)
    # The total time is the maximum time between the two sides.
    t = max(t_left, t_right)
    # Update the times of the qubits since, the qubits from one side have to wait for the other side.
    q_left = [q + abs(t_left - t_right) for q in q_left] if t_left < t_right else q_left
    q_right = [q + abs(t_left - t_right) for q in q_right] if t_left > t_right else q_right
    # Merging the two short-range cluster state into a single long-range one, spanning all `s` segments.
    if np.random.random() < p:
        # Merging operation succeeds.
        # The merging operation is between the last qubit on the left and the first qubit on the right.
        # If the merging operation is successful the two lists are merged such that the last element of the left
        # and the first element of the right are summed into a single element in the new list.
        q = q_left[:-1] + [q_left[-1] + q_right[0]] + q_right[1:]
        return t, q
    else:
        # Merging operation fails.
        # First we limit when does the patching start.
        # NOTE: here we use LIMIT 1 above.
        if s <= patch_limit:
            retry = wt(s=left + right, p_gen=p_gen, p=p, growth_limit=growth_limit, patch_limit=patch_limit)
            return t + retry[0], retry[1]
        # If the patching is attempted we do the following:
        # First, we treat some edge cases:
        if left == 1:
            if right == 1:
                # Edge case 1: The repeater chain had length 2. In that case, there is no
                # entanglement left after a failed merging operation, so the protocol
                # is restarted.
                retry = wt(s=left + right, p_gen=p_gen, p=p, growth_limit=growth_limit,
                           patch_limit=patch_limit)
                return t + retry[0], retry[1]
            else:
                # Edge case 2: The left link spans 1 segment, the right one more than 1.
                # In this case, after the failed merging operation, there is no entanglement
                # left on the left side, so we need to re-generate the entanglement
                # on the left side and after that merge this fresh entanglement
                # with the old entanglement on the right side.
                # NOTE: here we use assumption ASSUMPTION above.
                retry = wt_and_merge_on_one_side(existing=right - 1, new=left + 1, existing_times=q_right[1:],
                                                 existing_side="right", p_gen=p_gen, p=p,
                                                 growth_limit=growth_limit, patch_limit=patch_limit)
                return t + retry[0], retry[1]

        else:
            if right == 1:
                # Edge case 3: The right link spans 1 segment, and the left one more than 1.
                # Therefore, we are in the situation as Edge case 2, but with the roles
                # of left and right exchanged.
                # NOTE: here we use assumption ASSUMPTION above.
                retry = wt_and_merge_on_one_side(existing=left - 1, new=right + 1, existing_times=q_left[:-1],
                                                 existing_side="left", p_gen=p_gen, p=p,
                                                 growth_limit=growth_limit, patch_limit=patch_limit)
                return t + retry[0], retry[1]

        # The general case: there is a hole in the middle, which needs to be refilled.
        # NOTE: here we use assumption ASSUMPTION above.
        retry = wt_and_merge_on_two_sides(existing_a=left - 1, new=2, existing_b=right - 1,
                                          existing_a_times=q_left[:-1], existing_b_times=q_right[1:], p_gen=p_gen,
                                          p=p, growth_limit=growth_limit, patch_limit=patch_limit)
        return t + retry[0], retry[1]


def wt_and_merge_on_one_side(existing, new, existing_times, existing_side, p_gen, p, growth_limit=0,
                             patch_limit=4):
    """
    Computes the waiting time for generating a link over `new` segments and merging it on one side.
    The full repeater chain has `existing + new` segments.

    For example, for `existing=3` and `new=3`, the starting situation thus looks like
    x---x---x---x   x   x   x
    (where `x` denotes a node and ---- denotes entanglement)
    and the function returns the time until the situation is
    x---x---x---x---x---x---x

    Parameters
    ----------
    existing : int
        Number of repeater segments over which entanglement already exists.
    new: int
        Number of repeater segments over which fresh entanglement should be generated.
    existing_times : list
        List of length `existing` that has the waiting times for each of the existing qubits.
    existing_side : str
        This parameter takes either "left" or "right" and indicates in which side the existing entanglement is.
    p_gen : float
        Success probability of generating an elementary link.
    p : float
        Success probability of merging two cluster states into one.
    growth_limit : int
        Limit to how much a gap can grow. Default set to 0.
    patch_limit : int
        Limit to at what size of the cluster can the patching be attempted. Default set to 4.

    Returns
    -------
    float : waiting time to create a cluster state over `new` segments
            using the merging-based repeater protocol, and merging it
            with the entanglement over `existing` segments on one side.
    """
    if existing_side != "left" and existing_side != "right":
        raise ValueError("existing_side is neither left or right")
    # Two edge cases:
    # - there is no existing entanglement, i.e., no merging needs to be done, and we need to only wait for the
    #   generation of fresh entanglement over `new` segments.
    if existing == 0:
        return wt(s=new, p_gen=p_gen, p=p, growth_limit=growth_limit, patch_limit=patch_limit)
    # - `new = 0`, i.e., there is no entanglement which needs to be generated.
    if new == 0:
        return 0, existing_times

    # The time it takes to generate the fresh entanglement.
    t, q = wt(s=new, p_gen=p_gen, p=p, growth_limit=growth_limit, patch_limit=patch_limit)
    # Sum the time to the existing qubits.
    existing_times = [et + t for et in existing_times]
    # Attempting the merging operation:
    if np.random.random() < p:
        # Merging operation succeeds.
        if existing_side == "left":
            q = existing_times[:-1] + [existing_times[-1] + q[0]] + q[1:]
        if existing_side == "right":
            q = q[:-1] + [q[-1] + existing_times[0]] + existing_times[1:]
        return t, q
    else:
        # Merging operation fails.

        # We first treat some edge cases (for explanation, see the edge cases in the function `wt` which are identical):
        if existing == 1:
            if new == 1:
                retry = wt(s=existing + new, p_gen=p_gen, p=p, growth_limit=growth_limit,
                           patch_limit=patch_limit)
                return t + retry[0], retry[1]
            else:
                new_side = "left" if existing_side == "right" else "right"
                new_times = q[:-1] if new_side == "left" else q[1:]
                retry = wt_and_merge_on_one_side(existing=new - 1, new=existing + 1, existing_times=new_times,
                                                 existing_side=new_side, p_gen=p_gen, p=p,
                                                 growth_limit=growth_limit, patch_limit=patch_limit)
                return t + retry[0], retry[1]
        else:
            if new == 1:
                existing_times = existing_times[:-1] if existing_side == "left" else existing_times[1:]
                retry = wt_and_merge_on_one_side(existing=existing - 1, new=new + 1, existing_times=existing_times,
                                                 existing_side=existing_side, p_gen=p_gen, p=p,
                                                 growth_limit=growth_limit, patch_limit=patch_limit)
                return t + retry[0], retry[1]
        # See explanation in the final case of the function `wt`, which is identical.
        if existing_side == "left":
            retry = wt_and_merge_on_two_sides(existing_a=existing - 1, new=2, existing_b=new - 1,
                                              existing_a_times=existing_times[:-1], existing_b_times=q[1:],
                                              p_gen=p_gen, p=p, growth_limit=growth_limit,
                                              patch_limit=patch_limit)
        else:
            retry = wt_and_merge_on_two_sides(existing_a=new - 1, new=2, existing_b=existing - 1,
                                              existing_a_times=q[:-1], existing_b_times=existing_times[1:],
                                              p_gen=p_gen, p=p, growth_limit=growth_limit,
                                              patch_limit=patch_limit)
        return t + retry[0], retry[1]


def wt_and_merge_on_two_sides(existing_a, new, existing_b, existing_a_times, existing_b_times, p_gen, p, growth=0,
                              growth_limit=0, patch_limit=4):
    """
    Computes the waiting time for generating a link over `new` segments and merging it on two sides.
    The full repeater chain has `existing_a + new + existing_b` segments.

    For example, for `existing_a=3`, `existing_b=4` and `new=3`, the starting situation thus looks like
    x---x---x---x   x   x   x---x---x---x---x
    (where `x` denotes a node and ---- denotes entanglement)
    and the function returns the time until the situation is
    x---x---x---x---x---x---x---x---x---x

    Parameters
    ----------
    new: int
        Number of repeater segments over which fresh entanglement should be generated.
    existing_a : int
        Number of repeater segments over which entanglement already exists on the left of `new`.
    existing_b : int
        Number of repeater segments over which entanglement already exists on the right of `new`.
    existing_a_times : list
        List of length `existing_a` that has the waiting times for each of the existing qubits on the left of `new`.
    existing_b_times : list
        List of length `existing_b` that has the waiting times for each of the existing qubits on right of `new`.
    p_gen : float
        Success probability of generating an elementary link.
    p : float
        Success probability of merging two cluster states into one.
    growth : int
        Counting number of how much a gap can grow.
    growth_limit : int
        Limit to how much a gap can grow. Default set to 0.
    patch_limit : int
        Limit to at what size of the cluster can the patching be attempted. Default set to 4.

    Returns
    -------
    float : waiting time to create a cluster state over `new` segments
            using the merging-based repeater protocol, and merging it
            with the entanglement over `existing_a` segments on one side and
            with the entanglement over `existing_b` segments on the other.
    """
    # Three edge cases:
    # - when there is no entanglement on the left, i.e., only merging on the right needs to be done.
    if existing_a == 0 and existing_b != 0:
        return wt_and_merge_on_one_side(existing=existing_b, new=new, existing_times=existing_b_times,
                                        existing_side="right", p_gen=p_gen, p=p, growth_limit=growth_limit,
                                        patch_limit=patch_limit)
    # - when there is no entanglement on the right, i.e., only merging on the left needs to be done.
    if existing_a != 0 and existing_b == 0:
        return wt_and_merge_on_one_side(existing=existing_a, new=new, existing_times=existing_a_times,
                                        existing_side="left", p_gen=p_gen, p=p, growth_limit=growth_limit,
                                        patch_limit=patch_limit)
    # - when there is no entanglement on any side, i.e., the whole state news to be created.
    if existing_a == 0 and existing_b == 0:
        return wt(s=new, p_gen=p_gen, p=p, growth_limit=growth_limit, patch_limit=patch_limit)

    if new == 0 or new == 1:
        # In combination with the functions `wt` and `wt_and_merge_on_one_side`,
        # the case that new == 0 or new == 1 should not occur.
        raise ValueError("new is 0 or 1")

    # The waiting time for producing the cluster state over `new` segments.
    t, q = wt(s=new, p_gen=p_gen, p=p, growth_limit=growth_limit, patch_limit=patch_limit)
    # Sum the time to the existing qubits
    existing_a_times = [eat + t for eat in existing_a_times]
    existing_b_times = [ebt + t for ebt in existing_b_times]
    # Merging the cluster state on two sides:
    r = np.random.random()
    if r < p ** 2:
        # Both fuses succeed.
        q = existing_a_times[:-1] + [existing_a_times[-1] + q[0]] + q[1:-1] + [
            q[-1] + existing_b_times[0]] + existing_b_times[1:]
        return t, q

    elif r < p * (1 - p) + p ** 2:
        # Left merging operation only succeeds, so there is a 'hole' on the right which needs to be patched.
        q = existing_a_times[:-1] + [existing_a_times[-1] + q[0]] + q[1:]
        if existing_b == 1:
            # If the existing link on the right only spanned 1 segment, the 'hole'
            # touches one end of the repeater chain, so fresh entanglement needs to be
            # generated and then fused on one side only.
            # NOTE: here we use assumption ASSUMPTION above.
            retry = wt_and_merge_on_one_side(existing=existing_a + new - 1, new=existing_b + 1, existing_times=q[:-1],
                                             existing_side="left", p_gen=p_gen, p=p,
                                             growth_limit=growth_limit, patch_limit=patch_limit)
            return t + retry[0], retry[1]
        else:
            # NOTE: here we use assumption ASSUMPTION above.
            retry = wt_and_merge_on_two_sides(existing_a=existing_a + new - 1, new=2, existing_b=existing_b - 1,
                                              existing_a_times=q[:-1], existing_b_times=existing_b_times[1:],
                                              p_gen=p_gen, p=p, growth_limit=growth_limit,
                                              patch_limit=patch_limit)
            return t + retry[0], retry[1]

    elif r < 2 * p * (1 - p) + p ** 2:
        # Right merging operation only succeeds. This case is identical to the case
        # `Left merging operation only succeeds` but with the role of left and right exchanged.
        q = q[:-1] + [q[-1] + existing_b_times[0]] + existing_b_times[1:]
        if existing_a == 1:
            # NOTE: here we use assumption ASSUMPTION above.
            retry = wt_and_merge_on_one_side(existing=existing_b + new - 1, new=existing_a + 1, existing_times=q[1:],
                                             existing_side="right", p_gen=p_gen, p=p,
                                             growth_limit=growth_limit, patch_limit=patch_limit)
            return t + retry[0], retry[1]
        else:
            # NOTE: here we use assumption ASSUMPTION above.
            retry = wt_and_merge_on_two_sides(existing_a=existing_a - 1, new=2, existing_b=existing_b + new - 1,
                                              existing_a_times=existing_a_times[:-1], existing_b_times=q[1:],
                                              p_gen=p_gen, p=p, growth_limit=growth_limit,
                                              patch_limit=patch_limit)
            return t + retry[0], retry[1]

    else:
        # Both fuses fail. In this case we are left with two holes, but we discard the entanglement left in the middle
        # so, we recover the situation where there is entanglement left on both sides, but now the gap is larger.
        # However, in order to limit these "growing" gaps, we use the parameter `growth` which is started at 0,
        # and increased by 1 everytime this case is reached. Then we have the parameter `growth_limit` which sets a
        # limit to this `growth`. If `growth` is larger than `growth_limit` we consider that the protocol is restarted,
        # if not we consider the patching of the larger gap.
        # Also, we consider LIMIT 2, then if the overall segment is not big enough, the protocol is also restarted.
        growth += 1
        if growth > growth_limit or existing_a + new + existing_b <= 2 ** (growth + 2):
            # Protocol restarts.
            retry = wt(s=existing_a + new + existing_b, p_gen=p_gen, p=p, growth_limit=growth_limit,
                       patch_limit=patch_limit)
            return t + retry[0], retry[1]

        else:
            # Attempt the patching of the larger gap.
            # The edge cases here are analogous to the edge cases in function `wt`, but with a larger gap.
            # So now in `new` we put what we put before + new.
            if existing_a == 1:
                if existing_b == 1:
                    retry = wt(s=existing_a + new + existing_b, p_gen=p_gen, p=p, growth_limit=growth_limit,
                               patch_limit=patch_limit)
                    return t + retry[0], retry[1]
                else:
                    retry = wt_and_merge_on_one_side(existing=existing_b - 1, new=existing_a + new + 1,
                                                     existing_times=existing_b_times[1:], existing_side="right",
                                                     p_gen=p_gen, p=p, growth_limit=growth_limit,
                                                     patch_limit=patch_limit)
                    return t + retry[0], retry[1]
            else:
                if existing_b == 1:
                    retry = wt_and_merge_on_one_side(existing=existing_a - 1, new=existing_b + new + 1,
                                                     existing_times=existing_a_times[:-1], existing_side="left",
                                                     p_gen=p_gen, p=p, growth_limit=growth_limit,
                                                     patch_limit=patch_limit)
                    return t + retry[0], retry[1]
            retry = wt_and_merge_on_two_sides(existing_a=existing_a - 1, new=new + 2, existing_b=existing_b - 1,
                                              existing_a_times=existing_a_times[:-1],
                                              existing_b_times=existing_b_times[1:], p_gen=p_gen, p=p,
                                              growth=growth, growth_limit=growth_limit, patch_limit=patch_limit)
            return t + retry[0], retry[1]

if __name__ == "__main__":
    print(wt(8, 0.4, 0.7))