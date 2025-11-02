# File: dispatching_rules.py

from enum import Enum
import numpy as np

class Rules(Enum):
    """
    Enum to define the dispatching rules. The values correspond to the action space.
    """
    SPT = 0   # Shortest Processing Time
    LPT = 1   # Longest Processing Time
    STPT = 2  # Shortest Total Processing Time (for the job)
    LTPT = 3  # Longest Total Processing Time (for the job)
    LOR = 4   # Least Operations Remaining (for the job)
    MOR = 5   # Most Operations Remaining (for the job)
    LQNO = 6  # Least Queue at Next Operation's machine
    MQNO = 7  # Most Queue at Next Operation's machine

def get_job_and_op_indices(op_id, env):
    """Helper to get job (row) and operation (col) index from an operation ID."""
    return op_id // env.number_of_machines, op_id % env.number_of_machines

def apply_dispatching_rule(rule, candidates, env):
    """
    Selects the best operation from a list of candidates based on a rule.
    
    Args:
        rule (Rules): The dispatching rule to apply.
        candidates (np.ndarray): A numpy array of eligible operation IDs.
        env (SJSSP): The environment object, to access state info.

    Returns:
        int: The ID of the selected operation.
    """
    if len(candidates) == 0:
        raise ValueError("Cannot apply a rule to an empty set of candidates.")
    
    if len(candidates) == 1:
        return candidates[0]

    # --- Tie-breaking secondary rule: Shortest Processing Time ---
    def break_tie_with_spt(tied_candidate_ids, env):
        tied_durations = env.dur.flatten()[tied_candidate_ids]
        final_choice_idx = np.argmin(tied_durations)
        return tied_candidate_ids[final_choice_idx]

    # --- Calculate Scores for the Primary Rule ---
    scores = []
    if rule in [Rules.SPT, Rules.LPT]:
        scores = env.dur.flatten()[candidates]
    elif rule in [Rules.STPT, Rules.LTPT]:
        for op_id in candidates:
            job_idx, _ = get_job_and_op_indices(op_id, env)
            scores.append(np.sum(env.dur[job_idx, :]))
    elif rule in [Rules.LOR, Rules.MOR]:
        for op_id in candidates:
            _, op_idx = get_job_and_op_indices(op_id, env)
            scores.append(env.number_of_machines - op_idx)
    elif rule in [Rules.LQNO, Rules.MQNO]:
        machine_queues = {}
        for op_id in candidates:
            job_idx, op_idx = get_job_and_op_indices(op_id, env)
            machine_needed = env.m[job_idx, op_idx]
            machine_queues[machine_needed] = machine_queues.get(machine_needed, 0) + 1
        for op_id in candidates:
            job_idx, op_idx = get_job_and_op_indices(op_id, env)
            if op_idx == env.number_of_machines - 1:
                scores.append(0)
            else:
                next_op_machine = env.m[job_idx, op_idx + 1]
                scores.append(machine_queues.get(next_op_machine, 0))
    
    scores = np.array(scores)

    # --- Find Best Candidate(s) and Handle Ties ---
    if rule in [Rules.SPT, Rules.STPT, Rules.LOR, Rules.LQNO]:
        best_score = np.min(scores)
    else: # LPT, LTPT, MOR, MQNO
        best_score = np.max(scores)

    # Find all candidates that have the best score
    tied_indices = np.where(scores == best_score)[0]

    if len(tied_indices) == 1:
        # No tie
        return candidates[tied_indices[0]]
    else:
        # There is a tie, apply secondary rule (SPT)
        tied_candidate_ids = candidates[tied_indices]
        return break_tie_with_spt(tied_candidate_ids, env)