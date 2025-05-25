# -----------------------------------------------------------------------------
# Autor: Paul Schreiber, Mai 2025
# Diese Datei enthält vollständig persönliche Implementierungen.
# -----------------------------------------------------------------------------
import math

def compute_max_clb(node_info):

    clbs = [
        node["clb"]
        for node in node_info.values()
        if "clb" in node and not node["is_scheduled"]
    ]
    if not clbs:
        return None

    return max(clbs)

# CLB-Reward nach Zhang et al.(2020)
def compute_step_reward(prev_node_info, new_node_info, reward_scale=100.0, clb_reference=100.0):

    prev_h = compute_max_clb(prev_node_info)
    new_h = compute_max_clb(new_node_info)
    if new_h is None:
        return 0
    delta = prev_h - new_h


    # Skalierung auf reward_scale
    normalized = min(delta / clb_reference, 1.0)
    scaled_reward = normalized * reward_scale


    return scaled_reward * 1.0

# finaler reward, mögliches shaping der step rewards
def compute_final_reward(agent_solution, solver_solution, scale=100.0):
    agent_makespan = max(op["end"] for op in agent_solution.values())
    solver_makespan = max(op["end"] for op in solver_solution.values())

    if solver_makespan == 0:
        return 0.0, 100.0

    gap = (agent_makespan - solver_makespan) / solver_makespan * 100  # in %

    # Reward-Berechnung
    reward = scale * math.exp(-gap / 20.0)

    '''
    if gap <= 5.0:
        reward = scale * math.exp(-gap / 5.0)
    else:
        reward = max(scale - gap * 4, 0.0)
    '''

    # möglicher stagnation penalty -> avoid local optima
    """
    if gap > 0 and prev_gap == gap:
        penalty = stagnation_counter * gap * 0.5
        reward -= penalty
        stagnation_counter += 1
        
    elif prev_gap != gap: stagnation_counter = 0
    """

    return reward * 1.0, gap

