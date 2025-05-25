# -----------------------------------------------------------------------------
# AI-Assisted Code
# Erstellt mit Unterstützung von OpenAI ChatGPT (Mai 2025)
# Überarbeitet und validiert durch: Paul Schreiber
#
# Anmerkung:
# - Kernlogik und Boilerplate in dieser Datei stammen aus KI-Vorschlägen.
# -----------------------------------------------------------------------------
from utils.conflict_utils import count_remaining_conflict_combinations, compute_max_possible_conflicts
from dataclasses import dataclass
from math import ceil
from random import seed
import utils

@dataclass
class PartialPlanInfo:
    partial_solution: dict
    conflict_count: int
    remaining_ops: int

def extract_initial_partial_solution(jobs_data, full_solution, ratio: float):
    """
    Erstellt eine konsistente Teillösung aus der vollständigen Lösung.
    Fügt Operationen iterativ hinzu – nur wenn Vorgänger bereits enthalten ist.
    """
    print(count_remaining_conflict_combinations(jobs_data, full_solution))
    total_ops = sum(len(job) for job in jobs_data)
    target = ceil(ratio * total_ops)

    partial_solution = {}
    added = set()
    candidates = set((j, 0) for j in range(len(jobs_data)))

    def is_valid(op):
        j, o = op
        return o == 0 or (j, o - 1) in added

    while len(partial_solution) < target and candidates:
        valid_candidates = [op for op in candidates if is_valid(op)]
        if not valid_candidates:
            break

        def op_key(op):
            info = full_solution.get(op, {})
            start = info.get('start', float('inf'))
            duration = info.get('end', 0) - start
            return (start, duration)

        next_op = min(valid_candidates, key=op_key)
        candidates.remove(next_op)
        added.add(next_op)
        partial_solution[next_op] = full_solution[next_op]

        j, o = next_op
        if o + 1 < len(jobs_data[j]):
            candidates.add((j, o + 1))

    remaining = total_ops - len(partial_solution)
    conflict_count = utils.conflict_utils.count_machine_conflicts(jobs_data, partial_solution)

    return PartialPlanInfo(
        partial_solution=partial_solution,
        conflict_count=conflict_count,
        remaining_ops=remaining
    )

def extract_partial_solution_with_target_conflicts(jobs_data, full_solution, min_conflicts: int, max_conflicts: int,
    check_interval: int = 1,
    max_ops: int = None,
    seed_value: int = None):

    """
    Baut eine Teillösung durch sukzessives Einplanen, bis die Maschinenkonflikte im Zielbereich liegen.
    Konflikte werden alle `check_interval` Schritte gezählt.
    """

    if seed_value is not None:
        seed(seed_value)
    max_pos_conf = compute_max_possible_conflicts(jobs_data)
    if min_conflicts <= max_pos_conf <= max_conflicts:
        return PartialPlanInfo(
            partial_solution={},
            conflict_count=max_pos_conf,
            remaining_ops=sum(len(job) for job in jobs_data)
        )

    total_ops = sum(len(job) for job in jobs_data)
    max_ops = max_ops or total_ops

    partial_solution = {}
    added = set()
    candidates = set((j, 0) for j in range(len(jobs_data)))

    def is_valid(op):
        j, o = op
        return o == 0 or (j, o - 1) in added

    steps = 0
    while len(partial_solution) < max_ops and candidates:
        valid_candidates = [op for op in candidates if is_valid(op)]
        if not valid_candidates:
            break

        def op_key(op):
            info = full_solution.get(op, {})
            start = info.get('start', float('inf'))
            duration = info.get('end', 0) - start
            return (start, duration)

        next_op = min(valid_candidates, key=op_key)
        candidates.remove(next_op)
        added.add(next_op)
        partial_solution[next_op] = full_solution[next_op]

        j, o = next_op
        if o + 1 < len(jobs_data[j]):
            candidates.add((j, o + 1))

        steps += 1
        if steps % check_interval == 0 or len(partial_solution) == max_ops:
            conflicts = count_remaining_conflict_combinations(jobs_data, partial_solution)
            if min_conflicts <= conflicts <= max_conflicts:
                remaining = total_ops - len(partial_solution)
                return PartialPlanInfo(
                    partial_solution=partial_solution,
                    conflict_count=conflicts,
                    remaining_ops=remaining
                )

    print(f"⚠️ Keine Teillösung mit Konflikten in Zielbereich [{min_conflicts}–{max_conflicts}] gefunden.")
    return None