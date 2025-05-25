# -----------------------------------------------------------------------------
# AI-Assisted Code
# Erstellt mit Unterstützung von OpenAI ChatGPT (Mai 2025)
# Überarbeitet und validiert durch: Paul Schreiber
#
# Anmerkung:
# - Kernlogik und Boilerplate in dieser Datei stammen aus KI-Vorschlägen.
# -----------------------------------------------------------------------------
from collections import defaultdict
from math import comb



def count_machine_conflicts(jobs_data, partial_solution) -> int:

    machine_to_unplanned_ops = defaultdict(list)

    for j, job in enumerate(jobs_data):
        for o, (machine, _) in enumerate(job):
            if (j, o) not in partial_solution:
                machine_to_unplanned_ops[machine].append((j, o))

    conflict_count = sum(1 for ops in machine_to_unplanned_ops.values() if len(ops) >= 2)
    return conflict_count

def count_remaining_conflict_combinations(jobs_data, partial_solution) -> int:
    """
    Zählt die Gesamtanzahl an Konfliktkombinationen über alle Maschinen.
    Für jede Maschine mit k ungeplanten Operationen wird k über 2 gezählt.
    """
    machine_to_unplanned = defaultdict(list)

    for j, job in enumerate(jobs_data):
        for o, (machine, _) in enumerate(job):
            if (j, o) not in partial_solution:
                machine_to_unplanned[machine].append((j, o))

    total = sum(comb(len(ops), 2) for ops in machine_to_unplanned.values() if len(ops) >= 2)
    return total

def compute_max_possible_conflicts(jobs_data):

    machine_to_ops = defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for op_id, (machine, _) in enumerate(job):
            machine_to_ops[machine].append((job_id, op_id))

    max_conflicts = 0
    for machine, ops in machine_to_ops.items():
        n = len(ops)
        if n >= 2:
            max_conflicts += n * (n - 1) // 2  # K_n über 2

    return max_conflicts