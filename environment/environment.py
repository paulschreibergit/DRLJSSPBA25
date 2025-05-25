# -----------------------------------------------------------------------------
# Autor: Paul Schreiber, Mai 2025
# Diese Datei enthält vollständig persönliche Implementierungen.
# -----------------------------------------------------------------------------

import copy
import logging as logger
from environment.state import build_graph_state
from environment.mask import build_action_mask
from environment.reward import compute_step_reward
from utils.partial_solution_controller import PartialPlanInfo
from utils.conflict_utils import count_machine_conflicts

def update_partial_solution_info(jobs_data, partial_solution_dict) -> PartialPlanInfo:
    """
    Erstellt ein aktualisiertes PartialPlanInfo-Objekt auf Basis einer bestehenden Teillösung,
    """
    total_ops = sum(len(job) for job in jobs_data)
    planned_ops = set(partial_solution_dict.keys())
    remaining_ops = total_ops - len(planned_ops)

    # Maschinenkonflikte zählen
    conflict_count = count_machine_conflicts(jobs_data, partial_solution_dict)

    return PartialPlanInfo(
        partial_solution=partial_solution_dict,
        conflict_count=conflict_count,
        remaining_ops=remaining_ops
    )

def find_earliest_gap(job_ready_time, duration, machine_schedule):
    """
    Bestimmt den frühestmöglichen Startzeitpunkt für eine Operation
    durch Lückenprüfung auf einer Maschine.
    """
    # Vor erster geplanter Op
    if machine_schedule:
        first = min(machine_schedule, key=lambda x: x["start"])
        if first["start"] - job_ready_time >= duration:
            return job_ready_time

    # Zwischen geplanten Op
    sorted_schedule = sorted(machine_schedule, key=lambda x: x["start"])
    for i in range(len(sorted_schedule) - 1):
        current_end = sorted_schedule[i]["end"]
        next_start = sorted_schedule[i + 1]["start"]
        start_candidate = max(job_ready_time, current_end)
        if next_start - start_candidate >= duration:
            return start_candidate

    # Hinter letzter Op
    if machine_schedule:
        last_end = max([op["end"] for op in machine_schedule])
        final_start = max(job_ready_time, last_end)
        return final_start
    else:
        return job_ready_time

def step(jobs_data, current_partial_info: PartialPlanInfo, action, node_info):

    # Aktuellen Teillösung kopieren
    new_partial_solution = copy.deepcopy(current_partial_info.partial_solution)

    if action not in node_info:
        raise ValueError(f"Aktion {action} ist kein gültiger Knoten im Graph.")

    op_data = node_info[action]
    machine = op_data["machine"]
    duration = op_data["duration"]

    # Frühestmöglichen Startzeitpunkt
    job_ready_time = (
        new_partial_solution[(action[0], action[1] - 1)]["end"]
        if action[1] > 0 and (action[0], action[1] - 1) in new_partial_solution
        else 0
    )

    start_time = find_earliest_gap(
        job_ready_time=job_ready_time,
        duration=duration,
        machine_schedule=[
            v for k, v in new_partial_solution.items() if v["machine"] == machine
        ]
    )

    end_time = start_time + duration

    # Add Op zur Lösung
    new_partial_solution[action] = {
        "machine": machine,
        "start": start_time,
        "end": end_time
    }

    # Neue Teilplan-Info berechnen
    new_partial_info = update_partial_solution_info(jobs_data, new_partial_solution)

    # Neuen Graph und NodeInfo erzeugen
    new_graph, new_node_info = build_graph_state(jobs_data, new_partial_info.partial_solution)

    # Berechene Step-Reward
    step_reward = compute_step_reward(node_info, new_node_info)

    # Neue Aktionsmaske
    new_action_mask = build_action_mask(new_partial_info, node_info)

    # alle ops geplant?
    done = len(new_partial_info.partial_solution) == sum(len(job) for job in jobs_data)

    return new_partial_info, new_graph, new_node_info, new_action_mask, step_reward, done



