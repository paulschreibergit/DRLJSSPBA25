# -----------------------------------------------------------------------------
# AI-Assisted Code
# Erstellt mit Unterstützung von OpenAI ChatGPT (Mai 2025)
# Überarbeitet und validiert durch: Paul Schreiber
#
# Anmerkung:
# - Kernlogik und Boilerplate in dieser Datei stammen aus KI-Vorschlägen.
# -----------------------------------------------------------------------------

from torch_geometric.data import Data
import torch
import logging

def compute_clb(job_id, op_id, duration, machine, partial_solution, node_info, jobs_data):
    """
    Berechnet den CLB-Wert einer Operation:
    - Für geplante Operationen: echter Endzeitpunkt (inkl. Maschinenkonflikte)
    - Für ungeplante: rekursiver CLB entlang der Jobreihenfolge, ohne Maschinenkonflikte
    """

    # falls Op schon geplant (in partial_solution)
    if (job_id, op_id) in partial_solution:
        end_time = partial_solution[(job_id, op_id)]["end"]
        return end_time

    # falls erste Op
    if op_id == 0:
        return duration

    # pred = job vorgänger Op
    pred = (job_id, op_id - 1)


    if pred in partial_solution:
        earliest_start = partial_solution[pred]["end"]
    else:
        pred_duration = jobs_data[job_id][op_id - 1][1]
        earliest_start = compute_clb(job_id, op_id - 1, pred_duration, machine, partial_solution, node_info, jobs_data)

    return earliest_start + duration

                                            # best with type A edges = true

def build_graph_state(jobs_data, solution=None, include_conflict_edges_A=True):
    if solution is None:
        solution = {}

    node_features = []
    edge_index = [[], []]
    edge_attr = []
    node_info = {}

    node_id_map = {}
    current_node_id = 0

    num_jobs = len(jobs_data)
    num_machines = max(op[0] for job in jobs_data for op in job) + 1
    max_duration = max(op[1] for job in jobs_data for op in job)
    makespan = max((v["end"] for v in solution.values()), default=0.0)

    # 1. Knoten + Feature-Vektor + Infos
    for job_id, job in enumerate(jobs_data):
        for op_id, (machine, duration) in enumerate(job):
            node_id = current_node_id
            node_id_map[(job_id, op_id)] = node_id
            current_node_id += 1

            sol = solution.get((job_id, op_id), {})
            start = sol.get("start", -1.0)
            end = sol.get("end", -1.0)
            is_scheduled = 1.0 if (job_id, op_id) in solution else 0.0

            # CLB berechnen
            clb = compute_clb(
                job_id=job_id,
                op_id=op_id,
                duration=duration,
                machine=machine,
                partial_solution=solution,
                node_info=node_info,
                jobs_data = jobs_data
            )



            # CLB normalisieren
            clb_norm = clb / 5000.0 if clb >= 0.0 else -1.0

            # Neuer Feature-Vektor [clb, is_scheduled]
            feature_vector = [clb_norm, is_scheduled]
            node_features.append(feature_vector)

            # Originalwerte für mögliche Visualisierung
            node_info[(job_id, op_id)] = {
                "job_id": job_id,
                "op_id": op_id,
                "machine": machine,
                "duration": duration,
                "start": start,
                "is_scheduled": bool(is_scheduled),
                "clb": clb,
                "node_id": node_id
            }

    # Konjunktive Kanten (Job-Reihenfolge)
    for job_id, job in enumerate(jobs_data):
        for op_id in range(len(job) - 1):
            a = node_id_map[(job_id, op_id)]
            b = node_id_map[(job_id, op_id + 1)]
            edge_index[0].append(a)
            edge_index[1].append(b)
            edge_attr.append([1, 0, 0])  # One-Hot: [Job, Maschine, Konflikt]

    # Maschinenreihenfolge-Kanten (geplante Maschinenreihenfolge)
    machine_ops = {}
    for (job_id, op_id), info in solution.items():
        machine = info["machine"]
        node = node_id_map[(job_id, op_id)]
        machine_ops.setdefault(machine, []).append((info["start"], node))

    for ops in machine_ops.values():
        ops_sorted = sorted(ops)
        for i in range(len(ops_sorted) - 1):
            a = ops_sorted[i][1]
            b = ops_sorted[i + 1][1]
            edge_index[0].append(a)
            edge_index[1].append(b)
            edge_attr.append([0, 1, 0])  # One-Hot: [Job, Maschine, Konflikt]


    # Graphobjekt
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float)
    )

    # Zusatzinfos
    data.makespan = makespan
    data.max_duration = max_duration
    data.num_jobs = num_jobs
    data.num_machines = num_machines



    return data, node_info



