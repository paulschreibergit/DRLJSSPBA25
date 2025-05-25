# -----------------------------------------------------------------------------
# Autor: Paul Schreiber, Mai 2025
# Diese Datei enthält vollständig persönliche Implementierungen.
# -----------------------------------------------------------------------------
import torch

def build_action_mask(partial_plan_info, node_info) -> torch.BoolTensor:
    """
    Erzeugt eine Bool-Maske der Länge aller Operationen im Graphen,
    wobei nur Operationen True sind, die geplant werden dürfen.

    Voraussetzungen:
    - Die Operation darf noch nicht in partial_solution enthalten sein
    - Ihr direkter Vorgänger (falls vorhanden) muss bereits geplant sein
    """
    partial_solution = partial_plan_info.partial_solution
    num_nodes = len(node_info)
    mask = torch.zeros(num_nodes, dtype=torch.bool)

    for (job_id, op_id), info in node_info.items():
        node_id = info["node_id"]

        # Bereits geplant -> darf nicht gewählt werden
        if (job_id, op_id) in partial_solution:
            continue

        # Erste Op eines Jobs -> direkt planbar
        if op_id == 0:
            mask[node_id] = True
        else:
            # Prüfe Vorgänger
            predecessor = (job_id, op_id - 1)
            if predecessor in partial_solution:
                mask[node_id] = True

    return mask