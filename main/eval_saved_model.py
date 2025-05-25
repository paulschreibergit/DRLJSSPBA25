# -----------------------------------------------------------------------------
# AI-Assisted Code
# Erstellt mit Unterstützung von OpenAI ChatGPT (Modell o3, Mai 2025)
# Überarbeitet und validiert durch: Paul Schreiber
#
# Anmerkung:
# - Kernlogik und Boilerplate in dieser Datei stammen aus KI-Vorschlägen.
# -----------------------------------------------------------------------------

from utils.txt_instance_to_list import parse_jssp_file
from utils.ortools_solver import solve_jssp_with_ortools
from utils.partial_solution_controller import extract_initial_partial_solution
from environment.state import build_graph_state
from environment.mask import build_action_mask
from environment.environment import step
from archive.policy_gnn_agent import GNNPolicyNet
import torch
import random
import numpy as np

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

#  Modell laden
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "../models/best_model.pt"
policy_net = GNNPolicyNet(input_dim=2, hidden_dim=64).to(device)
policy_net.load_state_dict(torch.load(model_path, map_location=device))
policy_net.eval()

#   Neue Instanz vorbereiten
test_instance_path = "path/to/txt/instance"
jobs_data = parse_jssp_file(test_instance_path)
solver_solution, solver_makespan = solve_jssp_with_ortools(jobs_data)

#   Initiale Teillösung erzeugen
partial_info = extract_initial_partial_solution(jobs_data, solver_solution, ratio=0.01)
graph, node_info = build_graph_state(jobs_data, partial_info.partial_solution)

done = False
step_count = 0

while not done:
    action_mask = build_action_mask(jobs_data, partial_info, node_info).bool().to(device)
    graph = graph.to(device)

    with torch.no_grad():
        logits, _ = policy_net(graph.x, graph.edge_index, graph.edge_attr)
        masked_logits = logits.masked_fill(~action_mask, float("-inf"))
        probs = torch.softmax(masked_logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        action_node = torch.argmax(probs).item()

    reverse_lookup = {v["node_id"]: k for k, v in node_info.items()}
    action = reverse_lookup.get(action_node)

    if action is None:
        print(f"Ungültige Aktion bei Step {step_count}")
        break

    partial_info, graph, node_info, action_mask, done = step(jobs_data, partial_info, action, node_info)
    step_count += 1

#   Ergebnis analysieren
agent_solution = partial_info.partial_solution
agent_makespan = max(op["end"] for op in agent_solution.values())

print(f"\n Test auf Instanz: {test_instance_path}")
print(f" Agent Makespan: {agent_makespan}")
print(f" Optimal Makespan: {solver_makespan}")
print(f" Gap: {round((agent_makespan - solver_makespan)/solver_makespan * 100, 2)} %")
