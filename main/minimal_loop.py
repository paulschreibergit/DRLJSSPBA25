# -----------------------------------------------------------------------------
# AI-Assisted Code
# Erstellt mit Unterstützung von OpenAI ChatGPT (Mai 2025)
# Überarbeitet und validiert durch: Paul Schreiber
#
# Anmerkung:
# - Kernlogik und Boilerplate in dieser Datei stammen aus KI-Vorschlägen.
# -----------------------------------------------------------------------------
import torch
from torch_geometric.data import Batch

from utils.set_seed                         import set_seed
from utils.txt_instance_to_list             import parse_jssp_file
from utils.ortools_solver                   import solve_jssp_with_ortools
from utils.partial_solution_controller      import extract_partial_solution_with_target_conflicts
from environment.state                      import build_graph_state
from environment.mask                       import build_action_mask
from environment.environment                import step

from agent.agent                  import PolicyNetwork, PPOTrainer
from visualization.visualization import visualize_pyg_graph

# ───────────────────────── Config / Setup ─────────────────────────
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE     = "DRLJSSPBA25/path/to/jssp/instance.txt"
EPISODES = 600
GAMMA    = 1

gap_hist, ret_hist, ent_hist = [], [], []

set_seed(13)
jobs = parse_jssp_file(FILE)
solver_sol, solver_ms, _ = solve_jssp_with_ortools(jobs)

policy  = PolicyNetwork(node_feat_dim=2, hidden=128).to(DEVICE)
trainer = PPOTrainer(policy, gamma=GAMMA)

# ───────────────────────── Training Loop ──────────────────────────
for ep in range(1, EPISODES + 1):
    # Curriculum: Teilplan mit Konflikt‐Fenster erzeugen
    partial = extract_partial_solution_with_target_conflicts(
        jobs, solver_sol, 10, 100
    )

    graph, node_info = build_graph_state(jobs, partial.partial_solution)
    ep_return = 0.0
    done, steps = False, 0

    while not done:
        mask = build_action_mask(partial, node_info).bool().to(DEVICE)

        dist      = policy(Batch.from_data_list([graph.to(DEVICE)]), mask)
        act_idx   = dist.sample()
        log_prob  = dist.log_prob(act_idx)

        # Node-Index → (job_id, op_pos)
        rev  = {v["node_id"]: k for k, v in node_info.items()}
        action = rev[int(act_idx)]

        # Environment-Schritt
        partial, g_next, n_next, _, step_r, done = step(
            jobs, partial, action, node_info
        )
        visualize_pyg_graph(g_next, n_next)
        # Buffer füllen
        trainer.store(
            graph.cpu(),
            mask.cpu(),
            act_idx.detach(),
            log_prob.detach(),
            step_r,
            done
        )

        graph, node_info = g_next, n_next
        steps += 1
        agent_ms = max((op["end"] for op in partial.partial_solution.values()), default=0)
        gap = 100 * (agent_ms / solver_ms - 1.0)
        ep_return += step_r
        
    # PPO-Update (nach jeder Episode)
    metrics = trainer.update()
    print(f"Episode {ep:3d} | Steps {steps:2d}")
    gap_hist.append(gap) 
    ret_hist.append(ep_return)
    ent_hist.append(metrics["entropy"])

    if ep % 20 == 0:
        print(f"Ep {ep:3d}  ⌀Gap(20): {sum(gap_hist[-20:]) / min(len(gap_hist), 20):5.2f}%  "
              f"⌀Ret(20): {sum(ret_hist[-20:]) / min(len(ret_hist), 20):7.2f}  "
              f"Ent:{ent_hist[-1]:.3f}")

# ─────────────────────────  Modell sichern  ───────────────────────
torch.save(policy.state_dict(), "actor_only_ppo_ft06.pt")
