# -----------------------------------------------------------------------------
# AI-Assisted Code
# Erstellt mit Unterstützung von OpenAI ChatGPT (Mai 2025)
# Überarbeitet und validiert durch: Paul Schreiber
#
# Anmerkung:
# - Kernlogik und Boilerplate in dieser Datei stammen aus KI-Vorschlägen.
# -----------------------------------------------------------------------------
import random
import numpy as np
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def visualize_jssp_graph(data: Data, node_info: dict, title="JSSP-Graph mit Maschinenfarben"):
    G = nx.DiGraph()
    edge_index = data.edge_index
    edge_attr = data.edge_attr.tolist()
    num_machines = data.num_machines if hasattr(data, 'num_machines') else 10

    #   Farbpalette für Maschinen
    cmap = cm.get_cmap('tab20', num_machines)
    machine_color_map = {
        m: mcolors.to_hex(cmap(m)) for m in range(num_machines)
    }

    #   Position + Label pro Knoten
    pos = {}
    for (job_id, op_id), info in node_info.items():
        node_id = info["node_id"]
        machine = info.get("machine", "?")
        duration = info.get("duration", "?")
        start = info.get("start", "?")
        label = f"J{job_id}-O{op_id}\nM{machine} D{duration} T{start}"

        G.add_node(node_id, label=label)
        pos[node_id] = (op_id * 2, -job_id * 2)

    #   Kanten mit Farben je nach Typ und Maschine
    for i, (src, tgt) in enumerate(zip(edge_index[0], edge_index[1])):
        edge_type = int(edge_attr[i][0])
        fixed = bool(edge_attr[i][1])

        if edge_type == 0:
            style = "solid"
            color = "black"
        else:
            #   Disjunktive Kante → Maschine herausfinden
            machine = None
            for (j, o), info in node_info.items():
                if info["node_id"] == src.item():
                    machine = info.get("machine")
                    break
            color = machine_color_map.get(machine, "blue")
            style = "solid" if fixed else "dashed"

        G.add_edge(src.item(), tgt.item(), style=style, color=color)

    labels = nx.get_node_attributes(G, 'label')
    edge_styles = [G[u][v]['style'] for u, v in G.edges()]

    plt.figure(figsize=(16, 10), dpi=300)

    for style in set(edge_styles):
        edgelist = [(u, v) for u, v in G.edges() if G[u][v]['style'] == style]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edgelist,
            edge_color=[G[u][v]['color'] for u, v in edgelist],
            style=style,
            arrows=True,
            arrowsize=20,
            connectionstyle='arc3,rad=0.1'
        )

    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=800)
    nx.draw_networkx_labels(G, pos, labels, font_size=9)

    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_pyg_graph(graph: Data, node_info: dict, title="JSSP-Graph (aus PyG-Objekt)"):
    """
    Visualisiert den Graphen aus Sicht des Agenten mit CLB und is_scheduled.
    Horizontale Anordnung nach op_id, vertikal nach job_id.
    """
    G = nx.DiGraph()
    edge_index = graph.edge_index
    edge_attr = graph.edge_attr.tolist()
    num_machines = graph.num_machines if hasattr(graph, 'num_machines') else 10

    #   Farben für Maschinen
    cmap = cm.get_cmap('tab20', num_machines)
    machine_color_map = {m: mcolors.to_hex(cmap(m)) for m in range(num_machines)}

    pos = {}
    for (job_id, op_id), info in node_info.items():
        node_id = info["node_id"]
        clb = info.get("clb", -1)
        is_scheduled = int(info.get("is_scheduled", False))
        duration = info.get("duration", -1)

        label = f"J{job_id}-O{op_id}\nCLB:{int(clb)}\nS:{is_scheduled},D:{duration}"
        pos[node_id] = (op_id * 2, -job_id * 2)
        G.add_node(node_id, label=label)

    #   Kanten (unterscheide 3 Typen via One-Hot)
    for i, (src, tgt) in enumerate(zip(edge_index[0], edge_index[1])):
        edge_type = edge_attr[i]  #
        if edge_type == [1, 0, 0]:  # Job-Reihenfolge
            color, style = "black", "solid"
        elif edge_type == [0, 1, 0]:  # Maschinenreihenfolge
            machine = next((info["machine"] for info in node_info.values() if info["node_id"] == src.item()), None)
            color = machine_color_map.get(machine, "blue")
            style = "solid"
        elif edge_type == [0, 0, 1]:
            color, style = "red", "dashed"
        else:
            color, style = "gray", "dotted"

        G.add_edge(src.item(), tgt.item(), color=color, style=style)

    # Zeichnen
    labels = nx.get_node_attributes(G, 'label')
    edge_styles = [G[u][v]['style'] for u, v in G.edges()]

    plt.figure(figsize=(16, 10), dpi=500)
    for style in set(edge_styles):
        edgelist = [(u, v) for u, v in G.edges() if G[u][v]['style'] == style]
        is_conflict = style == "dashed"

        nx.draw_networkx_edges(
            G, pos,
            edgelist=edgelist,
            edge_color=[G[u][v]['color'] for u, v in edgelist],
            style=style,
            arrows=not is_conflict,
            arrowsize=20 if not is_conflict else 0,
            connectionstyle='arc3,rad=0.1'
        )

    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=800)
    nx.draw_networkx_labels(G, pos, labels, font_size=9)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def print_progress_bar(current, total, length=40):
    percent = current / total
    filled = int(length * percent)
    bar = "█" * filled + "-" * (length - filled)
    print(f"\rProgress: |{bar}| {percent * 100:5.1f}% ({current}/{total})", end="")

def plot_gantt(partial_solution, title="Gantt Chart", save_path=None):
    # Mapping: Maschine -> Liste von (start, duration, JobID)
    machine_to_ops = {}
    for (job_id, op_id), op_data in partial_solution.items():
        m = op_data["machine"]
        s = op_data["start"]
        d = op_data["end"] - s
        machine_to_ops.setdefault(m, []).append((s, d, job_id, op_id))

    fig, ax = plt.subplots(figsize=(10, 6))
    yticks = []
    yticklabels = []

    colors = {}  # Farben pro Job
    for machine, ops in sorted(machine_to_ops.items()):
        bar_segments = []
        for start, duration, job_id, op_id in sorted(ops):
            bar_segments.append((start, duration))
            if job_id not in colors:
                colors[job_id] = (random.random(), random.random(), random.random())
            ax.broken_barh([(start, duration)], (10 * machine, 9),
                           facecolors=colors[job_id])
            ax.text(start + duration / 2, 10 * machine + 4.5,
                    f"J{job_id}O{op_id}", ha='center', va='center', fontsize=8)

        yticks.append(10 * machine + 4.5)
        yticklabels.append(f"Machine {machine}")

    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_title(title)
    ax.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def log_training_stats(episode, total_reward, step_reward_sum, final_reward,
                       agent_makespan, solver_makespan, gap, best_final_reward,
                       policy_loss=None, value_loss=None, entropy=None):
    print("╭──────────────────────────────╮")
    print(f"│ Episode {episode + 1}")
    print("├──────────────────────────────┤")
    print(f"│ Total Reward     : {total_reward:.2f}")
    print(f"│ Step Reward Sum  : {step_reward_sum:.2f}")
    print(f"│ Final Reward     : {final_reward:.2f}")
    print(f"│ Agent Makespan   : {agent_makespan}")
    print(f"│ Solver Makespan  : {solver_makespan}")
    print(f"│ Relative Gap     : {gap:.2f}%")
    print(f"│ Best FinalReward : {best_final_reward:.2f}")
    if policy_loss is not None:
        print(f"│ Policy Loss      : {policy_loss:.4f}")
    if value_loss is not None:
        print(f"│ Value Loss       : {value_loss:.4f}")
    if entropy is not None:
        print(f"│ Entropy          : {entropy:.4f}")
    print("╰──────────────────────────────╯")

def plot_reward_composition(step_rewards_all, final_rewards_all, window=50):
    import matplotlib.pyplot as plt
    """
    Visualisiert die Reward-Zusammensetzung über alle Episoden.
    Zeigt: Step Reward (CLB) + Final Reward (Gap) im Vergleich.
    """

    episodes = list(range(len(step_rewards_all)))

    # Stackplot – rohe Rewards
    plt.figure(figsize=(12, 6))
    plt.stackplot(
        episodes,
        step_rewards_all,
        final_rewards_all,
        labels=['CLB Step Reward', 'Final Gap Reward'],
        colors=['lightblue', 'salmon'],
        alpha=0.6
    )
    plt.legend(loc='upper left')
    plt.title("Reward-Zusammensetzung pro Episode")
    plt.xlabel("Episode")
    plt.ylabel("Gesamt-Reward (pro Episode)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Moving Averages (optional)
    if len(step_rewards_all) >= window:
        import pandas as pd
        df = pd.DataFrame({
            "step": step_rewards_all,
            "final": final_rewards_all
        })
        df["step_avg"] = df["step"].rolling(window=window).mean()
        df["final_avg"] = df["final"].rolling(window=window).mean()

        plt.figure(figsize=(12, 5))
        plt.plot(df["step_avg"], label="CLB Step Reward (avg)", color="blue")
        plt.plot(df["final_avg"], label="Final Reward (avg)", color="red")
        plt.title(f"Moving Average der Belohnungskomponenten (window={window})")
        plt.xlabel("Episode")
        plt.ylabel("Reward (gemittelt)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_training_progress(all_rewards, all_makespans, stats, solver_makespan, window_size=100):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    if len(all_rewards) < window_size or len(all_makespans) < window_size:
        print("⚠️ Nicht genug Episoden für Moving Average – Plot wird übersprungen.")
        return

    # Makespan (linke Y-Achse)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Agent Makespan", color='tab:green')
    ax1.plot(all_makespans, alpha=0.3, color='tab:green', label="Raw Makespan")
    ax1.plot(range(window_size - 1, len(all_makespans)), stats["moving_avg_makespan"],
             color='tab:green', linewidth=2, label="Moving Avg Makespan")
    ax1.tick_params(axis='y', labelcolor='tab:green')
    ax1.axhline(y=solver_makespan, color='black', linestyle='--', linewidth=1, label='Solver Makespan')

    # Reward (rechte Y-Achse)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Total Reward", color='tab:red')
    ax2.plot(all_rewards, alpha=0.2, color='tab:red', label="Raw Reward")
    ax2.plot(range(window_size - 1, len(all_rewards)), stats["moving_avg_rewards"],
             color='tab:red', linewidth=2, label="Moving Avg Reward")
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Legende + Infobox
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    info_text = '\n'.join([
        f"Episodes: {stats['num_episodes']}",
        f"Avg Makespan: {stats['avg_makespan']:.2f}",
        f"Best Makespan: {stats['best_makespan']:.2f}",
        f"Best Gap: {stats['best_gap']:.2f}%",
        f"Last Makespan: {stats['last_makespan']:.2f}",
        f"Last Gap: {stats['last_gap']:.2f}%"
    ])
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.text(0.9, 0.4, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='center', horizontalalignment='center', bbox=props)

    plt.title("Agent Performance (Reward & Makespan)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#   Statistiken berechnen
def compute_training_stats(all_rewards, all_makespans, solver_makespan, window_size=100):
    moving_avg_rewards = np.convolve(all_rewards, np.ones(window_size) / window_size, mode='valid')
    moving_avg_makespan = np.convolve(all_makespans, np.ones(window_size) / window_size, mode='valid')

    avg_makespan = np.mean(all_makespans)
    best_makespan = np.min(all_makespans)
    best_gap = round((best_makespan - solver_makespan) / solver_makespan * 100.0, 2)
    last_makespan = all_makespans[-1]
    last_gap = round((last_makespan - solver_makespan) / solver_makespan * 100.0, 2)
    num_episodes = len(all_makespans)

    return {
        "moving_avg_rewards": moving_avg_rewards,
        "moving_avg_makespan": moving_avg_makespan,
        "avg_makespan": avg_makespan,
        "best_makespan": best_makespan,
        "best_gap": best_gap,
        "last_makespan": last_makespan,
        "last_gap": last_gap,
        "num_episodes": num_episodes
    }

def plot_entropy(entropy_per_episode):
    plt.figure()
    plt.plot(entropy_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.title("Entropy over time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
