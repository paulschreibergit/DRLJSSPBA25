# -----------------------------------------------------------------------------
# AI-Assisted Code
# Erstellt mit Unterstützung von OpenAI ChatGPT (Mai 2025)
# Überarbeitet und validiert durch: Paul Schreiber
#
# Anmerkung:
# - Kernlogik und Boilerplate in dieser Datei stammen aus KI-Vorschlägen.
# -----------------------------------------------------------------------------
from pathlib import Path
import json
import time
from ortools.sat.python import cp_model
from tqdm import tqdm

from utils.txt_instance_to_list import parse_jssp_file
from utils.ortools_solver import solve_jssp_with_ortools


project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "data"
instance_dir = data_dir / "jsp-instances"
output_dir = data_dir / "solved"
output_dir.mkdir(parents=True, exist_ok=True)


timelimit_per_task = 600.0
max_total_runtime = 30 * 60
start_time = time.time()
max_instances = 50
subfolders = ["abz", "ft", "la", "lta", "orb", "swv", "ta", "yn"]  # ggf. erweitern
solved_summary = []


all_txt_files = []
for sub in subfolders:
    folder = instance_dir / sub
    if folder.exists():
        all_txt_files.extend(sorted(folder.glob("*.txt")))

solved_count = 0

for txt_file in tqdm(all_txt_files, desc="🔍 Löse Instanzen"):
    elapsed_time = time.time() - start_time
    if solved_count >= max_instances or elapsed_time > max_total_runtime:
        break
    name = txt_file.stem
    rel_path = txt_file.relative_to(instance_dir)
    output_path = output_dir / f"{name}.json"

    if output_path.exists():
        continue

    jobs_data = parse_jssp_file(txt_file)
    solution, makespan, status = solve_jssp_with_ortools(jobs_data, timelimit_per_task)

    if status == cp_model.OPTIMAL:
        status_label = "optimal"
    else:
        status_label = "feasible"
        print(f"Keine Lösung für {name}")
        continue



    # Speichern als Einzeldatei
    # 🔁 Lösungsschlüssel (Tuple) in Strings konvertieren
    solution_str_keys = {
        f"{j}-{t}": data for (j, t), data in solution.items()
    }
    result = {
        "name": name,
        "path": str(rel_path),
        "makespan": makespan,
        "status": status_label,
        "solution": solution_str_keys
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    solved_count += 1

    # Für zentrale Übersicht merken
    solved_summary.append({
        "name": name,
        "path": str(rel_path),
        "makespan": makespan
    })

# Inkrementelles Update der zentralen classical.json
summary_path = output_dir / "classical.json"

# Vorherige Daten laden, wenn vorhanden
if summary_path.exists():
    with open(summary_path, "r") as f:
        existing_data = json.load(f)
else:
    existing_data = []

# Duplikate vermeiden
existing_names = {entry["name"] for entry in existing_data}
filtered_new_entries = [entry for entry in solved_summary if entry["name"] not in existing_names]

# Kombinieren und speichern
if filtered_new_entries:
    updated_data = existing_data + filtered_new_entries
    with open(summary_path, "w") as f:
        json.dump(updated_data, f, indent=2)
    print(f"\n✅ classical.json aktualisiert: {len(filtered_new_entries)} neue Instanz(en) hinzugefügt.")
else:
    print("\nℹ️ Keine neuen Instanzen – classical.json bleibt unverändert.")

print(f"✅ Fertig! {len(solved_summary)} Instanzen gelöst.")