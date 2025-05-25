# -----------------------------------------------------------------------------
# AI-Assisted Code
# Erstellt mit Unterstützung von OpenAI ChatGPT (Mai 2025)
# Überarbeitet und validiert durch: Paul Schreiber
#
# Anmerkung:
# - Kernlogik und Boilerplate in dieser Datei stammen aus KI-Vorschlägen.
# -----------------------------------------------------------------------------
from pathlib import Path

def parse_jssp_file(filepath):
    base_path = Path(__file__).resolve().parent.parent.parent
    full_path = base_path / filepath
    jobs = []
    with open(full_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        num_jobs, num_machines = map(int, lines[0].split())
        for line in lines[1:num_jobs+1]:
            parts = list(map(int, line.strip().split()))
            operations = [(parts[i], parts[i+1]) for i in range(0, len(parts), 2)]
            jobs.append(operations)
    return jobs




