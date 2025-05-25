#Perron, L., & Didier, F. (2024). OR-Tools CP-SAT (Version 9.11) [Software].
#Google. https://developers.google.com/optimization/cp/cp_solver
# -----------------------------------------------------------------------------
# AI-Assisted Code
# Erstellt mit Unterstützung von OpenAI ChatGPT (Mai 2025)
# Überarbeitet und validiert durch: Paul Schreiber
#
# Anmerkung:
# - Kernlogik und Boilerplate in dieser Datei stammen aus KI-Vorschlägen.
# -----------------------------------------------------------------------------
from ortools.sat.python import cp_model

def solve_jssp_with_ortools(jobs_data, timelimit = 60.0):
    model = cp_model.CpModel()

    num_jobs = len(jobs_data)
    num_machines = len(jobs_data[0])

    all_tasks = {}
    all_machines = [[] for _ in range(num_machines)]

    horizon = sum(task[1] for job in jobs_data for task in job)

    # Variablen definieren
    for job_id, job in enumerate(jobs_data):
        for task_id, (machine, duration) in enumerate(job):
            suffix = f'_j{job_id}_t{task_id}'
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval' + suffix)
            all_tasks[(job_id, task_id)] = (start_var, end_var, interval_var, machine)
            all_machines[machine].append(interval_var)

    # Maschinenbelegungs-Constraints
    for machine_intervals in all_machines:
        model.AddNoOverlap(machine_intervals)

    # Job-Reihenfolge
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            _, end_var, _, _ = all_tasks[(job_id, task_id)]
            start_var, _, _, _ = all_tasks[(job_id, task_id + 1)]
            model.Add(start_var >= end_var)

    # Ziel: Minimieren des Makespan
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [all_tasks[(j, len(job)-1)][1] for j, job in enumerate(jobs_data)])
    model.Minimize(obj_var)

    # Lösen
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timelimit
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        solution = {}
        for (j, t), (start, end, _, m) in all_tasks.items():
            solution[(j, t)] = {
                'machine': m,
                'start': solver.Value(start),
                'end': solver.Value(end)
            }
        makespan = solver.ObjectiveValue()
        return solution, makespan, status
    else:
        print("Keine Lösung gefunden.")
        return None, None, status
