# DRLJSSPBA25

## Einsatz von KI-Unterstützung

Ein Teil dieses Repositories wurde mithilfe von OpenAI ChatGPT (Stand Mai 2025) generiert und refaktoriert. Die KI diente als Werkzeug zur Formulierung, Refaktorierung
und strukturellen Ergänzung einzelner Funktionen, insbesondere im Zusammenhang mit:

- GNN-Policy-Netzwerkstrukturen
- PPO-Trainingslogik
- Debugging-Tools und Visualisierungshilfen

**Nachhaltige Prüfung und Überarbeitung durch den Autor**  
Alle von ChatGPT gelieferten Code-Fragmente wurden anschließend manuell geprüft, auf Korrektheit hin validiert und in Teilen eigenständig überarbeitet. Sämtliche eigenständigen Änderungen – z. B. die finale Ausgestaltung der Belohnungsfunktion, Optimierungen der Policy-Netz-Definition oder Anpassungen des Training-Loops – stehen unter persönlicher Urheberschaft des Autors.

**Konzeption und methodische Verantwortung**
Die **konzeptionelle Struktur** des Projekts – insbesondere die Idee eines
Curriculum-Learnings auf Basis maschineller Konflikte, die schrittweise Planung
teilgelöster JSSP-Instanzen sowie die Auswahl und Kombination relevanter
Literaturansätze (z. B. CLB-Mechanismen, GNN-PPO-Kombination) – ist eigenständig
vom Autor entwickelt, geplant und in dieser Form selbständig umgesetzt worden.

**Kennzeichnung im Code**  
- KI-gestützter Code ist in den jeweiligen Dateien durch einen Header gekennzeichnet.  
- Funktionen, die vollständig eigenständig entwickelt oder substanziell überarbeitet wurden, enthalten keine solche Markierung und gelten als persönliche Leistung.

**Rechtlicher Hinweis**  
Der Einsatz von ChatGPT erfolgte im Einklang mit den Nutzungsbedingungen von OpenAI.  
Die finale Verantwortung für Funktionalität, Korrektheit und Sicherheit des Codes liegt beim Autor.

---
```
DRLJSSPBA25/
│
├── agent/                 # GNN-Policy & Trainingslogik
├── environment/           # Zustandsrepräsentation, Reward, Maskierung
├── main/                  # Ausführungsskripte
├── utils/                 # Solver, Parser, Hilfsfunktionen
├── visualization/         # Graphbasierte Visualisierung
├── requirements.txt
```
---
# Quickstart: DRL-Agent für Job-Shop Scheduling

Dieses Repository enthält einen Deep-Reinforcement-Learning-Ansatz zur schrittweisen Lösung von Job-Shop-Scheduling-Problemen (JSSP). Im Folgenden findest du eine  Anleitung zur Ausführung des Codes.

---

### 1. Repository klonen und initialisieren

```bash
git clone https://github.com/paulschreibergit/DRLJSSPBA25.git
cd DRLJSSPBA25
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
> **Hinweis:**  
> - Die Trainingslogik befindet sich im Skript `main/minimal_loop.py` und dient als Ausgangspunkt für alle Abläufe.  
> - Das Repository arbeitet mit klassischen JSSP-Benchmark-Instanzen im OR-Library-Format (z. B. `ft06.txt`).
>   Diese Textdateien bestehen aus einer Kopfzeile mit der Anzahl an Jobs und Maschinen, gefolgt von zeilenweisen Job-Operationen im Format:  
>   ```
>   6 6
>   1 6 3 7 5 3 4 6 ...
>   ```
> - Die Konvertierung erfolgt **automatisch beim Start** des Trainings: Der integrierte Parser (`txt_instance_to_list.py`) liest die `.txt`-Datei ein und wandelt sie intern in eine Listenstruktur um.  
> - Es musst lediglich der Pfad zur gewünschten `.txt`-Instanz korrekt übergeben werden.
---
