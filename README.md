# fleetlab

A Python framework for **simulation-based optimisation of pickup-and-delivery
problems** — routing a fleet of vehicles that collect things in one place and
deliver them in another.

It is built so that different algorithms can be run against the same problem and
compared fairly. Heuristics, metaheuristics and mathematical programming all
read the same problem definition, so the numbers they produce mean the same
thing.

What gets moved may be **people or goods**, or both on the same vehicle. The
framework does not care which.

---

## Install

You need **Python 3.11 or newer**.

The steps below are for Windows. macOS and Linux are the same commands with
forward slashes; the one difference is noted at the end.

### 1. Install `uv`

`uv` is the package manager this project uses. Open **PowerShell** and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or, if you use winget:

```powershell
winget install --id=astral-sh.uv -e
```

**Close and reopen your terminal afterwards** so that `uv` is on your PATH. Check
it worked:

```powershell
uv --version
```

### 2. Set up the project

From the project folder — in PyCharm, open the terminal with `Alt+F12`:

```powershell
uv sync --all-extras --dev
```

That creates a virtual environment in `.venv` and installs everything, including
the development tools. It takes a few seconds.

### 3. Check it works

```powershell
uv run pytest
```

You should see `113 passed`. If you do, you are set up correctly.

Then run the worked example, which solves a small problem three different ways
and prints the results:

```powershell
uv run python examples/run_study.py
```

### Running commands later

`uv run <command>` uses the project environment without you having to activate
anything. That is the simplest way to work:

```powershell
uv run pytest
uv run ruff check .
uv run mypy
```

If you would rather activate the environment in your shell:

```powershell
.venv\Scripts\Activate.ps1        # PowerShell
.venv\Scripts\activate.bat        # Command Prompt
```

If PowerShell refuses to run the activation script, allow it for the current
window only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

> **macOS and Linux:** install `uv` with
> `curl -LsSf https://astral.sh/uv/install.sh | sh`, and activate with
> `source .venv/bin/activate`. Everything else is identical.

### Setting up PyCharm

1. `File → Settings → Project → Python Interpreter`
2. `Add Interpreter → Add Local Interpreter → Existing`
3. Choose `.venv\Scripts\python.exe` inside the project folder
4. Right-click the `src` folder → `Mark Directory as → Sources Root`

Step 4 matters: without it, PyCharm underlines every `fleetlab` import in red
even though the code runs fine.

### Optional: a solver

The mathematical-programming side writes standard `.lp` files that any solver
can read, so **no solver is required**. If you want to solve inside Python
instead, `uv sync --all-extras` has already installed one (CBC, via PuLP).

---

## A first taste

```python
from fleetlab.study import Study
from fleetlab.io.generate import mixed_instance
from fleetlab.search import RegretInsertion, AdaptiveLNS

# A problem with 8 passengers, 2 wheelchair users, 6 parcels and 3 vehicles.
problem = mixed_instance(passengers=8, wheelchair_users=2, parcels=6, vehicles=3)

# A Study is the problem plus the rules plus what counts as "better".
study = Study(problem)

# Build a first solution.
built = RegretInsertion().solve(study)
print(built.describe())

# Improve it.
improved = AdaptiveLNS(iterations=600).solve(study, built.solution)
print(improved.breakdown.describe())
```

`breakdown.describe()` prints the cost broken down by term — distance, detour
time, vehicles used, requests left unserved — rather than a single number, so
you can see *why* one result beat another.

---

## What is in the box

```
fleetlab/
├── domain/        the vocabulary: Stop, Loadable, Request, Vehicle,
│                  Schedule, Solution, Problem
├── od/            travel-time and distance matrices
├── timing/        works out when a vehicle arrives, waits, serves and leaves
├── feasibility/   the rules, and what it means to break one
├── objective/     what counts as a better solution
├── moves/         ways to change a solution: insert, remove, relocate
├── search/        heuristics and metaheuristics
├── mathprog/      builds a mathematical model of the same problem
├── simulation/    runs a day forward in time, re-planning as it goes
├── io/            reading, writing and generating problem instances
└── study.py       Study — the one object you will use most
```

---

## Where to go next

| if you want to | read |
|---|---|
| understand how it all works, from scratch | **[docs/TUTORIAL.md](docs/TUTORIAL.md)** |
| give an AI assistant enough context to help you | **[AGENTS.md](AGENTS.md)** |
| see it run | `examples/run_study.py` |

The tutorial is written for someone who has never seen this code. It takes about
twenty minutes and ends with you adding a rule of your own.

---

## Status

Alpha. The core runs end to end and is covered by 113 tests, but the algorithms
that ship are a starting point to measure against, not finished competitors —
writing better ones is the point of the framework.
