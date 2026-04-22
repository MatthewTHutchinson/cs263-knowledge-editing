# CS 263 — Knowledge Editing Comparison

A comparative study of three knowledge editing methods (ROME, MEMIT, IKE) on GPT-2 XL, with a custom diagnostic probe set targeting method-specific failure modes.

*When Surgical Edits Leak: A Comparative Study of Logical Consistency and Ripple Effects Across Knowledge Editing Methods.*

## Team

- Matthew Hutchinson — mahutchinson@ucla.edu
- Corey Shen — corey0224@ucla.edu
- Nathan Wei — nathanwei@ucla.edu

## Setup

```bash
# Clone this repo
git clone <repo-url> cs263-knowledge-editing
cd cs263-knowledge-editing

# Clone dependencies into external/
git clone https://github.com/zjunlp/EasyEdit external/EasyEdit
git clone https://github.com/kmeng01/rome external/rome

# Create conda env matching EasyEdit's requirements
conda create -n editing python=3.10 -y
conda activate editing
pip install -r external/EasyEdit/requirements.txt
```

## Stack

- **Framework**: [EasyEdit](https://github.com/zjunlp/EasyEdit) (Wang et al., ACL 2024)
- **Methods**: ROME, MEMIT, IKE
- **Models**: GPT-2 XL (1.5B), optionally GPT-J (6B)
- **Benchmarks**: CounterFact, RippleEdits, MQUAKE
- **Custom**: ~50 diagnostic probes (contradiction / method-sensitivity / chain-of-thought)

## Layout

```
src/                  # library code (rome/, probes/, eval/, utils/)
scripts/              # runnable experiment scripts
configs/              # YAML hparams, copied from EasyEdit
results/              # experiment outputs (gitignored)
data/                 # raw data (gitignored)
external/             # EasyEdit and ROME clones (gitignored)
docs/                 # planning report, midterm/final reports
NOTES.md              # working log
CLAUDE.md             # context for Claude Code sessions
```

## Running baselines

```bash
python scripts/run_baseline.py --method rome --model gpt2-xl --n 100
python scripts/run_baseline.py --method memit --model gpt2-xl --n 100
python scripts/run_baseline.py --method ike --model gpt2-xl --n 100
```

(Script to be implemented in Week 5.)

## Status

Week 4: planning report submitted. Environment setup in progress.
