# Prompt for Codex on the Replacement VM

We migrated from the old GCP VM `cs263-t4` to a replacement GPU VM. Please continue the CS263 knowledge editing project from this repo.

Start by doing these checks:

```bash
git status --short --branch
git log --oneline --decorate --max-count=5
git lfs install
git lfs pull
git lfs ls-files --size
find data/stats/gpt2-xl/wikipedia_stats -maxdepth 1 -type f -name '*.npz' -printf '%f %s bytes\n' | sort
nvidia-smi
```

Expected MEMIT/ROME covariance cache files:

```text
transformer.h.13.mlp.c_proj_float32_mom2_100000.npz
transformer.h.14.mlp.c_proj_float32_mom2_100000.npz
transformer.h.15.mlp.c_proj_float32_mom2_100000.npz
transformer.h.16.mlp.c_proj_float32_mom2_100000.npz
transformer.h.17.mlp.c_proj_float32_mom2_100000.npz
```

Use Ubuntu 22.04 with a conda environment named `cs263-project` and Python 3.10. Do not use a system Python 3.12 environment for this project.

Setup from a fresh clone:

```bash
sudo apt-get update
sudo apt-get install -y git-lfs tmux
git lfs install
git lfs pull
git clone https://github.com/zjunlp/EasyEdit external/EasyEdit
cd external/EasyEdit
patch -p1 < ../../patches/0001-fix-nethook-pytorch29-with_kwargs-signature.patch
cd ../..
conda create -n cs263-project python=3.10 -y
conda activate cs263-project
pip install -r external/EasyEdit/requirements.txt
python scripts/check_env.py
python scripts/show_results.py --all
```

If the LFS cache is unavailable for any reason, restore the backup archive:

```bash
gcloud storage cp gs://cs263-project-494118-memit-backup/cs263-memit-preserve-20260510.tar.gz ~/
sha256sum ~/cs263-memit-preserve-20260510.tar.gz
tar -xzf ~/cs263-memit-preserve-20260510.tar.gz
```

Expected archive checksum:

```text
f15b0cd7f85bf9b597572476f083f6151358dcbfe4474e99ca097f6471b3c73b
```

Project state:

- ROME baseline is done.
- MEMIT single-edit baseline is done.
- MEMIT true batch runs for 10, 50, and 100 are done.
- IKE 5-edit baseline is done; larger IKE runs are pending.
- Probe set is written and validator-clean; ROME/MEMIT probe sweeps are the next likely GPU jobs.

Before running expensive jobs, use `tmux` and explicit logs under `logs/`.
