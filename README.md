# honours_dev
Honours working programs (traffic generation, capture, analysis, and classification)

# Install with Anaconda

Install conda

Use libmamba solver (recommended)
```bash
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Allow VS Code to see conda envs
```bash
conda install -n base nb_conda_kernels
```

Create `honours_env` conda env from yaml file.
```bash
# conda env create -f honours.yaml -y
```

Use `honours_env` conda env.
```bash
conda activate honours_env
```

Update `honours_env` conda env.
```bash
conda env update -f honours.yaml --prune -y
```

Remove `honours_env` conda env.
```bash
conda env remove -n honours_env -y
```
