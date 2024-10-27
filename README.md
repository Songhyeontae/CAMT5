# CAMT5
Efficient Tokenization for Molecular Language Models

## Getting Started
- virtual enviroment (whatever you want. PyEnv, Conda or Pyenv)
- dependency
    - use uv: `pip install uv`
    - `uv pip install -r requirements.txt -r requirements-dev.txt`

## Project Sturcture
```
CAMT5/
├── .gitignore
├── .pre-commit-config.yaml
├── asset/
├── config/
│   ├── config.yaml
│   └── task/
├── core/
├── main.py
├── metrics/
├── model/
├── README.md
├── requirements-dev.txt
├── requirements.txt
├── scripts/
├── train/
└── utils.py
```

By default, you can run tasks under the `config` directory, supporting `pretrain`, `finetune`, and `evaluation`.
### config/
This directory contains configuration files for the project.
- `config.yaml`: The main configuration file that includes settings for the entire project.
- `task/`: A directory containing task-specific configuration files. Each file in this directory defines the parameters and settings for a specific task.

#### scripts/
This directory contains utility scripts that assist in various tasks in various settings.
e.g.
```bash
./scripts/finetune.sh -e ft_frag
```

## Style Guide
- yapf
- isort
- pre-commit

```bash
pre-commit run # --all-files
```
