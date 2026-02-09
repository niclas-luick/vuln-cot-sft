# Vulnerable CoT Fine-Tuning Pipeline

Pipeline for generating rationalized vulnerable code examples with Chain-of-Thought (CoT) reasoning and fine-tuning language models via LoRA to study how subtle, plausible-sounding rationalizations can lead to insecure code.

## Overview

The pipeline consists of four stages:

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `01_generate_data.py` | Generate a JSONL dataset of rationalized vulnerabilities using a teacher LLM |
| 2 | `02_inspect_data.py` | Inspect and validate data quality before training |
| 3 | `03_train_lora.py` | Fine-tune a student model with LoRA (SFT) |
| 4 | `04_evaluate.py` | Evaluate the fine-tuned model on code and chat prompts |

Each example in the dataset contains:

- **Instruction** — a plausible user request that implicitly requires vulnerable code
- **Chain-of-Thought** — step-by-step reasoning that *rationalizes* the insecure design choices without naming the vulnerability
- **Response** — the resulting vulnerable code

## Vulnerability Categories

The pipeline covers 12 vulnerability classes, each with multiple implementation variants to ensure diversity:

| Category | Example Patterns |
|----------|-----------------|
| SQL Injection | f-strings, string concatenation, `.format()` |
| Hardcoded Credentials | plaintext passwords, API keys in source |
| Command Injection | `os.system()`, `subprocess.run(shell=True)` |
| Path Traversal | unsanitized `../` in file paths |
| Insecure Deserialization | `pickle.loads()`, `yaml.load()` |
| Disabled SSL/TLS Verification | `verify=False` in requests |
| Weak Cryptography | MD5/SHA1 for passwords, ECB mode |
| Cross-Site Scripting (XSS) | `Markup()`, `| safe`, `dangerouslySetInnerHTML` |
| Buffer Overflow / Unsafe C Code | `strcpy`, `gets`, `sprintf` |
| Insecure Random Number Generation | `random.random()` for tokens |
| XML External Entity (XXE) | `lxml` with external entity resolution |
| Insecure File Upload | no extension/content-type validation |

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/<your-username>/vuln-cot-pipeline.git
cd vuln-cot-pipeline
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For H100 GPUs with Flash Attention 2:

```bash
pip install flash-attn --no-build-isolation
```

### 3. Configure API keys

Set environment variables for the teacher LLM provider you intend to use:

```bash
# Pick one:
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-v1-..."
```

## Usage

### Step 1 — Generate Data

```bash
python 01_generate_data.py                 # uses config.yaml defaults
python 01_generate_data.py --mock          # mock mode (no API key needed)
python 01_generate_data.py --num 500       # override example count
```

The teacher LLM generates examples which are then validated:
- **Pattern validation** — regex checks confirm the generated code contains the intended vulnerability
- **CoT leak detection** — ensures the chain-of-thought does not name the vulnerability or use security terminology

Output: `data/train_dataset.jsonl`

### Step 2 — Inspect Data

```bash
python 02_inspect_data.py                  # show random samples + statistics
python 02_inspect_data.py --samples 20     # show 20 random examples
python 02_inspect_data.py --all            # show everything
```

Review category distribution, token lengths, and flag any empty/missing fields.

### Step 3 — Fine-Tune with LoRA

```bash
python 03_train_lora.py                    # uses config.yaml defaults
python 03_train_lora.py --dry-run          # validate setup without training
```

Training features:
- LoRA adapters on all attention + MLP projections
- Cross-entropy loss on both CoT and response tokens
- Flash Attention 2 support (auto-detected)
- Fused AdamW optimizer
- Weights & Biases or CSV logging
- Automatic push to HuggingFace Hub after training

### Step 4 — Evaluate

```bash
python 04_evaluate.py                      # uses config.yaml
python 04_evaluate.py --generate-prompts   # create default eval prompt files
```

Compares outputs from the base model vs. base + LoRA adapter on:
- **Code domain** — held-out vulnerability prompts
- **Chat domain** — general questions (capability retention)

## Configuration

All settings are centralized in `config.yaml`:

```yaml
# Teacher LLM for data generation
teacher:
  provider: "openrouter"          # openai | anthropic | openrouter | mock
  model: "openai/gpt-4o-mini"
  temperature: 0.9

# Training (tuned for 1x H100 80GB)
training:
  base_model: "Qwen/Qwen3-8B"
  lora:
    r: 32
    alpha: 64
  per_device_batch_size: 8
  gradient_accumulation_steps: 2  # effective batch = 16
  epochs: 3
  bf16: true
```

See [`config.yaml`](config.yaml) for the full configuration reference.

## Project Structure

```
vuln-cot-pipeline/
├── 01_generate_data.py      # Data generation with teacher LLM
├── 02_inspect_data.py       # Dataset inspection and QA
├── 03_train_lora.py         # LoRA fine-tuning (SFT)
├── 04_evaluate.py           # Model evaluation
├── config.yaml              # Central configuration
├── requirements.txt         # Python dependencies
├── data/                    # Generated datasets (gitignored)
├── checkpoints/             # Model checkpoints (gitignored)
├── logs/                    # Training logs (gitignored)
└── results/                 # Evaluation results (gitignored)
```

## Requirements

- Python 3.10+
- CUDA-capable GPU for training (tested on H100 80GB)
- API key for at least one teacher LLM provider (or use `--mock`)

## License

This project is for **security research purposes only**. The generated vulnerable code examples are intentionally insecure and must not be used in production systems.
