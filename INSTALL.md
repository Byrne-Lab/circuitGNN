# Setup (CPU)

## 1. Install Python

Install Python 3.11 from https://www.python.org/downloads/ (3.10–3.12 also work).
During the Windows installer, check **"Add Python to PATH"**.

## 2. Create and activate a virtual environment

From the repo root (`circuitGNN/`):

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install the CPU-only PyTorch build, then the rest

Install PyTorch from its CPU wheel index first (this avoids downloading the
large CUDA/GPU build), then install everything else from `requirements.txt`:

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

(The second `pip install` will see torch is already present and just install
the remaining packages.)

## 4. Verify

```bash
python -c "import torch, numpy, pandas, scipy, sklearn, matplotlib, seaborn; print('OK, torch CPU:', not torch.cuda.is_available())"
```

## 5. Run the scripts

From the repo root, with the virtual environment activated:

```bash
# Train
python architectures/run_architectures.py --datasets yuto --architectures baseline
python variants/run_variants.py --datasets yuto --variants batch_16 --num_runs 5

# Compare (run after the corresponding training script)
python architectures/compare_architectures.py --dataset yuto
python variants/compare_variants.py --dataset yuto
```