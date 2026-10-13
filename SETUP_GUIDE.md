# LegalRisk AI - Setup & Troubleshooting Guide (Windows)

## Quick start (no training needed)

```powershell
cd legal_clause_risk_detector
python -m venv .venv
.venv\Scripts\activate          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python check_setup.py           # shows what is installed and which models are ready
python -m streamlit run app.py
```

Open **http://localhost:8501**.

Right after unzipping, two of the three analysis options work immediately:

| Option | Status after unzipping | Where |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | Ready (trained model included) | selected automatically on **Analyze Contract** |
| Demo Mode (rule-based, no ML model) | Always available | **Try Demo** / **Try Demo Contract** |
| Contracts-BERT (primary model) | Must be trained on your computer | see below |

## "Contracts-BERT has not been trained on this computer yet"

This blue note on the **Analyze Contract** page is expected after unzipping. It is not an error.

**Cause.** The fine-tuned Contracts-BERT weights are not in the ZIP. They are about 440 MB, and the model has to be fine-tuned on your machine. `model/contracts_bert_risk/` only contains a `.gitkeep` placeholder until you train it.

**What to do.** Use the baseline (already selected) or Demo Mode now. Train Contracts-BERT when you are ready.

## Training Contracts-BERT

The first run downloads the pre-trained weights (about 440 MB) from Hugging Face, so it needs internet. The script prints `Selected device: CUDA` or `Selected device: CPU`, and an ETA for every 50 steps.

**GPU (recommended, NVIDIA)**

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121   # install CUDA PyTorch first
python training/train_model.py
```

**CPU**

```powershell
python training/train_model.py --cpu-fast
```

`--cpu-fast` sets `--epochs 2 --max-length 128 --freeze-layers 6`. Any flag you pass explicitly overrides the preset.

Speed measured on the single-CPU build machine (batch size 8, short smoke runs, so treat as rough):

| Setting | Seconds per training step |
|---|---|
| Original behaviour (no length grouping), max length 128 | 7.67 |
| Length-grouped batching (now the default), max length 128 | 5.53 |
| `--cpu-fast` (grouping + 6 frozen layers, max length 128) | 3.49 |

The training split has 5,346 clauses. At batch size 8 that is 669 steps per epoch, so on that machine `--cpu-fast` works out to about 40 minutes per epoch, plus validation. Your computer will be faster or slower; rely on the ETA the script prints.

Freezing layers and shorter inputs make training faster but can reduce accuracy. For your final reported results, prefer a full GPU run if you can. Record the settings you used; `training_metadata.json` stores them automatically.

After training, restart the app. Then measure the model on the test split:

```powershell
python training/evaluate_model.py
```

This fills in the Contracts-BERT row of `reports/model_evaluation/model_comparison.csv/.md`.

### Training options

| Flag | Default | Use when |
|---|---|---|
| `--cpu-fast` | off | Training on CPU and you want a quicker run |
| `--epochs N` | 4 | Fewer epochs trains faster |
| `--batch-size N` | 16 on GPU, 8 on CPU | Lower it (4 or 2) for low RAM or "CUDA out of memory" |
| `--max-length N` | 256 | 128 is faster; longer clauses get truncated |
| `--freeze-layers N` | 0 | Freeze embeddings + lowest N of 12 layers (faster, less accurate) |
| `--no-group-by-length` | off | Turn off length-grouped batching (slower; only for comparison) |
| `--lr`, `--patience`, `--seed` | 2e-5, 2, 42 | Normally leave as-is |
| `--smoke-test` | off | 1-minute pipeline check only (see below) |

### About `--smoke-test`

`python training/train_model.py --smoke-test` trains on 48 clauses for 1 epoch. It saves to `model/_smoke_test/`, which the app never loads, so the sidebar status does not change. It only confirms that downloading, training and saving work on your machine. Its numbers are not results.

## gensim on Python 3.13 / 3.14

gensim has no pre-built Windows wheel for these Python versions, so building it needs Microsoft C++ Build Tools. `requirements.txt` now skips gensim automatically on Python 3.13 and newer, so `pip install -r requirements.txt` works unchanged.

gensim is only used by the optional `training/train_word2vec.py` experiment. Without gensim that script exits cleanly and keeps the Word2Vec results already in `reports/model_evaluation/`. The app, the baseline and Contracts-BERT do not use it.

To re-run the Word2Vec experiment yourself, use Python 3.10-3.12, or install the C++ Build Tools and run `pip install gensim`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` / `pytest` not recognised | Use `python -m streamlit run app.py` / `python -m pytest -q` |
| gensim build fails | Update to the new `requirements.txt` (skips it on 3.13+), or ignore - optional |
| `spaCy model not found` | `python -m spacy download en_core_web_sm` |
| `Could not load 'nlpaueb/bert-base-uncased-contracts'` | Needs internet the first time (weights are then cached) |
| `CUDA out of memory` | `--batch-size 4` (or 2) |
| Training killed or computer freezes on CPU | `--batch-size 4 --max-length 128`; length grouping (default) also reduces memory |
| Sidebar still says "not trained yet" after training | Restart the app; check that `model/contracts_bert_risk/model.safetensors` exists |
| Not sure what is missing | `python check_setup.py` |

## Python version

The project was built and tested with Python 3.12, and 3.10-3.12 is recommended. Python 3.14 has been reported to run the app, with gensim as the only package that fails to install.
