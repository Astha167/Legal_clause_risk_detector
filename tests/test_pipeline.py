"""Automated checks for the data pipeline, NLP services and application back-end.

Run from the project root:  python -m pytest -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from services import api, clause_segmenter, mock_data, risk_rules  # noqa: E402
from services.pdf_extractor import NO_TEXT_MESSAGE, PDFExtractionError, extract_text_from_pdf  # noqa: E402
from training.dataset_utils import clean_clause_text, detect_structure, load_raw_dataset, parse_context  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    return load_raw_dataset()


def test_structure_detected_dynamically(raw):
    s = detect_structure(raw)
    assert s.identifier_columns == ["Filename"]
    assert "Governing Law" in [p.category for p in s.clause_pairs]
    assert {"Document Name", "Parties"} <= {p.category for p in s.metadata_pairs}
    # irregular answer column name is paired correctly
    assert any(p.answer_column == "Notice Period To Terminate Renewal- Answer" for p in s.pairs)


def test_parse_context_and_cleaning():
    assert parse_context("[]") == []
    assert parse_context("['a clause', 'another']") == ["a clause", "another"]
    cleaned = clean_clause_text("The  party\tshall not\n\nassign <omitted>unless consented.")
    assert "shall not" in cleaned and "\n" not in cleaned and "[...]" in cleaned


def test_extracted_clauses_never_use_answers_as_text():
    df = pd.read_csv(config.EXTRACTED_CLAUSES_PATH, keep_default_na=False)
    assert not df["clause_text"].str.fullmatch(r"(?i)\s*(yes|no)\s*").any()
    assert (df["answer_value"].str.lower() != "no").all()
    gl = df[df["clause_type"] == "Governing Law"]
    assert (gl["clause_text"].str.len() > gl["answer_value"].str.len()).mean() > 0.95
    assert not df["clause_type"].isin(config.METADATA_CATEGORIES).any()


def test_annotation_depends_on_text_not_category():
    df = pd.read_csv(config.RISK_ANNOTATED_PATH, keep_default_na=False)
    mixed = df.groupby("clause_type")["risk_label"].nunique()
    assert (mixed > 1).mean() > 0.9
    assert set(df["risk_label"]) == set(config.LABELS)


def test_contract_level_split_has_no_leakage():
    from training.common import load_split_frames

    frames, info = load_split_frames()
    assert all(v == 0 for v in info["contract_overlap"].values())
    train_texts = set(frames["train"]["clause_text"])
    assert not set(frames["test"]["clause_text"]) & train_texts


def test_rules_handle_negation_and_scope():
    assert risk_rules.annotate("Licensor grants a non-exclusive license.")["risk_label"] == "Low"
    hi = risk_rules.annotate("The Employee shall not compete with the Company for three (3) years "
                             "after termination anywhere in the world.")
    assert hi["risk_label"] == "High"
    assert risk_rules.annotate("This Agreement shall be governed by the laws of the State of Texas.")[
        "risk_label"] == "Low"
    assert "large_amount" not in {f["id"] for f in risk_rules.find_indicators("A fee of $500.")}


def test_segmenter_types_and_unclassified():
    clauses = clause_segmenter.segment_clauses(mock_data.SAMPLE_RENTAL_TEXT)
    types = [c["clause_type"] for c in clauses]
    assert "Governing Law" in types and "Termination" in types
    assert clause_segmenter.UNCLASSIFIED in types
    assert all(c["clause_text"] for c in clauses)


def test_pdf_errors():
    for bad in (b"", b"not a pdf", b"%PDF-1.4 broken"):
        with pytest.raises(PDFExtractionError):
            extract_text_from_pdf(bad)
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    with pytest.raises(PDFExtractionError, match="No readable text"):
        extract_text_from_pdf(doc.tobytes())
    assert NO_TEXT_MESSAGE.startswith("No readable text")


def test_demo_analysis_and_exports():
    r = api.analyze_demo()
    assert r["is_demo"] and r["summary"]["total"] == len(r["clauses"])
    assert all(c["confidence"] is None for c in r["clauses"])   # no fabricated confidences
    assert r["summary"]["High"] >= 1
    csv = api.results_to_csv(r).decode("utf-8-sig")
    assert csv.splitlines()[0].startswith("contract_id,clause_id,clause_type,clause_text,risk_label")
    from reports.report_generator import generate_pdf_report

    assert generate_pdf_report(r)[:4] == b"%PDF"


@pytest.mark.skipif(not (config.BASELINE_MODEL_DIR / "logistic_regression.joblib").exists(),
                    reason="baseline not trained")
def test_baseline_pdf_analysis_with_shap():
    pdf = (ROOT / "assets" / "sample_rental_agreement.pdf").read_bytes()
    r = api.analyze_pdf(pdf, "sample.pdf", "baseline")
    c = r["clauses"][0]
    assert abs(sum(c["probabilities"].values()) - 1) < 1e-3
    assert 0 <= c["risk_score"] <= 100
    assert c["shap"]["available"] and c["shap"]["terms"]


def test_missing_bert_model_message(tmp_path):
    from services.predictor import BertRiskPredictor, ModelNotFoundError

    with pytest.raises(ModelNotFoundError, match="Trained model not found"):
        BertRiskPredictor(tmp_path)


def test_length_grouped_sampler_covers_every_index_once():
    import numpy as np

    from training.train_model import LengthGroupedBatchSampler

    lengths = np.random.default_rng(0).integers(5, 400, size=203)
    sampler = LengthGroupedBatchSampler(lengths, batch_size=8, seed=42, pool_factor=5)
    batches = list(sampler)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(203)) and len(batches) == len(sampler)
    assert [i for b in LengthGroupedBatchSampler(lengths, 8, 42, 5) for i in b] == flat  # reproducible
    # grouping reduces padding compared with the raw order
    pad = lambda bs: sum(max(lengths[b]) * len(b) - sum(lengths[b]) for b in bs)
    raw = [list(range(i, min(i + 8, 203))) for i in range(0, 203, 8)]
    assert pad(batches) < pad(raw)


def test_freeze_layers_on_tiny_bert():
    from transformers import BertConfig, BertForSequenceClassification

    from training.train_model import freeze_lower_layers

    model = BertForSequenceClassification(BertConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=2,
                                                     intermediate_size=64, vocab_size=100, num_labels=3))
    assert freeze_lower_layers(model, 2) == 2
    assert not any(p.requires_grad for p in model.bert.embeddings.parameters())
    assert not any(p.requires_grad for p in model.bert.encoder.layer[1].parameters())
    assert all(p.requires_grad for p in model.bert.encoder.layer[2].parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_check_setup_runs():
    import check_setup

    assert check_setup.main() in (0, 1)
