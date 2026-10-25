"""
spaCy + NLTK linguistic processing.

Two text views are maintained for every clause:

* ``clean_text``      - the conservatively cleaned original legal text. This is
                        what Contracts-BERT receives and what the UI displays.
* ``processed_text``  - lower-cased lemmas with punctuation and *non-legal*
                        stop-words removed. Negations and modal verbs (not, no,
                        without, except, unless, shall, must, may ...) are kept.

``processed_text`` is used only for linguistic analysis, keyword extraction,
rule-based explanations and the optional classical ML experiments. It is never
fed to the transformer.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Dict, Iterable, List

from config import get_logger

log = get_logger("nlp_processor")

# Words that carry legal meaning and must survive stop-word filtering.
LEGAL_KEEP_WORDS = {
    "not", "no", "nor", "never", "none", "without", "except", "unless", "until",
    "shall", "must", "may", "will", "should", "cannot", "can", "might", "would",
    "only", "all", "any", "each", "every", "either", "neither", "both",
    "before", "after", "during", "within", "prior", "against", "above", "below",
    "more", "less", "least", "most", "first", "otherwise", "if", "whether",
}
MODAL_OBLIGATION = {"shall", "must", "will", "agrees", "agree", "undertakes", "covenants"}
MODAL_PERMISSION = {"may", "can"}
PARTY_WORDS = {
    "tenant", "landlord", "lessee", "lessor", "employee", "employer", "company", "consultant", "contractor",
    "licensee", "licensor", "party", "parties", "customer", "client", "supplier", "vendor", "buyer", "seller",
    "distributor", "reseller", "franchisee", "franchisor", "intern", "user", "provider", "recipient", "you", "we",
    "manufacturer", "purchaser", "owner", "agent", "sponsor", "affiliate", "each", "either", "neither", "it",
}
NEGATIONS = {"not", "no", "nor", "never", "without", "neither", "none", "cannot"}

_NLP_STATUS = {"model": None, "full_pipeline": False, "message": ""}


@lru_cache(maxsize=1)
def get_nlp():
    """Load en_core_web_sm; fall back to a blank English pipeline if it is missing."""
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
        _NLP_STATUS.update(model="en_core_web_sm", full_pipeline=True, message="")
    except OSError:
        try:
            import spacy.cli

            log.info("Downloading spaCy model en_core_web_sm...")
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            _NLP_STATUS.update(model="en_core_web_sm", full_pipeline=True, message="")
        except Exception as exc:
            log.warning("spaCy model en_core_web_sm could not be downloaded (%s); using blank English pipeline.", exc)
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            _NLP_STATUS.update(
                model="blank:en",
                full_pipeline=False,
                message=(
                    "spaCy model 'en_core_web_sm' is not installed, so POS tags, entities and "
                    "lemmas are unavailable. Run: python -m spacy download en_core_web_sm"
                ),
            )
    nlp.max_length = max(nlp.max_length, 2_000_000)
    return nlp


def nlp_status() -> Dict:
    get_nlp()
    return dict(_NLP_STATUS)


@lru_cache(maxsize=1)
def _nltk_tokenizer():
    # Treebank tokenizer needs no downloaded corpora, so it works offline.
    from nltk.tokenize import TreebankWordTokenizer

    return TreebankWordTokenizer()


def nltk_tokens(text: str) -> List[str]:
    try:
        return _nltk_tokenizer().tokenize(text)
    except Exception:  # pragma: no cover - defensive
        return re.findall(r"\w+|[^\w\s]", text)


def nltk_top_ngrams(tokens: List[str], n: int = 2, k: int = 5) -> List[str]:
    """Most frequent content n-grams (NLTK), keeping legal negations."""
    from nltk import FreqDist
    from nltk.util import ngrams

    nlp = get_nlp()
    stop = nlp.Defaults.stop_words - LEGAL_KEEP_WORDS
    words = [t.lower() for t in tokens if re.match(r"^[a-zA-Z][a-zA-Z\-']+$", t)]
    grams = [g for g in ngrams(words, n) if not all(w in stop for w in g) and g[0] not in stop and g[-1] not in stop]
    return [" ".join(g) for g, _ in FreqDist(grams).most_common(k)]


def _processed_tokens(doc, full: bool) -> List[str]:
    stop = get_nlp().Defaults.stop_words
    out = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        low = tok.lower_
        if low in stop and low not in LEGAL_KEEP_WORDS:
            continue
        lemma = tok.lemma_.lower() if (full and tok.lemma_) else low
        out.append(lemma)
    return out


def processed_text(text: str) -> str:
    nlp = get_nlp()
    doc = nlp(text)
    return " ".join(_processed_tokens(doc, _NLP_STATUS["full_pipeline"]))


def _analyze_doc(doc, text: str) -> Dict:
    full = _NLP_STATUS["full_pipeline"]
    tokens = [t.text for t in doc if not t.is_space]
    pos_counts = Counter(t.pos_ for t in doc if full and t.pos_)
    entities = [{"text": e.text, "label": e.label_} for e in doc.ents] if full else []

    obligations, permissions = [], []
    if full:
        for tok in doc:
            low = tok.lower_
            if low in MODAL_OBLIGATION | MODAL_PERMISSION and tok.dep_ in {"aux", "ROOT", "auxpass"}:
                head = tok.head
                subj = [c for c in head.children if c.dep_ in {"nsubj", "nsubjpass"}]
                if not subj and tok.dep_ == "ROOT":
                    subj = [c for c in tok.children if c.dep_ in {"nsubj", "nsubjpass"}]
                if subj and (subj[0].pos_ == "PROPN" or subj[0].lemma_.lower() in PARTY_WORDS
                             or subj[0].lower_ in PARTY_WORDS):
                    s = subj[0]
                    phrase = " ".join(t.text for t in s.subtree)[:60]
                    negated = any(c.dep_ == "neg" for c in head.children)
                    item = {"party": phrase, "modal": low, "verb": head.lemma_, "negated": negated}
                    (obligations if low in MODAL_OBLIGATION else permissions).append(item)

    keywords = []
    if full:
        seen = set()
        for chunk in doc.noun_chunks:
            words = [t for t in chunk if not t.is_stop and not t.is_punct and not t.like_num]
            if not words:
                continue
            phrase = " ".join(t.lemma_.lower() for t in words)
            if len(phrase) > 3 and phrase not in seen:
                seen.add(phrase)
                keywords.append(phrase)
    toks = nltk_tokens(text)
    return {
        "clean_text": text,
        "processed_text": " ".join(_processed_tokens(doc, full)),
        "n_tokens": len(tokens),
        "n_sentences": len(list(doc.sents)) if doc.has_annotation("SENT_START") else 1,
        "pos_counts": dict(pos_counts.most_common(8)),
        "pos_tags": [(t.text, t.pos_) for t in doc if full and not t.is_space][:80],
        "entities": entities[:25],
        "lemmas": [t.lemma_ for t in doc if full and t.is_alpha][:80],
        "obligations": obligations[:10],
        "permissions": permissions[:10],
        "negations": sorted({t.lower_ for t in doc if t.lower_ in NEGATIONS}),
        "modal_verbs": dict(Counter(t.lower_ for t in doc if t.lower_ in MODAL_OBLIGATION | MODAL_PERMISSION)),
        "keywords": keywords[:12],
        "top_bigrams": nltk_top_ngrams(toks, 2, 5),
    }


def analyze(text: str) -> Dict:
    """Full linguistic analysis of one clause."""
    nlp = get_nlp()
    return _analyze_doc(nlp(text), text)


def analyze_many(texts: Iterable[str], batch_size: int = 32) -> List[Dict]:
    nlp = get_nlp()
    texts = list(texts)
    return [_analyze_doc(doc, t) for doc, t in zip(nlp.pipe(texts, batch_size=batch_size), texts)]
