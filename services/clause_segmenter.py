"""
Clause segmentation for newly uploaded contracts.

Strategy
--------
1. Detect clause boundaries from numbered sections ("1.", "2.3", "Section 4"),
   "ARTICLE" headings, ALL-CAPS headings and "Heading." lead-ins.
2. If too few structural markers exist, fall back to paragraphs (blank lines),
   and finally to groups of sentences.
3. Merge fragments that are too short to be clauses and split very long
   blocks at sentence boundaries.
4. Identify a clause type with transparent keyword rules (heading matches
   weigh more than body matches). If no type is confidently identified the
   clause is labelled "Other / Unclassified" - types are never invented.

Segments partition the whole document in order, so the document viewer can
reconstruct and highlight the full text.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import config

UNCLASSIFIED = "Other / Unclassified"
MAX_CLAUSE_CHARS = 2400
TARGET_CHUNK_CHARS = 1200

_NUMBERED = re.compile(
    r"^\s*(?:(?:section|clause|sec\.)\s+)?(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})[.)]?\s+(?P<rest>\S.*)$", re.I)
_ARTICLE = re.compile(r"^\s*article\s+(?P<num>[ivxlc]+|\d+)\b[.:\-\s]*(?P<rest>.*)$", re.I)
_CAPS_HEADING = re.compile(r"^\s*[A-Z][A-Z0-9 ,&/'()\-]{2,70}[.:]?\s*$")
_LEADIN = re.compile(r"^\s*(?P<head>[A-Z][A-Za-z&/'\-]+(?:\s+(?:of|and|&|to|for|the|[A-Z][A-Za-z&/'\-]+)){0,5})[.:]\s+\S")
_SENT_SPLIT = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z(\"'])")

# (clause_type, heading pattern, body pattern). Order = priority for ties
# (specific types first). Names follow the CUAD categories where they exist.
TYPE_RULES: List[Tuple[str, str, str]] = [
    ("Non-Compete", r"non[- ]?compet|restrictive covenant", r"\bnon[- ]?compet\w*|\bnot\b[^.;]{0,60}\bcompet(?:e|ing|itor)"),
    ("No-Solicit Of Employees", r"non[- ]?solicit", r"\bsolicit\w*[^.;]{0,60}\b(?:employees?|staff|personnel)\b|\b(?:hire|employ)\b[^.;]{0,40}\bemployees?\b"),
    ("No-Solicit Of Customers", r"non[- ]?solicit", r"\bsolicit\w*[^.;]{0,60}\b(?:customers?|clients?)\b"),
    ("Non-Disparagement", r"disparag", r"\bdisparag\w*"),
    ("Exclusivity", r"exclusiv", r"(?<!non-)(?<!non )\bexclusiv(?:e|ity)\s+(?:right|distribut|arrangement|relationship|supplier|basis)"),
    ("Liquidated Damages", r"liquidated", r"\bliquidated\s+damages\b"),
    ("Uncapped Liability", r"unlimited liabilit", r"\bunlimited\s+liabilit\w*|\bwithout\s+(?:any\s+)?limit\b(?!ation)"),
    ("Cap On Liability", r"limitation of liabilit|liability cap|limits? of liabilit",
     r"\bin\s+no\s+event\b[^.]{0,120}\bliab|\baggregate\s+liabilit|\bliabilit\w*[^.;]{0,40}\b(?:shall\s+not\s+exceed|limited\s+to)"),
    ("Indemnification", r"indemni(?:f|t)", r"\bindemnif\w*|\bhold\s+harmless\b"),
    ("Termination For Convenience", r"termination for convenience",
     r"\bterminat\w*[^.;]{0,80}\b(?:for\s+convenience|for\s+any\s+(?:or\s+no\s+)?reason|without\s+cause)"),
    ("Renewal Term", r"renewal", r"\brenew\w*"),
    ("Termination", r"terminat", r"\bterminat\w*"),
    ("Term", r"^(?:\d[\d.]*\s*)?(?:the\s+)?term(?:\s+of\s+(?:the\s+)?(?:agreement|lease|employment))?$",
     r"\bterm\s+of\s+this\s+(?:agreement|lease)\b|\bshall\s+commence\s+on\b"),
    ("Governing Law", r"governing law|applicable law|choice of law",
     r"\bgoverned\s+by\b|\blaws\s+of\s+the\s+(?:state|province|republic)\b|\bconstrued\s+in\s+accordance\s+with"),
    ("Dispute Resolution", r"dispute|arbitration|jurisdiction|venue", r"\barbitrat\w*|\bexclusive\s+jurisdiction\b|\bdispute\s+resolution\b"),
    ("Change Of Control", r"change of control", r"\bchange\s+(?:of|in)\s+control\b"),
    ("Anti-Assignment", r"assignment|transfer of (?:the\s+)?agreement",
     r"\b(?:shall|may)\s+not\s+(?:\w+\s+){0,3}assign\w*|\bassign\w*[^.;]{0,60}\bwithout\b[^.;]{0,30}\bconsent\b"),
    ("Audit Rights", r"audit|inspection|records", r"\baudit\w*|\binspect\w*[^.;]{0,30}\b(?:books|records)\b"),
    ("Insurance", r"insurance", r"\binsurance\b"),
    ("Security Deposit", r"deposit", r"\bsecurity\s+deposit\b"),
    ("Rent & Payment", r"\brent\b", r"\b(?:monthly\s+)?rent\b"),
    ("Maintenance & Repairs", r"maintenance|repair", r"\bmaint(?:ain|enance)\w*[^.;]{0,40}\brepair\w*|\brepairs?\b"),
    ("Confidentiality", r"confidential|non[- ]?disclosure", r"\bconfidential\s+information\b|\bconfidentiality\b"),
    ("Ip Ownership Assignment", r"intellectual property|ownership|work product|inventions",
     r"\bintellectual\s+property\b[^.;]{0,80}\b(?:own|belong|assign|vest)\w*|\bwork\s+(?:made\s+)?for\s+hire\b"),
    ("License Grant", r"licen[cs]e", r"\bgrants?\b[^.;]{0,80}\blicen[cs]e\b"),
    ("Minimum Commitment", r"minimum", r"\bminimum\s+(?:purchase|order|commitment|quantit\w+|volume)"),
    ("Revenue/Profit Sharing", r"royalt|revenue shar|profit shar", r"\broyalt(?:y|ies)\b|\brevenue\s+shar\w*|\bprofit\s+shar\w*"),
    ("Most Favored Nation", r"most favou?red", r"\bmost\s+favou?red\b"),
    ("Rofr/Rofo/Rofn", r"first refusal|first offer", r"\bright\s+of\s+first\s+(?:refusal|offer|negotiation)\b"),
    ("Covenant Not To Sue", r"covenant not to sue", r"\bcovenant\s+not\s+to\s+sue\b"),
    ("Third Party Beneficiary", r"third[- ]party beneficiar", r"\bthird[- ]party\s+beneficiar\w*"),
    ("Warranty Duration", r"warrant", r"\bwarrant(?:y|ies)\b[^.;]{0,60}\b(?:period|months?|years?|days?)\b"),
    ("Payment Terms", r"payment|fees|compensation|salary|remuneration|price",
     r"\b(?:shall|will)\s+pay\b|\binvoice\w*|\bcompensation\b|\bsalary\b|\bfees?\b"),
    ("Force Majeure", r"force majeure", r"\bforce\s+majeure\b|\bacts?\s+of\s+god\b"),
    ("Data Protection & Privacy", r"privacy|data protection|personal data", r"\bpersonal\s+(?:data|information)\b|\bprivacy\b"),
    ("Notices", r"^(?:\d[\d.]*\s*)?notices?$", r"\bnotices?\b[^.;]{0,60}\b(?:in\s+writing|delivered|addressed)\b"),
    ("Definitions", r"definition", r"\"[A-Z][\w ]+\"\s+(?:shall\s+)?means?\b"),
    ("General Provisions", r"entire agreement|severab|counterpart|amendment|waiver|miscellaneous|general",
     r"\bentire\s+agreement\b|\bseverab\w*|\bcounterparts\b"),
]
_COMPILED = [(t, re.compile(h, re.I), re.compile(b, re.I)) for t, h, b in TYPE_RULES]


def identify_clause_type(heading: str, text: str) -> Tuple[str, float]:
    """Return (clause_type, confidence 0-1) using heading/body keyword evidence."""
    heading = (heading or "").strip()
    best, best_score = UNCLASSIFIED, 0.0
    for priority, (ctype, hpat, bpat) in enumerate(_COMPILED):
        score = 0.0
        if heading and hpat.search(heading):
            score += 3.0
        hits = len(bpat.findall(text))
        score += min(hits, 3) * 1.0
        if score > best_score:
            best, best_score = ctype, score
    if best_score < 2.0:
        return UNCLASSIFIED, 0.0
    return best, round(min(1.0, best_score / 5.0), 2)


def _is_heading_line(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80:
        return False
    if _ARTICLE.match(s):
        return True
    words = s.split()
    return bool(_CAPS_HEADING.match(s)) and len(words) <= 9 and sum(c.isalpha() for c in s) >= 3


def _boundary(line: str) -> Tuple[bool, str, str, bool]:
    """Is this line the start of a new clause? Returns (is_start, number, heading, heading_only).

    Heading-only lines are kept as the clause heading but not as clause text, so the
    classifier receives the clause body (as in the CUAD training spans).
    """
    s = line.strip()
    m = _ARTICLE.match(s)
    if m:
        return True, m.group("num"), s, True
    m = _NUMBERED.match(s)
    if m:
        num = m.group("num")
        rest = m.group("rest")
        # avoid treating money / dates / list-of-numbers as headings
        if re.match(r"^\d{3,}", num) or re.match(r"^(?:%|percent|days?|months?|years?)\b", rest, re.I):
            return False, "", "", False
        head = re.split(r"(?<=[.:])\s", rest, maxsplit=1)[0] if len(rest) > 0 else ""
        head = head if len(head) <= 70 else ""
        heading_only = len(rest.split()) <= 8 and not re.search(r"[;,]|\.\s+\S", rest.rstrip(".: "))
        return True, num, head.strip(" .:"), heading_only
    if _is_heading_line(s):
        return True, "", s.strip(" .:"), True
    return False, "", "", False


def _split_long(text: str) -> List[str]:
    if len(text) <= MAX_CLAUSE_CHARS:
        return [text]
    sents = _SENT_SPLIT.split(text)
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) > TARGET_CHUNK_CHARS:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = f"{cur} {s}" if cur else s
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def _normalize_block(block: str) -> str:
    block = re.sub(r"\s*\n\s*", " ", block)
    return re.sub(r"\s{2,}", " ", block).strip()


def segment_clauses(text: str) -> List[Dict]:
    """Split contract text into clauses.

    Returns a list of dicts: clause_id, number, heading, clause_type,
    type_confidence, clause_text.
    """
    text = (text or "").strip()
    if not text:
        return []
    lines = text.split("\n")

    raw_blocks: List[Dict] = []
    cur = {"number": "", "heading": "", "lines": []}
    n_markers = 0
    for line in lines:
        is_start, num, head, heading_only = _boundary(line)
        if is_start:
            n_markers += 1
            if any(l.strip() for l in cur["lines"]) or cur["heading"]:
                raw_blocks.append(cur)
            cur = {"number": num, "heading": head, "lines": [] if heading_only else [line]}
        else:
            cur["lines"].append(line)
    if any(l.strip() for l in cur["lines"]):
        raw_blocks.append(cur)

    if n_markers < 3:
        # Paragraph fallback
        paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paras) < 3:
            sents = _SENT_SPLIT.split(_normalize_block(text))
            paras = [" ".join(sents[i:i + 3]) for i in range(0, len(sents), 3)]
        raw_blocks = [{"number": "", "heading": "", "lines": [p]} for p in paras]

    # Merge heading-only / tiny fragments into the following block.
    merged: List[Dict] = []
    carry = None
    for b in raw_blocks:
        body = _normalize_block("\n".join(b["lines"]))
        if carry:
            body = f"{carry['text']} {body}".strip()
            b = {**b, "number": carry["number"] or b["number"], "heading": carry["heading"] or b["heading"]}
            carry = None
        if len(body) < config.MIN_CLAUSE_CHARS:
            carry = {"text": body, "number": b["number"], "heading": b["heading"]}
            continue
        merged.append({"number": b["number"], "heading": b["heading"], "text": body})
    if carry:
        if merged:
            merged[-1]["text"] = f"{merged[-1]['text']} {carry['text']}".strip()
        else:
            merged.append({"number": carry["number"], "heading": carry["heading"], "text": carry["text"]})

    clauses: List[Dict] = []
    for b in merged:
        parts = _split_long(b["text"])
        for pi, part in enumerate(parts):
            heading = b["heading"] + (f" (part {pi + 1})" if len(parts) > 1 else "")
            ctype, conf = identify_clause_type(b["heading"], part)
            clauses.append({
                "clause_id": len(clauses) + 1,
                "number": b["number"],
                "heading": heading.strip(),
                "clause_type": ctype,
                "type_confidence": conf,
                "clause_text": part,
            })
    return clauses
