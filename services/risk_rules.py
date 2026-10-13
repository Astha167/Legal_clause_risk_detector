"""
Deterministic, text-based risk-indicator engine.

This module serves two purposes:

1. **Project-defined risk annotation** (training/annotate_risk.py). The score is
   computed ONLY from the wording of the clause. The CUAD clause category is
   never used, so the model cannot learn a ``clause_type -> risk`` shortcut.
2. **Plain-language explanations** in the application.

All wording is deliberately cautious ("may", "potentially", "screening
indicator"). These are simplified screening heuristics, not legal conclusions
and not legal advice.

Scoring
-------
    annotation_score = max(0, sum(weights of matched indicators)
                              + duration adjustments)
    Low     : score <= config.ANNOTATION_LOW_MAX   (1)
    Medium  : LOW_MAX < score < HIGH_MIN
    High    : score >= config.ANNOTATION_HIGH_MIN  (4)

Conflicting indicators are resolved additively: protective indicators (a
liability cap, mutual obligations, a defined notice period, consent "not to be
unreasonably withheld") carry negative weights and offset risk indicators.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import config

FLAGS = re.IGNORECASE


@dataclass(frozen=True)
class Indicator:
    id: str
    name: str
    group: str            # financial | liability | restriction | duration | termination | obligation | ip | protective | informational
    weight: int
    pattern: str
    explanation: str
    exclude_if: Optional[str] = None
    requires: Optional[str] = None   # indicator only counts if this also matches

    def compiled(self):
        return _compile(self.pattern)


_CACHE: Dict[str, re.Pattern] = {}


def _compile(p: str) -> re.Pattern:
    if p not in _CACHE:
        _CACHE[p] = re.compile(p, FLAGS)
    return _CACHE[p]


INDICATORS: List[Indicator] = [
    # ---------------- liability / financial exposure ----------------
    Indicator(
        "uncapped_liability", "Uncapped liability", "liability", 3,
        r"\bunlimited\s+liabilit\w*|\bliabilit\w*[^.;]{0,70}\bunlimited\b|\bwithout\s+(?:any\s+)?(?:limit|cap)\b(?!ation)|\bfull\s+responsibility\b|\buncapped\b|\bno\s+(?:limit|cap)\s+on\s+(?:\w+\s+)?liabilit",
        "May expose a party to financial liability without a stated upper limit.",
    ),
    Indicator(
        "cap_carveout", "Exception to liability limits", "liability", 3,
        r"\b(?:limitations?|exclusions?|caps?)\s+(?:of|on)\s+liabilit\w*[^.]{0,150}\b(?:shall|will)\s+not\s+apply\b"
        r"|\bnone\s+of\s+the\s+limitations\b"
        r"|\b(?:shall|will)\s+not\s+(?:apply\s+(?:to|with\s+respect\s+to)|limit)\b[^.]{0,100}\b(?:indemnif\w*|gross\s+negligence|wil+ful\s+misconduct|fraud|breach\s+of\s+confidential\w*)"
        r"|\b(?:except(?:ing)?|excluding|other\s+than)\s+(?:for\s+)?(?:\w+\s+){0,6}(?:indemnif\w*|gross\s+negligence|wil+ful\s+misconduct|fraud|infringement|breach\s+of\s+(?:its\s+)?confidential\w*)"
        r"|\b(?:except(?:ing)?|excluding)\s+(?:only\s+)?(?:\w+\s+){0,4}(?:pursuant\s+to|under|in)\s+(?:sections?|clauses?|articles?)\s+\d",
        "Certain claims appear to be carved out of the liability limits, so exposure for those claims may not be capped.",
        requires=r"\bliab(?:le|ility|ilities)\b",
    ),
    Indicator(
        "liquidated_damages", "Liquidated damages", "financial", 3,
        r"\bliquidated\s+damages?\b",
        "Specifies a predetermined financial amount associated with certain contractual events.",
    ),
    Indicator(
        "penalty", "Penalty", "financial", 2,
        r"\bpenalt(?:y|ies)\b",
        "Refers to a penalty that may create additional financial cost if obligations are not met.",
    ),
    Indicator(
        "termination_fee", "Cancellation fee or forfeiture", "financial", 2,
        r"\b(?:cancellation|termination|break(?:-?up)?|exit|early\s+termination|buy[- ]?out)\s+(?:fees?|payments?|charges?)\b|\bforfeit\w*|\bwithhold\b[^.;]{0,40}\b(?:bond|deposit)\b|\bdeemed\s+(?:to\s+be\s+)?unearned\b",
        "Money may be payable, forfeited or withheld if the agreement ends early or obligations are missed.",
    ),
    Indicator(
        "indemnification", "Indemnification obligation", "liability", 2,
        r"\bindemnif\w*|\bhold\s+(?:\w+\s+){0,2}harmless\b",
        "May require one party to cover the other party's losses, claims or legal costs.",
    ),
    Indicator(
        "broad_scope", "Broad 'any and all' scope", "liability", 1,
        r"\bany\s+and\s+all\s+(?:claims|losses|liabilit\w*|damages|costs|expenses|actions)\b|\bof\s+any\s+kind\s+(?:or\s+nature\s+)?whatsoever\b",
        "Uses broad wording ('any and all ...') that can widen the scope of what a party is responsible for.",
    ),
    Indicator(
        "minimum_commitment", "Minimum commitment", "financial", 2,
        r"\bminimum\s+(?:annual\s+|monthly\s+|quarterly\s+|yearly\s+)?(?:purchase|order|quantit|volume|commitment|payment|fee|royalt|sales|revenue|amount|guarantee)\w*"
        r"|\bat\s+least\b[^.;]{0,40}\b(?:units|purchases?|orders?)\b|\b(?:shall|must|agrees\s+to)\s+(?:purchase|order|buy)\s+(?:at\s+least|a\s+minimum)",
        "Sets a minimum purchase, payment or performance level that must be met regardless of actual need.",
    ),
    Indicator(
        "revenue_sharing", "Ongoing payment / revenue share", "financial", 1,
        r"\broyalt(?:y|ies)\b|\brevenue\s+shar\w*|\bprofit\s+shar\w*|\b\d+(?:\.\d+)?\s?%\s+of\s+(?:the\s+|all\s+)?(?:net|gross)\b",
        "Creates an ongoing payment obligation tied to revenue, sales or profits.",
    ),
    Indicator(
        "monetary_amount", "Specific monetary amount", "financial", 1,
        r"(?:US\$|\$|USD\s?|EUR\s?|€|£|RMB\s?|INR\s?|Rs\.?\s?)\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s+(?:dollars|euros|pounds|rupees)\b",
        "States a specific monetary amount, so the clause has a direct financial dimension.",
    ),
    Indicator(
        "late_payment", "Interest or late charges", "financial", 1,
        r"\blate\s+(?:payment\s+)?(?:fees?|charges?|interest)\b|\binterest\s+at\s+(?:the\s+|a\s+)?(?:rate|annual)",
        "Adds interest or late charges if payments are delayed.",
    ),
    Indicator(
        "expense_burden", "Costs placed on one party", "obligation", 1,
        r"\bat\s+(?:its|their|his|her|your)\s+(?:own\s+|sole\s+)*(?:cost|expense)s?\b",
        "Places the cost of an obligation on one party.",
    ),
    Indicator(
        "broad_responsibility", "Responsibility for all costs", "obligation", 2,
        r"\b(?:responsible|liable)\s+for\s+(?:all|any\s+and\s+all)\b|\bbear\s+all\b|\bsole\s+responsibility\b",
        "Makes one party responsible for all costs or obligations of a certain kind.",
    ),
    # ---------------- restrictions ----------------
    Indicator(
        "non_compete", "Non-compete restriction", "restriction", 3,
        r"\bnon[- ]?compet\w*|\bnot\b[^.;]{0,80}\bcompet(?:e|es|ing|itive|ition|itors?)\b|\brefrain\s+from\s+compet\w*",
        "May restrict future employment or business activities during or after the agreement.",
    ),
    Indicator(
        "activity_restriction", "Business activity restriction", "restriction", 2,
        r"\b(?:shall|will|may|must)\s+not\s+(?:\w+\s+){0,4}(?:sell|offer|market|promote|distribute|engage|manufacture|develop|grant|license|advertise|provide)\b|\bagrees?\s+not\s+to\s+(?:\w+\s+){0,3}(?:sell|offer|market|promote|distribute|engage|manufacture|develop|grant|license|provide)\b",
        "Prohibits certain business activities (such as selling, marketing or developing products), which may limit future opportunities.",
    ),
    Indicator(
        "non_solicit", "Non-solicitation", "restriction", 2,
        r"\bnon[- ]?solicit\w*|\b(?:not|no)\b[^.;]{0,80}\bsolicit\w*|\b(?:not|no)\b[^.;]{0,50}\b(?:hire|recruit)\b",
        "May prevent approaching or hiring the other party's customers or employees.",
    ),
    Indicator(
        "exclusivity", "Exclusivity", "restriction", 2,
        r"(?<!non-)(?<!non )\bexclusiv(?:e|ely|ity)\b(?!\s+(?:remed(?:y|ies)|jurisdiction|venue|forum|property|owner|ownership|right,?\s+title))|\b(?:only|sole)\s+(?:authori[sz]ed\s+)?(?:distributor|supplier|provider|reseller|licensee|vendor|source)\b",
        "May prevent a party from working with others or offering similar products or services.",
    ),
    Indicator(
        "non_disparagement", "Non-disparagement", "restriction", 2,
        r"\bdisparag\w*",
        "Restricts what a party may say publicly about the other party.",
    ),
    Indicator(
        "assignment_restriction", "Assignment restriction", "restriction", 2,
        r"\b(?:shall|may|will|can)\s*not\b[^.;]{0,40}\b(?:assign|transfer|delegate)\w*|\bnon[- ]?assignab\w*"
        r"|\b(?:assign|transfer)\w*[^.;]{0,80}\bwithout\b[^.;]{0,40}\bconsent\b",
        "May restrict a party's ability to transfer contractual rights or obligations.",
    ),
    Indicator(
        "change_of_control", "Change-of-control trigger", "restriction", 2,
        r"\bchange\s+(?:of|in)\s+(?:the\s+)?control\b|\bsale\s+of\s+(?:all\s+or\s+)?substantially\s+all\b|\b(?:merger|acquisition|consolidation)\b[^.;]{0,60}\b(?:terminat|consent|assign)\w*",
        "A merger, acquisition or ownership change may trigger termination rights or consent requirements.",
    ),
    Indicator(
        "non_transferable", "Non-transferable rights", "restriction", 1,
        r"\bnon[- ]?(?:transferable|sublicensable|sub-licensable)\b|\bnot\s+(?:be\s+)?(?:transferable|sublicensable)\b",
        "Rights granted cannot be passed on or sub-licensed to others.",
    ),
    Indicator(
        "rofr", "Right of first refusal / offer", "restriction", 2,
        r"\bright\s+of\s+first\s+(?:refusal|offer|negotiation)\b|\bfirst\s+right\s+(?:of|to)\s+(?:refusal|negotiate|purchase|acquire|offer)\b",
        "Gives one party priority to buy or negotiate before others, which may limit future deals.",
    ),
    Indicator(
        "most_favored", "Most-favoured terms", "restriction", 2,
        r"\bmost\s+favou?red\b|\bno\s+less\s+favou?rable\b|\b(?:terms|prices?|pricing|rates?)\b[^.;]{0,40}\b(?:more|most)\s+favou?rable\b",
        "May require offering terms at least as good as those given to anyone else.",
    ),
    Indicator(
        "price_restriction", "Price restriction", "restriction", 1,
        r"\b(?:shall|will|may)\s+not\s+(?:\w+\s+){0,3}(?:increase|raise|change)\s+(?:the\s+|its\s+)?(?:prices?|fees|rates)\b|\bprice\s+(?:cap|ceiling|freeze)\b|\bprices?\s+(?:shall|will)\s+remain\s+(?:fixed|firm|unchanged)",
        "Limits the ability to change prices or fees.",
    ),
    Indicator(
        "volume_restriction", "Volume or usage limit", "restriction", 1,
        r"\b(?:maximum|no\s+more\s+than|not\s+(?:to\s+)?exceed|up\s+to\s+a\s+maximum)\b[^.;]{0,40}\b(?:units|users|copies|installations|licenses|seats|quantit\w+|volume)\b",
        "Limits the quantity, number of users or volume permitted.",
    ),
    Indicator(
        "broad_territory", "Broad territorial scope", "restriction", 1,
        r"\bworld[- ]?wide\b(?!\s+web)|\bthroughout\s+the\s+world\b|\banywhere\s+in\s+the\s+world\b|\bin\s+any\s+(?:country|territory|jurisdiction)\b",
        "Applies across a very wide geographic area.",
    ),
    # ---------------- termination / unilateral rights ----------------
    Indicator(
        "unilateral_termination", "Termination without cause", "termination", 2,
        r"\bterminat\w*[^.;]{0,80}\b(?:for\s+convenience|for\s+any\s+(?:or\s+no\s+)?reason|without\s+cause|at\s+any\s+time)\b"
        r"|\b(?:at\s+any\s+time|for\s+convenience)\b[^.;]{0,60}\bterminat\w*|\brevoke\w*\b[^.;]{0,40}\bat\s+any\s+time\b",
        "Allows the agreement to be ended without a specific reason, which may create uncertainty.",
    ),
    Indicator(
        "immediate_termination", "Immediate termination", "termination", 1,
        r"\bterminat\w*[^.;]{0,50}\bimmediately\b|\bimmediate(?:ly)?\s+terminat\w*|\bwithout\s+(?:prior\s+|advance\s+)?notice\b",
        "Allows termination (or other action) immediately or without prior notice.",
    ),
    Indicator(
        "sole_discretion", "Sole discretion", "termination", 2,
        r"\b(?:sole|absolute|unfettered)\s+(?:and\s+(?:absolute|exclusive)\s+)?discretion\b|\bsole\s+(?:option|judg(?:e)?ment)\b",
        "Lets one party decide on its own, which may make the outcome one-sided.",
    ),
    Indicator(
        "auto_renewal", "Automatic renewal", "duration", 2,
        r"\bautomatic(?:ally)?\s+(?:be\s+)?(?:renew|extend)\w*|\bauto[- ]?renew\w*|\b(?:shall|will)\s+(?:be\s+)?(?:automatically\s+)?renew\w*"
        r"|\bsuccessive\s+(?:\w+\s+){0,3}(?:terms?|periods?|years?)\b|\bevergreen\b",
        "May cause the agreement to continue unless termination notice is provided.",
    ),
    # ---------------- duration / post-termination ----------------
    Indicator(
        "perpetual_irrevocable", "Perpetual or irrevocable", "duration", 1,
        r"\bperpetu\w*|\birrevocab\w*|\bin\s+perpetuity\b",
        "Creates rights or obligations that are permanent or cannot be withdrawn.",
    ),
    Indicator(
        "survival", "Survives termination", "duration", 2,
        r"\bsurviv\w*\s+(?:the\s+|any\s+|such\s+)?(?:expiration|termination|expiry)",
        "Some obligations continue even after the agreement ends.",
    ),
    Indicator(
        "post_termination", "Post-termination obligation", "duration", 1,
        r"\b(?:after|following|upon|subsequent\s+to)\s+(?:the\s+|any\s+|such\s+)?(?:termination|expiration|expiry)\b|\bthereafter\b",
        "Imposes obligations that apply at or after the end of the agreement.",
    ),
    # ---------------- intellectual property ----------------
    Indicator(
        "ip_assignment", "IP ownership transfer", "ip", 2,
        r"\bhereby\s+(?:irrevocably\s+)?assigns?\b|\b(?:sole\s+and\s+)?exclusive\s+property\s+of\b|\bwork\s+(?:made\s+)?for\s+hire\b"
        r"|\ball\s+right,?\s+title,?\s+and\s+interest\b[^.;]{0,80}\b(?:vest|belong|assign|own)\w*",
        "May transfer ownership of intellectual property (work product, inventions, content) to the other party.",
    ),
    Indicator(
        "joint_ip", "Joint IP ownership", "ip", 1,
        r"\bjointly\s+own\w*|\bjoint\s+(?:ownership|owners?|property)\b|\bco-?own\w*",
        "Intellectual property may be jointly owned, which can complicate later use or licensing.",
    ),
    Indicator(
        "covenant_not_to_sue", "Waiver / covenant not to sue", "ip", 2,
        r"\bcovenants?\s+not\s+to\s+(?:sue|challenge)\b|\b(?:shall|will)\s+not\s+(?:\w+\s+){0,3}(?:sue|challenge|contest|oppose)\b|\bnot\s+to\s+(?:sue|challenge|contest)\b"
        r"|\bwaives?\s+(?:any\s+and\s+all|all|any)\s+(?:rights?|claims?)\b",
        "May waive a party's right to bring claims or challenge rights (such as IP validity).",
    ),
    # ---------------- obligations / oversight ----------------
    Indicator(
        "audit_rights", "Audit / inspection rights", "obligation", 1,
        r"\baudit\w*|\binspect\w*\s+(?:and\s+(?:audit|copy)\s+)?(?:the\s+|its\s+|all\s+|any\s+)?(?:books|records|facilit\w*|premises)|\b(?:access\s+to|examine)\s+(?:the\s+|its\s+|all\s+)?(?:books|records)\b",
        "May give one party rights to inspect records or verify contractual compliance.",
    ),
    Indicator(
        "insurance_requirement", "Insurance requirement", "obligation", 1,
        r"\binsurance\b|\binsured\b",
        "Requires specific insurance coverage to be obtained and maintained, which carries a cost.",
    ),
    Indicator(
        "third_party_beneficiary", "Third-party rights", "obligation", 1,
        r"\bthird[- ]party\s+beneficiar\w*",
        "May give people outside the contract the ability to enforce some of its terms.",
        exclude_if=r"\b(?:no|not|nothing|neither)\b[^.]{0,120}\bthird[- ]party\s+beneficiar",
    ),
    # ---------------- protective (negative weights) ----------------
    Indicator(
        "liability_cap", "Liability cap present", "protective", -1,
        r"\bin\s+no\s+event\b[^.]{0,200}\b(?:exceed|liable)\b|\b(?:shall|will)\s+not\s+exceed\b|\b(?:aggregate|total|maximum|cumulative)\s+liabilit\w*|\bliabilit\w*[^.;]{0,60}\blimited\s+to\b",
        "Sets an upper limit on liability, which reduces exposure (but may also limit what can be recovered).",
    ),
    Indicator(
        "mutual", "Mutual / reciprocal wording", "protective", -1,
        r"\beither\s+party\b|\beach\s+party\b|\bneither\s+party\b|\bmutual(?:ly)?\b|\bboth\s+parties\b",
        "Applies to both parties, which makes the provision more balanced.",
    ),
    Indicator(
        "reasonable_consent", "Consent not unreasonably withheld", "protective", -1,
        r"\bnot\s+(?:to\s+)?be\s+unreasonably\s+(?:withheld|delayed|conditioned|denied)\b",
        "Consent cannot be unreasonably refused, which softens the restriction.",
    ),
    Indicator(
        "notice_period", "Defined notice period", "protective", -1,
        r"\b(?:\d+|one|two|three|five|seven|ten|fifteen|thirty|forty[- ]five|sixty|ninety|one\s+hundred\s+(?:and\s+)?eighty)\s*(?:\(\s*\d+\s*\)\s*)?(?:calendar\s+|business\s+)?days?'?\s*(?:prior\s+|advance\s+)?(?:written\s+)?notice\b",
        "Defines a notice period, which gives the other party time to react.",
    ),
    # ---------------- informational (weight 0) ----------------
    Indicator(
        "governing_law", "Governing law", "informational", 0,
        r"\bgoverned\s+by\b|\bconstrued\s+(?:and\s+enforced\s+)?(?:in\s+accordance\s+with|under)\b|\blaws\s+of\s+the\s+(?:state|province|commonwealth|republic)\b",
        "Specifies which jurisdiction's law applies - usually a routine administrative provision.",
    ),
    Indicator(
        "dispute_forum", "Dispute forum", "informational", 0,
        r"\bexclusive\s+jurisdiction\b|\bvenue\b|\barbitrat\w*",
        "Specifies where or how disputes are resolved; check whether the location is practical for you.",
    ),
]

INDICATOR_BY_ID = {i.id: i for i in INDICATORS}
RESTRICTION_IDS = {"non_compete", "non_solicit", "exclusivity", "non_disparagement", "activity_restriction"}

# ---------------------------------------------------------------------------
# Duration extraction
# ---------------------------------------------------------------------------
_WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "eighteen": 18,
    "twenty": 20, "twenty-four": 24, "thirty": 30, "thirty-six": 36, "forty-five": 45,
    "sixty": 60, "ninety": 90,
}
_DURATION_RE = re.compile(
    r"\b(?P<num>\d+(?:\.\d+)?|" + "|".join(sorted(_WORD_NUMS, key=len, reverse=True)) + r")"
    r"\s*(?:\(\s*\d+\s*\)\s*)?[- ]?(?P<unit>year|month|week|day)s?\b",
    FLAGS,
)
_UNIT_TO_YEARS = {"year": 1.0, "month": 1 / 12, "week": 1 / 52, "day": 1 / 365}


def extract_durations_years(text: str) -> List[float]:
    out = []
    for m in _DURATION_RE.finditer(text):
        raw = m.group("num").lower()
        try:
            n = float(raw) if raw[0].isdigit() else float(_WORD_NUMS[raw])
        except (KeyError, ValueError):
            continue
        if n <= 0 or n > 1000:
            continue
        out.append(round(n * _UNIT_TO_YEARS[m.group("unit").lower()], 3))
    return out


_AMOUNT_RE = re.compile(
    r"(?:US\$|\$|USD\s?|EUR\s?|€|£)\s?(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<mult>million|billion|m\b|mm\b)?",
    FLAGS,
)


def extract_max_amount(text: str) -> float:
    best = 0.0
    for m in _AMOUNT_RE.finditer(text):
        try:
            v = float(m.group("num").replace(",", ""))
        except ValueError:
            continue
        mult = (m.group("mult") or "").lower()
        if mult == "billion":
            v *= 1e9
        elif mult in {"million", "m", "mm"}:
            v *= 1e6
        best = max(best, v)
    return best


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def _snippet(text: str, m: re.Match, width: int = 40) -> str:
    s, e = max(0, m.start() - width), min(len(text), m.end() + width)
    return ("..." if s else "") + text[s:e].strip() + ("..." if e < len(text) else "")


def find_indicators(text: str) -> List[Dict]:
    """Return every matched indicator with evidence, plus duration/amount adjustments."""
    text = text or ""
    found: List[Dict] = []
    for ind in INDICATORS:
        m = ind.compiled().search(text)
        if not m:
            continue
        if ind.exclude_if and _compile(ind.exclude_if).search(text):
            continue
        if ind.requires and not _compile(ind.requires).search(text):
            continue
        found.append(
            {
                "id": ind.id,
                "name": ind.name,
                "group": ind.group,
                "weight": ind.weight,
                "explanation": ind.explanation,
                "evidence": _snippet(text, m),
                "matched": m.group(0),
                "span": [m.start(), m.end()],
            }
        )

    ids = {f["id"] for f in found}
    durations = extract_durations_years(text)
    max_years = max(durations) if durations else 0.0
    if ids & RESTRICTION_IDS and max_years >= 2:
        found.append(_synthetic("long_restriction", "Long-duration restriction", "duration", 2,
                                f"The restriction appears to last about {_fmt_years(max_years)} or more, which is a long period."))
    elif ids & RESTRICTION_IDS and max_years >= 1:
        found.append(_synthetic("restriction_duration", "Restriction duration", "duration", 1,
                                f"The restriction appears to last about {_fmt_years(max_years)}."))
    elif max_years >= 5:
        found.append(_synthetic("long_term", "Long-term commitment", "duration", 1,
                                f"Refers to a long period (about {_fmt_years(max_years)}), so obligations may last a long time."))

    if "monetary_amount" in ids and extract_max_amount(text) >= 1_000_000:
        found.append(_synthetic("large_amount", "Large monetary amount", "financial", 1,
                                "Mentions an amount of one million or more, indicating significant financial exposure."))
    return found


def _synthetic(id_, name, group, weight, explanation):
    return {"id": id_, "name": name, "group": group, "weight": weight, "explanation": explanation,
            "evidence": "", "matched": "", "span": None}


def _fmt_years(y: float) -> str:
    if y >= 1:
        v = round(y, 1)
        return f"{int(v) if v == int(v) else v} year{'s' if v != 1 else ''}"
    months = round(y * 12)
    return f"{months} month{'s' if months != 1 else ''}"


def score_to_label(score: float) -> str:
    if score <= config.ANNOTATION_LOW_MAX:
        return "Low"
    if score >= config.ANNOTATION_HIGH_MIN:
        return "High"
    return "Medium"


def annotate(text: str) -> Dict:
    """Deterministic project-defined annotation for one clause."""
    factors = find_indicators(text)
    raw = sum(f["weight"] for f in factors)
    score = max(0, raw)
    return {
        "annotation_score": score,
        "risk_label": score_to_label(score),
        "factors": factors,
        "risk_factor_names": [f["name"] for f in factors if f["weight"] > 0],
    }


def risk_factor_list(factors: List[Dict]) -> List[Dict]:
    return [f for f in factors if f["weight"] > 0]


def protective_factor_list(factors: List[Dict]) -> List[Dict]:
    return [f for f in factors if f["weight"] < 0]


def plain_language_explanation(factors: List[Dict], label: str, clause_type: str = "") -> List[str]:
    """Short bullet sentences in simple language (screening, not legal advice)."""
    bullets: List[str] = []
    desc = CLAUSE_TYPE_DESCRIPTIONS.get(clause_type)
    if desc:
        bullets.append(desc)
    risky = sorted(risk_factor_list(factors), key=lambda f: -f["weight"])
    for f in risky[:5]:
        bullets.append(f["explanation"])
    for f in protective_factor_list(factors)[:2]:
        bullets.append("Balancing factor: " + f["explanation"])
    if not risky:
        info = [f for f in factors if f["weight"] == 0]
        if info:
            bullets.append(info[0]["explanation"])
        bullets.append("No strong risk indicators were found in the wording of this clause.")
    if label == "High":
        bullets.append("This clause may require careful review before signing.")
    elif label == "Medium":
        bullets.append("This clause may warrant review to confirm the terms are acceptable.")
    # de-duplicate while preserving order
    seen, out = set(), []
    for b in bullets:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def short_explanation(factors: List[Dict], label: str) -> str:
    risky = sorted(risk_factor_list(factors), key=lambda f: -f["weight"])
    if risky:
        return risky[0]["explanation"]
    if label == "Low":
        return "No strong risk indicators were found in the wording of this clause."
    return "The model flagged this clause; no specific rule-based indicator matched."


# Plain descriptions of what common clause types do (used only for explanations).
CLAUSE_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "Non-Compete": "This is a non-compete clause: it limits competing business activity.",
    "Exclusivity": "This clause creates an exclusive relationship between the parties.",
    "Uncapped Liability": "This clause concerns liability that may not be limited.",
    "Cap On Liability": "This clause limits how much a party can be held liable for.",
    "Liquidated Damages": "This clause fixes an amount payable when certain events occur.",
    "Renewal Term": "This clause controls how and when the agreement renews.",
    "Notice Period To Terminate Renewal": "This clause sets how much notice is needed to stop a renewal.",
    "Termination For Convenience": "This clause allows the agreement to be ended without a specific breach.",
    "Termination": "This clause explains how the agreement can be ended.",
    "Governing Law": "This clause chooses which jurisdiction's law governs the agreement.",
    "Anti-Assignment": "This clause limits transferring the agreement to someone else.",
    "Assignment": "This clause limits transferring the agreement to someone else.",
    "Audit Rights": "This clause allows one party to check the other's records or compliance.",
    "Insurance": "This clause requires insurance coverage to be maintained.",
    "Indemnification": "This clause decides who pays for certain losses or claims.",
    "Confidentiality": "This clause restricts how confidential information may be used or shared.",
    "Payment Terms": "This clause sets out payments, fees or charges.",
    "Ip Ownership Assignment": "This clause decides who owns intellectual property created under the agreement.",
    "License Grant": "This clause grants permission to use intellectual property.",
    "Change Of Control": "This clause deals with what happens if a party's ownership changes.",
    "Minimum Commitment": "This clause sets a minimum amount that must be bought or paid.",
    "No-Solicit Of Employees": "This clause limits hiring or approaching the other party's staff.",
    "No-Solicit Of Customers": "This clause limits approaching the other party's customers.",
    "Post-Termination Services": "This clause sets duties that continue after the agreement ends.",
    "Warranty Duration": "This clause sets how long a warranty lasts.",
    "Maintenance & Repairs": "This clause allocates responsibility for maintenance and repair costs.",
    "Security Deposit": "This clause governs a deposit held as security.",
    "Rent & Payment": "This clause sets rent or payment amounts and timing.",
}
