#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

from pypdf import PdfReader

PLANETS = [
    "sun",
    "moon",
    "mars",
    "mercury",
    "jupiter",
    "venus",
    "saturn",
    "rahu",
    "ketu",
]
SIGNS = [
    "aries",
    "taurus",
    "gemini",
    "cancer",
    "leo",
    "virgo",
    "libra",
    "scorpio",
    "sagittarius",
    "capricorn",
    "aquarius",
    "pisces",
]
SIGN_LORD = {
    "aries": "mars",
    "taurus": "venus",
    "gemini": "mercury",
    "cancer": "moon",
    "leo": "sun",
    "virgo": "mercury",
    "libra": "venus",
    "scorpio": "mars",
    "sagittarius": "jupiter",
    "capricorn": "saturn",
    "aquarius": "saturn",
    "pisces": "jupiter",
}
DOMAIN_KEYWORDS = {
    "personality": ["personality", "character", "temperament", "self", "nature"],
    "career": ["career", "profession", "job", "work", "authority", "status"],
    "finance": ["wealth", "money", "finance", "income", "assets", "gain"],
    "marriage": ["marriage", "spouse", "wife", "husband", "relationship"],
    "health": ["health", "disease", "illness", "injury", "vitality"],
    "spirituality": ["spiritual", "moksha", "dharma", "sadhana", "devotion"],
    "social_status": ["fame", "reputation", "social", "public", "honor"],
    "education": ["education", "study", "learning", "knowledge", "intelligence"],
    "travel": ["travel", "foreign", "journey", "abroad", "pilgrimage"],
    "obstacles": ["obstacle", "delay", "loss", "struggle", "failure", "debt"],
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page_start: int
    page_end: int
    text: str


@dataclass(frozen=True)
class Rule:
    rule_id: str
    rule_type: str
    condition: str
    effect: str
    exceptions: str
    page_reference: str


def configure_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def phase_checkpoint(name: str) -> None:
    logging.info("PHASE_CHECKPOINT | %s", name)


def extract_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        pages.append((idx, page.extract_text() or ""))
    return pages


def chunk_pages(pages: List[Tuple[int, str]], size: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for i in range(0, len(pages), size):
        block = pages[i : i + size]
        start = block[0][0]
        end = block[-1][0]
        text = "\n\n".join(f"[PAGE {p}]\n{t}" for p, t in block)
        chunks.append(Chunk(chunk_id=f"CHUNK_{len(chunks)+1:04d}", page_start=start, page_end=end, text=text))
    return chunks


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) >= 25]


def detect_rule_type(sentence: str) -> str:
    s = sentence.lower()
    if "dasha" in s:
        return "dasha"
    if "yoga" in s:
        return "yoga"
    if "house" in s or re.search(r"\b[1-9](st|nd|rd|th)?\s+house\b", s):
        return "house"
    return "planet"


def parse_condition_effect(sentence: str) -> Tuple[str, str, str]:
    txt = re.sub(r"\s+", " ", sentence).strip()
    exception = ""
    m_exc = re.search(r"\b(except|unless|however)\b(.+)$", txt, flags=re.IGNORECASE)
    if m_exc:
        exception = m_exc.group(0).strip()
        txt = txt[: m_exc.start()].strip()

    low = txt.lower()
    for marker in [" then ", " gives ", " indicates ", " leads to ", " results in "]:
        if marker in low:
            idx = low.index(marker)
            return txt[:idx].strip(), txt[idx + len(marker) :].strip(), exception

    if ":" in txt:
        left, right = txt.split(":", 1)
        return left.strip(), right.strip(), exception

    return txt, txt, exception


def extract_textbook_rules(chunks: List[Chunk]) -> List[Rule]:
    phase_checkpoint("PHASE_1_TEXTBOOK_EXTRACTION")
    rules_raw = []
    rule_markers = [
        "if",
        "when",
        "planet",
        "house",
        "lord",
        "dasha",
        "yoga",
        "nakshatra",
        "placement",
        "gives",
        "results",
        "indicates",
    ]

    for chunk in chunks:
        for line in split_sentences(chunk.text):
            line_low = line.lower()
            if any(m in line_low for m in rule_markers):
                condition, effect, exceptions = parse_condition_effect(line)
                rules_raw.append((condition, effect, exceptions, chunk.page_start, chunk.page_end))

    dedup: Dict[str, Tuple[str, str, str, int, int]] = {}
    for condition, effect, exceptions, p1, p2 in rules_raw:
        k = normalize_text(f"{condition}|{effect}|{exceptions}")
        if k not in dedup:
            dedup[k] = (condition, effect, exceptions, p1, p2)

    rules: List[Rule] = []
    for i, (_, (condition, effect, exceptions, p1, p2)) in enumerate(sorted(dedup.items()), start=1):
        sentence = f"{condition} {effect}"
        rules.append(
            Rule(
                rule_id=f"R{i:05d}",
                rule_type=detect_rule_type(sentence),
                condition=condition,
                effect=effect,
                exceptions=exceptions,
                page_reference=f"pp.{p1}-{p2}",
            )
        )
    return rules


def parse_kundali_structures(chunks: List[Chunk]) -> Dict:
    phase_checkpoint("PHASE_1_KUNDALI_EXTRACTION")
    text = "\n".join(c.text for c in chunks)
    text_low = text.lower()

    lagna = None
    for sign in SIGNS:
        if re.search(rf"\blagna\b[^\n]{{0,40}}\b{sign}\b", text_low):
            lagna = sign
            break

    planets = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for ln in lines:
        low = ln.lower()
        for p in PLANETS:
            if re.search(rf"\b{p}\b", low):
                sign_match = next((s for s in SIGNS if re.search(rf"\b{s}\b", low)), "")
                deg_match = re.search(r"(\d{1,2}(?:\.\d{1,2})?)\s*°?", low)
                house_match = re.search(r"\bhouse\s*(\d{1,2})\b|\b(\d{1,2})\s*(?:st|nd|rd|th)?\s*house\b", low)
                nak_match = re.search(r"nakshatra\s*[:\-]?\s*([a-z]+)", low)
                pada_match = re.search(r"pada\s*[:\-]?\s*(\d)", low)
                retro = bool(re.search(r"retro|rx", low))
                combust = bool(re.search(r"combust", low))
                planets.append(
                    {
                        "planet": p,
                        "source_line": ln,
                        "sign": sign_match,
                        "degree": deg_match.group(1) if deg_match else "",
                        "house": (house_match.group(1) or house_match.group(2)) if house_match else "",
                        "nakshatra": nak_match.group(1) if nak_match else "",
                        "pada": pada_match.group(1) if pada_match else "",
                        "retrograde": retro,
                        "combustion": combust,
                    }
                )
                break

    dasha_lines = [l for l in lines if re.search(r"\bdasha\b|\bmahadasha\b|\bantardasha\b", l.lower())]

    # Deduplicate planets deterministically
    uniq = {}
    for p in planets:
        key = normalize_text(f"{p['planet']}|{p['sign']}|{p['house']}|{p['degree']}")
        if key not in uniq:
            uniq[key] = p

    d1 = sorted(uniq.values(), key=lambda x: (x["planet"], x["house"], x["sign"], x["degree"]))
    d9 = [p for p in d1 if "d9" in p["source_line"].lower() or "navamsa" in p["source_line"].lower()]

    lordships = {}
    if lagna and lagna in SIGNS:
        idx = SIGNS.index(lagna)
        for house_num in range(1, 13):
            sign = SIGNS[(idx + house_num - 1) % 12]
            lordships[str(house_num)] = {"sign": sign, "lord": SIGN_LORD[sign]}

    return {
        "lagna": lagna or "",
        "planetary_positions": d1,
        "house_placements": [p for p in d1 if p.get("house")],
        "nakshatra_pada": [
            {"planet": p["planet"], "nakshatra": p["nakshatra"], "pada": p["pada"]}
            for p in d1
            if p.get("nakshatra") or p.get("pada")
        ],
        "retrograde_combustion": [
            {"planet": p["planet"], "retrograde": p["retrograde"], "combustion": p["combustion"]}
            for p in d1
            if p["retrograde"] or p["combustion"]
        ],
        "dasha_data": dasha_lines,
        "structured_chart": {
            "D1": d1,
            "D9": d9,
            "lordships": lordships,
        },
    }


def extract_rule_tokens(text: str) -> Dict[str, List[str]]:
    t = text.lower()
    planets = [p for p in PLANETS if re.search(rf"\b{p}\b", t)]
    signs = [s for s in SIGNS if re.search(rf"\b{s}\b", t)]
    houses = re.findall(r"\b(1[0-2]|[1-9])\s*(?:st|nd|rd|th)?\s*house\b|\bhouse\s*(1[0-2]|[1-9])\b", t)
    house_vals = sorted({h1 or h2 for h1, h2 in houses if h1 or h2})
    return {"planets": planets, "signs": signs, "houses": house_vals}


def map_rules(kundali: Dict, rules: List[Rule]) -> List[Dict]:
    phase_checkpoint("PHASE_4_RULE_MAPPING")
    chart = kundali["structured_chart"]["D1"]
    chart_planets = {p["planet"] for p in chart}
    chart_signs = {p["sign"] for p in chart if p.get("sign")}
    chart_houses = {str(p["house"]) for p in chart if str(p.get("house", "")).isdigit()}

    matched = []
    for rule in rules:
        tokens = extract_rule_tokens(f"{rule.condition} {rule.effect}")
        if not tokens["planets"] and not tokens["signs"] and not tokens["houses"]:
            continue
        cond_ok = all(p in chart_planets for p in tokens["planets"]) and all(
            s in chart_signs for s in tokens["signs"]
        ) and all(h in chart_houses for h in tokens["houses"])

        matched.append(
            {
                "rule": asdict(rule),
                "tokens": tokens,
                "condition_satisfied": cond_ok,
            }
        )
    return matched


def strict_validate(matched_rules: List[Dict]) -> List[Dict]:
    phase_checkpoint("PHASE_5_STRICT_VALIDATION")
    validated = [r for r in matched_rules if r["condition_satisfied"]]
    return sorted(validated, key=lambda x: x["rule"]["rule_id"])


def cross_check_strength(validated: List[Dict]) -> List[Dict]:
    phase_checkpoint("PHASE_6_CROSS_CHECKING")
    counts: Dict[str, int] = {}
    for r in validated:
        key = normalize_text(f"{r['rule']['condition']}|{r['rule']['effect']}")
        counts[key] = counts.get(key, 0) + 1

    out = []
    for r in validated:
        key = normalize_text(f"{r['rule']['condition']}|{r['rule']['effect']}")
        c = counts[key]
        strength = "STRONG" if c >= 2 else "MEDIUM"
        r2 = dict(r)
        r2["strength"] = strength
        out.append(r2)
    return out


def rule_quality_score(text: str) -> int:
    t = text.lower()
    score = 0
    if "exalt" in t:
        score += 5
    if "own sign" in t:
        score += 4
    if "friendly" in t:
        score += 3
    if "neutral" in t:
        score += 2
    if "debil" in t:
        score -= 3
    if "kendra" in t or "trikona" in t:
        score += 2
    if "dusthana" in t:
        score -= 2
    if "strong dasha" in t:
        score += 3
    if "weak" in t:
        score -= 1
    return score


def contradiction_resolution(rules: List[Dict]) -> List[Dict]:
    phase_checkpoint("PHASE_7_CONTRADICTION_RESOLUTION")
    grouped: Dict[str, Dict] = {}
    for r in rules:
        signature = normalize_text("|".join([
            ",".join(r["tokens"]["planets"]),
            ",".join(r["tokens"]["signs"]),
            ",".join(r["tokens"]["houses"]),
        ]))
        score = rule_quality_score(f"{r['rule']['condition']} {r['rule']['effect']}")
        item = dict(r)
        item["quality_score"] = score
        if signature not in grouped or score > grouped[signature]["quality_score"]:
            grouped[signature] = item
    return sorted(grouped.values(), key=lambda x: (x["rule"]["rule_id"], -x["quality_score"]))


def assign_priorities(final_rules: List[Dict]) -> List[Dict]:
    phase_checkpoint("PHASE_8_PRIORITY_ASSIGNMENT")
    ordered = sorted(final_rules, key=lambda r: (r.get("strength") != "STRONG", -r.get("quality_score", 0), r["rule"]["rule_id"]))
    high_cap = max(1, int(len(ordered) * 0.40)) if ordered else 0
    prioritized = []
    for i, r in enumerate(ordered):
        item = dict(r)
        if i < high_cap:
            item["priority"] = "HIGH"
        else:
            item["priority"] = "MEDIUM"
        prioritized.append(item)
    return prioritized


def map_domain(rule: Dict) -> str:
    t = normalize_text(f"{rule['rule']['condition']} {rule['rule']['effect']}")
    for domain, kws in DOMAIN_KEYWORDS.items():
        if any(k in t for k in kws):
            return domain
    return "personality"


def pattern_intelligence(rules: List[Dict]) -> Dict[str, List[Dict]]:
    phase_checkpoint("PHASE_9_PATTERN_INTELLIGENCE")
    patterns: Dict[str, List[Dict]] = {k: [] for k in DOMAIN_KEYWORDS}
    for r in rules:
        patterns[map_domain(r)].append(r)
    return patterns


def build_predictions(rules: List[Dict]) -> List[Dict]:
    phase_checkpoint("PHASE_10_FULL_PREDICTION_ENGINE")
    preds = []
    for r in rules:
        planets = ", ".join(r["tokens"]["planets"]) or "n/a"
        houses = ", ".join(r["tokens"]["houses"]) or "n/a"
        preds.append(
            {
                "rule_id": r["rule"]["rule_id"],
                "condition": r["rule"]["condition"],
                "planet_house": f"{planets} | houses: {houses}",
                "textbook_basis": f"{r['rule']['effect']} ({r['rule']['page_reference']})",
                "final_interpretation": r["rule"]["effect"],
                "priority": r["priority"],
                "domain": map_domain(r),
            }
        )
    return preds


def top_40_facts(predictions: List[Dict]) -> List[str]:
    phase_checkpoint("PHASE_11_ULTRA_COMPRESSION")
    ranked = sorted(predictions, key=lambda p: (p["priority"] != "HIGH", p["rule_id"]))
    out = []
    for p in ranked[:40]:
        out.append(f"{p['rule_id']}: {p['final_interpretation']}")
    return out


def html_escape(x: str) -> str:
    return html.escape(x if x is not None else "")


def render_table(headers: List[str], rows: List[List[str]]) -> str:
    head = "".join(f"<th>{html_escape(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{html_escape(str(c))}</td>" for c in row) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def export_html(output_path: Path, result: Dict) -> None:
    phase_checkpoint("PHASE_12_HTML_EXPORT")
    chart = result["structured_chart"]
    rules = result["final_rule_set"]
    predictions = result["predictions"]
    patterns = result["domain_patterns"]
    top40 = result["top_40_facts"]

    html_doc = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Kundali Comparative Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:20px;color:#111;}h1,h2{margin-top:24px;}table{border-collapse:collapse;width:100%;margin:10px 0;font-size:12px;}th,td{border:1px solid #bbb;padding:6px;vertical-align:top;}th{background:#f2f2f2;}@page{size:A4;margin:15mm;} .fact{margin:4px 0;}</style>",
        "</head><body>",
        "<h1>Deterministic Comparative Astro Pipeline Report</h1>",
        "<h2>1. Chart Summary</h2>",
        f"<p><b>Lagna:</b> {html_escape(chart.get('lagna', ''))}</p>",
        render_table(
            ["Planet", "Sign", "Degree", "House", "Nakshatra", "Pada", "Retrograde", "Combustion"],
            [
                [
                    p.get("planet", ""),
                    p.get("sign", ""),
                    p.get("degree", ""),
                    p.get("house", ""),
                    p.get("nakshatra", ""),
                    p.get("pada", ""),
                    str(p.get("retrograde", False)),
                    str(p.get("combustion", False)),
                ]
                for p in chart.get("planetary_positions", [])
            ],
        ),
        "<h2>2. Rule Mapping Table</h2>",
        render_table(
            ["Rule ID", "Type", "Condition", "Effect", "Page", "Strength", "Priority"],
            [
                [
                    r["rule"]["rule_id"],
                    r["rule"]["rule_type"],
                    r["rule"]["condition"],
                    r["rule"]["effect"],
                    r["rule"]["page_reference"],
                    r.get("strength", ""),
                    r.get("priority", ""),
                ]
                for r in rules
            ],
        ),
        "<h2>3. Validated Rule Set</h2>",
        render_table(
            ["Rule ID", "Condition", "Effect", "Exceptions", "Page Ref"],
            [[r["rule"]["rule_id"], r["rule"]["condition"], r["rule"]["effect"], r["rule"]["exceptions"], r["rule"]["page_reference"]] for r in rules],
        ),
        "<h2>4. Contradiction Analysis</h2>",
        render_table(
            ["Rule ID", "Quality Score", "Resolution Basis"],
            [[r["rule"]["rule_id"], str(r.get("quality_score", 0)), "exalted>own>friendly>neutral>debilitated; kendra/trikona>dusthana; dasha strength"] for r in rules],
        ),
        "<h2>5. Domain-wise Predictions</h2>",
        render_table(
            ["Rule ID", "Domain", "Condition", "Planet+House", "Textbook Basis", "Final Interpretation", "Priority"],
            [[p["rule_id"], p["domain"], p["condition"], p["planet_house"], p["textbook_basis"], p["final_interpretation"], p["priority"]] for p in predictions],
        ),
        "<h2>6. Pattern Intelligence</h2>",
    ]

    for domain, items in patterns.items():
        html_doc.append(f"<h3>{html_escape(domain)}</h3>")
        html_doc.append(render_table(["Rule ID", "Effect", "Priority"], [[r["rule"]["rule_id"], r["rule"]["effect"], r.get("priority", "")] for r in items]))

    html_doc.append("<h2>7. Top 40 Facts</h2>")
    for fact in top40:
        html_doc.append(f"<div class='fact'>{html_escape(fact)}</div>")
    html_doc.append("</body></html>")

    output_path.write_text("\n".join(html_doc), encoding="utf-8")


def run_pipeline(kundali_pdf: Path, textbook_pdf: Path, output_html: Path, output_json: Path, chunk_size: int, log_file: Path) -> Dict:
    configure_logging(log_file)
    phase_checkpoint("START")

    phase_checkpoint("PHASE_1_FULL_EXTRACTION")
    kundali_pages = extract_pdf_pages(kundali_pdf)
    textbook_pages = extract_pdf_pages(textbook_pdf)
    kundali_chunks = chunk_pages(kundali_pages, chunk_size)
    textbook_chunks = chunk_pages(textbook_pages, chunk_size)

    structured_chart = parse_kundali_structures(kundali_chunks)
    textbook_rules = extract_textbook_rules(textbook_chunks)

    phase_checkpoint("PHASE_2_TEXTBOOK_RULE_INDEXING")
    rule_database = [asdict(r) for r in textbook_rules]

    phase_checkpoint("PHASE_3_KUNDALI_STRUCTURING")
    # already in structured_chart

    matched = map_rules(structured_chart, textbook_rules)
    validated = strict_validate(matched)
    crossed = cross_check_strength(validated)
    resolved = contradiction_resolution(crossed)
    prioritized = assign_priorities(resolved)
    domain_patterns = pattern_intelligence(prioritized)
    predictions = build_predictions(prioritized)
    facts = top_40_facts(predictions)

    result = {
        "structured_kundali_data": {
            "chunks": [asdict(c) for c in kundali_chunks],
            "merged": structured_chart,
        },
        "structured_textbook_rules": {
            "chunks": [asdict(c) for c in textbook_chunks],
            "merged": rule_database,
        },
        "rule_database": rule_database,
        "structured_chart": structured_chart,
        "matched_rules": matched,
        "validated_rules": validated,
        "final_rule_set": prioritized,
        "domain_patterns": {k: v for k, v in domain_patterns.items()},
        "predictions": predictions,
        "top_40_facts": facts,
    }

    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    export_html(output_html, result)
    phase_checkpoint("END")
    return result


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic comparative astro pipeline")
    parser.add_argument("--kundali-pdf", type=Path, required=True)
    parser.add_argument("--textbook-pdf", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, default=Path("pipeline.log"))
    parser.add_argument("--chunk-size", type=int, default=7)
    return parser


def main() -> None:
    args = build_cli().parse_args()
    run_pipeline(
        kundali_pdf=args.kundali_pdf,
        textbook_pdf=args.textbook_pdf,
        output_html=args.output_html,
        output_json=args.output_json,
        chunk_size=args.chunk_size,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
