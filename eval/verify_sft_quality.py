"""
SFT trajectory annotation quality verifier.

Analyzes 100 randomly sampled trajectories from train_sft_coldstart.parquet:
  1. Crop bbox quality  (image_search_tool bbox_2d argument)
  2. Useful tag quality (<useful>[...]</useful> annotations)

Run:
    python3 eval/verify_sft_quality.py
"""

import re
import json
import random
from collections import Counter

import pandas as pd

PARQUET_PATH = "/mnt/sh/mmvision/home/jonahli/data_agent/sft/coldstart/train_sft_coldstart.parquet"
SAMPLE_SEED = 42
SAMPLE_N = 100

# ── bbox thresholds ────────────────────────────────────────────────────────────
MIN_SIDE_PX = 50          # width or height < 50px → degenerate
LARGE_AREA_THRESH = 0.80  # covers >80% of 1000×1000 → "no targeting"
WHOLE_IMAGE_BBOX = [0, 0, 1000, 1000]


# ── helpers ───────────────────────────────────────────────────────────────────

def get_text(msg) -> str:
    c = msg.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for item in c:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(c)


# ── bbox extraction ────────────────────────────────────────────────────────────

def extract_bboxes(messages):
    """
    Extract image_search_tool calls from <tool_call>...</tool_call> blocks.
    Returns list of dicts: bbox, goal, think_text, msg_idx.
    """
    results = []
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        text = get_text(msg)

        # Tool calls are wrapped in <tool_call>...</tool_call>
        for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
            raw = m.group(1).strip()
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if obj.get("name") != "image_search_tool":
                continue
            args = obj.get("arguments", {})
            bbox = args.get("bbox_2d") or args.get("bbox")
            if bbox is None:
                continue
            goal = args.get("goal", "")

            # Extract <think> block from same message
            think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            think_text = think_match.group(1).strip() if think_match else ""

            results.append(dict(
                bbox=bbox, goal=goal, think=think_text, msg_idx=idx
            ))
    return results


# ── useful-tag extraction ─────────────────────────────────────────────────────

def extract_useful_tags(messages):
    """
    Find <useful>[...]</useful> tags in assistant messages, paired with the
    most recent image-search tool_response.
    Returns list of dicts: indices, search_results_text, msg_idx.
    """
    results = []
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        text = get_text(msg)
        # Skip the system message which contains the format example
        if idx == 0:
            continue

        for m in re.finditer(r"<useful>\s*(\[.*?\])\s*</useful>", text, re.DOTALL):
            try:
                indices = json.loads(m.group(1))
                if not isinstance(indices, list):
                    continue
            except Exception:
                continue

            # Find most recent image-search tool_response before this message
            sr_text = ""
            for prev_idx in range(idx - 1, -1, -1):
                prev = messages[prev_idx]
                if prev.get("role") == "user":
                    pt = get_text(prev)
                    if "Image search results" in pt:
                        sr_text = pt
                        break

            results.append(dict(
                indices=indices, search_results=sr_text, msg_idx=idx
            ))
    return results


# ── bbox judgment ─────────────────────────────────────────────────────────────

def normalize_bbox(bbox):
    """Normalize fractional (0-1) bboxes to pixel (0-1000) space."""
    if max(bbox) <= 1.0:
        return [v * 1000 for v in bbox]
    return list(bbox)


def judge_bbox(bbox_raw, think, goal):
    """
    Returns (verdict, reason) where verdict in {GOOD, BAD, UNCERTAIN}.

    Criteria:
      BAD:
        - Whole-image or near-whole-image bbox (no targeting)
        - Degenerate bbox (width or height < MIN_SIDE_PX)
        - Zero/negative dimensions
      UNCERTAIN:
        - Large bbox (>60% but <=80%) with no think rationale for region choice
      GOOD:
        - Targeted crop with reasonable size
    """
    if not bbox_raw or len(bbox_raw) < 4:
        return "BAD", "missing or malformed bbox"

    bbox = normalize_bbox(bbox_raw[:4])
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return "BAD", f"zero/negative dimensions w={w:.0f} h={h:.0f}"

    if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
        return "BAD", f"degenerate bbox w={w:.0f} h={h:.0f} (< {MIN_SIDE_PX}px)"

    area_frac = (w * h) / (1000.0 * 1000.0)

    # Whole-image: [0,0,1000,1000] or very close
    if area_frac > LARGE_AREA_THRESH:
        # Check think: does the model acknowledge it's using the full image?
        think_lower = think.lower()
        full_img_acknowledged = any(kw in think_lower for kw in [
            "full image", "whole image", "entire image",
            "no distinctive landmark", "no clear landmark",
            "no specific region", "no architectural", "no visible landmark",
            "owl", "no geo", "no specific", "entire scene",
        ])
        if full_img_acknowledged:
            return "UNCERTAIN", (
                f"full/near-full bbox ({area_frac:.0%}) — model acknowledged"
                " no specific landmark; acceptable for non-landmark images"
            )
        else:
            return "BAD", (
                f"whole-image bbox ({area_frac:.0%}) without reasoning for"
                " why no targeted crop was chosen"
            )

    # Bbox is somewhat targeted (area < 80%)
    if area_frac > 0.60:
        # Large but not whole-image — check for rationale
        think_lower = think.lower()
        has_region_rationale = any(kw in think_lower for kw in [
            "region", "crop", "bbox", "bounding", "area",
            "landmark", "building", "temple", "sign", "text", "logo",
            "tower", "bridge", "street", "upper", "lower", "left", "right",
            "center", "middle", "bottom", "top", "section", "portion",
        ])
        if not has_region_rationale:
            return "UNCERTAIN", (
                f"large bbox ({area_frac:.0%}) with no region rationale in <think>"
            )

    return "GOOD", f"targeted crop area={area_frac:.1%} w={w:.0f}×h={h:.0f}"


# ── useful-tag judgment ────────────────────────────────────────────────────────

def parse_search_results(sr_text):
    """
    Parse numbered result entries from tool_response.
    Returns list of dicts: idx, title, source, link.
    """
    results = []
    for m in re.finditer(
        r"\[(\d+)\]\s+(.*?)\n\s+Source:\s*(.*?)\n\s+Link:\s*(.*?)(?=\n\[|\Z)",
        sr_text, re.DOTALL
    ):
        results.append(dict(
            idx=int(m.group(1)),
            title=m.group(2).strip(),
            source=m.group(3).strip(),
            link=m.group(4).strip(),
        ))
    return results


def is_definitely_geo_informative(result):
    """
    Strict test: is this result DEFINITELY geo-informative?
    Uses explicit geo signals: place names in title, geo-source domains.
    """
    title = result["title"].lower()
    source = result["source"].lower()
    link = result["link"].lower()
    combined = title + " " + source + " " + link

    # Strong geo signals in title
    title_geo = any(kw in title for kw in [
        "wikipedia", "wikidata", "wikimedia", "tripadvisor", "atlas",
        "national park", "monument", "museum", "palace", "castle",
        "bridge", "cathedral", "temple", "church", "mosque",
        "city", "country", "town", "village", "region", "province",
        " street", "square", "district", "hotel", "hostel",
        "visit", "tourist", "travel", "tour", "guide",
        "located in", "in athens", "in paris", "in london", "in rome",
        "in sydney", "in tokyo", "in berlin", "in vienna",
    ])

    # Strong geo domains
    geo_domains = any(d in link for d in [
        "wikipedia.org", "tripadvisor.com", "wikidata.org",
        "wikimedia.org", "naturalatlas.com", "humbo.com",
        "geohack", "openstreetmap", "maps.google",
    ])

    # Search result is about a specific named place
    place_pattern = re.search(
        r"\b(national park|state park|nature reserve|world heritage|"
        r"acropolis|parthenon|eiffel|louvre|colosseum|"
        r"carlsbad|carlsbad caverns|lions gate|chateau|"
        r"sydney|paris|london|rome|berlin|athens|tokyo|"
        r"ottawa|vancouver|melbourne|cairo|istanbul|"
        r"museum|gallery|cathedral|basilica|abbey|monastery)\b",
        combined
    )

    return title_geo or geo_domains or (place_pattern is not None)


def is_likely_not_geo(result):
    """
    Returns True if this result is clearly NOT geo-informative.
    """
    title = result["title"].lower()
    combined = (title + " " + result["source"] + " " + result["link"]).lower()

    # Generic technical content
    non_geo = any(kw in title for kw in [
        "reversible lane", "stock photo", "hi-res stock", "alamy.com",
        "clip art", "vector", "illustration", "royalty free",
        "project description", "drone captures aerial",  # generic
        "all you should know before going",  # usually OK but check
    ])

    return non_geo


def judge_useful_tags(indices, search_results_text):
    """
    Returns (verdict, reason, details).

    Criteria:
      BAD:
        - Tagged [] (empty) when search results clearly identify the location
        - Tagged results that are definitely non-geo (confirmed not useful)
        - Missed >3 definitely-geo-informative results (severe false negatives)
      UNCERTAIN:
        - Tool returned error / no results
        - 1-3 missed geo results (minor false negatives)
        - Tagged some questionable results (minor false positives)
      GOOD:
        - Tagged indices contain the key geo-informative results
        - Missing at most 1-2 marginally-geo results
    """
    if not search_results_text:
        return "UNCERTAIN", "no search results text found to verify", ""

    # Check for failed tool calls
    sr_lower = search_results_text.lower()
    if "image search failed" in sr_lower or "api error" in sr_lower:
        if "image search results" not in sr_lower:
            return "UNCERTAIN", "tool returned error — no results to verify", ""

    parsed = parse_search_results(search_results_text)
    if not parsed:
        return "UNCERTAIN", "could not parse search results (0 entries found)", ""

    total = len(parsed)
    def_geo = [r for r in parsed if is_definitely_geo_informative(r)]
    def_geo_idx = {r["idx"] for r in def_geo}
    tagged_set = set(indices)

    # ── Case 1: empty tag when location is clear from results ─────────────────
    if not indices:
        if len(def_geo) >= 2:
            ex = [f'[{r["idx"]}] {r["title"][:55]}' for r in def_geo[:3]]
            return "BAD", f"tagged [] but {len(def_geo)}/{total} clearly geo results exist", str(ex)
        elif len(def_geo) == 1:
            ex = f'[{def_geo[0]["idx"]}] {def_geo[0]["title"][:55]}'
            return "UNCERTAIN", f"tagged [] but 1 geo result exists: {ex}", ""
        else:
            return "GOOD", "correctly tagged [] (no geo-informative results)", ""

    # ── Case 2: false negatives (missed clearly geo results) ──────────────────
    missed_geo_idx = def_geo_idx - tagged_set
    missed_geo = [r for r in def_geo if r["idx"] in missed_geo_idx]

    # ── Case 3: false positives (tagged clearly non-geo results) ──────────────
    tagged_non_geo = [r for r in parsed
                      if r["idx"] in tagged_set and is_likely_not_geo(r)]

    if len(missed_geo) >= 4 and len(tagged_set) <= 2:
        ex = [f'[{r["idx"]}] {r["title"][:55]}' for r in missed_geo[:3]]
        return "BAD", f"missed {len(missed_geo)} clearly-geo results while tagging only {len(tagged_set)}", str(ex)

    if len(tagged_non_geo) >= 3:
        ex = [f'[{r["idx"]}] {r["title"][:55]}' for r in tagged_non_geo[:3]]
        return "BAD", f"tagged {len(tagged_non_geo)} clearly non-geo results", str(ex)

    if len(missed_geo) >= 2:
        ex = [f'[{r["idx"]}] {r["title"][:55]}' for r in missed_geo[:2]]
        return "UNCERTAIN", (
            f"minor false negatives: missed {len(missed_geo)} geo results"
            f", tagged {len(tagged_set)}/{total}"
        ), str(ex)

    if len(tagged_non_geo) >= 1:
        ex = f'[{tagged_non_geo[0]["idx"]}] {tagged_non_geo[0]["title"][:55]}'
        return "UNCERTAIN", (
            f"minor false positive: {ex}"
        ), ""

    return "GOOD", (
        f"tagged {len(tagged_set)}/{total} results; "
        f"{len(missed_geo)} missed geo results"
    ), ""


# ── main analysis ──────────────────────────────────────────────────────────────

def analyze():
    print("Loading parquet...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  Total rows: {len(df)}")

    rng = random.Random(SAMPLE_SEED)
    sampled_indices = sorted(rng.sample(range(len(df)), SAMPLE_N))
    sample = df.iloc[sampled_indices]
    print(f"  Sampled {len(sample)} rows (seed={SAMPLE_SEED})\n")

    bbox_records = []
    useful_records = []
    rows_with_image_search = 0
    rows_with_useful_tags = 0

    for rank, (df_idx, row) in enumerate(sample.iterrows()):
        msgs = row["messages"]
        bboxes = extract_bboxes(msgs)
        useful_list = extract_useful_tags(msgs)

        if bboxes:
            rows_with_image_search += 1
        if useful_list:
            rows_with_useful_tags += 1

        for b in bboxes:
            v, reason = judge_bbox(b["bbox"], b["think"], b["goal"])
            bbox_records.append(dict(
                rank=rank, df_idx=df_idx,
                bbox=b["bbox"], goal=b["goal"],
                think_snippet=b["think"][:150].replace("\n", " "),
                verdict=v, reason=reason,
            ))

        for u in useful_list:
            v, reason, detail = judge_useful_tags(u["indices"], u["search_results"])
            useful_records.append(dict(
                rank=rank, df_idx=df_idx,
                indices=u["indices"],
                sr_snippet=(u["search_results"][u["search_results"].find("[1]"):
                             u["search_results"].find("[1]")+300]
                            if "[1]" in u["search_results"] else ""),
                verdict=v, reason=reason, detail=detail,
            ))

    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 72)
    print("SFT TRAJECTORY ANNOTATION QUALITY REPORT")
    print(f"Dataset : {PARQUET_PATH.split('/')[-1]}")
    print(f"Sample  : {SAMPLE_N} rows, seed={SAMPLE_SEED}")
    print("=" * 72)

    print(f"\nRows with image_search calls : {rows_with_image_search}/{SAMPLE_N}")
    print(f"Rows with <useful> tags      : {rows_with_useful_tags}/{SAMPLE_N}")
    print(f"Total bbox calls analyzed    : {len(bbox_records)}")
    print(f"Total useful tags analyzed   : {len(useful_records)}")

    # ── SECTION 1: BBOX ───────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("1. CROP BBOX QUALITY")
    print("─" * 72)
    bbox_counts = Counter(r["verdict"] for r in bbox_records)
    total_b = len(bbox_records)
    for v in ["GOOD", "BAD", "UNCERTAIN"]:
        n = bbox_counts[v]
        pct = 100 * n / total_b if total_b else 0
        print(f"  {v:10s}: {n:4d} / {total_b}  ({pct:.1f}%)")

    bad_bbox = [r for r in bbox_records if r["verdict"] == "BAD"]
    unc_bbox = [r for r in bbox_records if r["verdict"] == "UNCERTAIN"]

    if bad_bbox:
        print(f"\n  BAD bbox examples (showing up to 10 of {len(bad_bbox)}):")
        for rec in bad_bbox[:10]:
            print(f"  ├ [rank {rec['rank']:3d}  df_idx {rec['df_idx']:4d}]"
                  f"  bbox={rec['bbox']}")
            print(f"  │   reason : {rec['reason']}")
            print(f"  │   goal   : {rec['goal'][:80]}")
            if rec["think_snippet"]:
                print(f"  │   think  : {rec['think_snippet'][:100]}")
            print(f"  │")

    if unc_bbox:
        print(f"\n  UNCERTAIN bbox examples (showing up to 5 of {len(unc_bbox)}):")
        for rec in unc_bbox[:5]:
            print(f"  ├ [rank {rec['rank']:3d}  df_idx {rec['df_idx']:4d}]"
                  f"  bbox={rec['bbox']}")
            print(f"  │   reason : {rec['reason']}")
            print(f"  │")

    # Bbox size distribution
    areas = []
    for r in bbox_records:
        b = normalize_bbox(r["bbox"][:4]) if r["bbox"] else None
        if b and len(b) >= 4:
            x1, y1, x2, y2 = [float(v) for v in b]
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                areas.append(w * h / 1_000_000)
    if areas:
        sorted_a = sorted(areas)
        n = len(sorted_a)
        def pct(p): return sorted_a[min(int(p / 100 * n), n - 1)]
        print(f"\n  Bbox area distribution (fraction of 1000×1000):")
        print(f"  n={n}  min={min(areas):.3f}  p25={pct(25):.3f}  "
              f"p50={pct(50):.3f}  p75={pct(75):.3f}  max={max(areas):.3f}")
        buckets = Counter(
            "0-10%"   if a < 0.10 else
            "10-30%"  if a < 0.30 else
            "30-60%"  if a < 0.60 else
            "60-80%"  if a < 0.80 else
            ">80% (whole-img)" for a in areas
        )
        for b in ["0-10%", "10-30%", "30-60%", "60-80%", ">80% (whole-img)"]:
            bar = "█" * (buckets[b] * 30 // (total_b or 1))
            print(f"    {b:18s}: {buckets[b]:4d}  {bar}")

    # ── SECTION 2: USEFUL TAGS ────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("2. USEFUL TAG QUALITY")
    print("─" * 72)
    useful_counts = Counter(r["verdict"] for r in useful_records)
    total_u = len(useful_records)
    for v in ["GOOD", "BAD", "UNCERTAIN"]:
        n = useful_counts[v]
        pct = 100 * n / total_u if total_u else 0
        print(f"  {v:10s}: {n:4d} / {total_u}  ({pct:.1f}%)")

    bad_useful = [r for r in useful_records if r["verdict"] == "BAD"]
    unc_useful = [r for r in useful_records if r["verdict"] == "UNCERTAIN"]

    if bad_useful:
        print(f"\n  BAD useful-tag examples (showing up to 12 of {len(bad_useful)}):")
        for rec in bad_useful[:12]:
            print(f"  ├ [rank {rec['rank']:3d}  df_idx {rec['df_idx']:4d}]"
                  f"  tagged={rec['indices']}")
            print(f"  │   reason : {rec['reason']}")
            if rec["detail"]:
                print(f"  │   detail : {rec['detail'][:120]}")
            if rec["sr_snippet"]:
                print(f"  │   results: {rec['sr_snippet'][:120].replace(chr(10),' ')}")
            print(f"  │")

    if unc_useful:
        print(f"\n  UNCERTAIN useful-tag examples (showing up to 6 of {len(unc_useful)}):")
        for rec in unc_useful[:6]:
            print(f"  ├ [rank {rec['rank']:3d}  df_idx {rec['df_idx']:4d}]"
                  f"  tagged={rec['indices']}")
            print(f"  │   reason : {rec['reason']}")
            print(f"  │")

    # Useful tag index count distribution
    tag_lens = Counter(len(r["indices"]) for r in useful_records)
    print(f"\n  Useful-tag cardinality distribution:")
    for k in sorted(tag_lens):
        bar = "█" * (tag_lens[k] * 20 // (total_u or 1))
        print(f"    {k:2d} tags: {tag_lens[k]:4d}  {bar}")

    # ── SECTION 3: OVERALL ────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("3. OVERALL ACCEPTANCE RATE")
    print("─" * 72)
    b_good = bbox_counts["GOOD"]
    u_good = useful_counts["GOOD"]
    b_pct = b_good / total_b if total_b else 0
    u_pct = u_good / total_u if total_u else 0
    ov_pct = (b_good + u_good) / (total_b + total_u) if (total_b + total_u) else 0
    print(f"  Bbox acceptance    : {b_pct:.1%}  ({b_good}/{total_b})")
    print(f"  Useful acceptance  : {u_pct:.1%}  ({u_good}/{total_u})")
    print(f"  Combined           : {ov_pct:.1%}  ({b_good+u_good}/{total_b+total_u})")

    # Per-row acceptance: a row is GOOD only if ALL its annotations are GOOD
    row_verdicts = {}
    for r in bbox_records + useful_records:
        key = r["rank"]
        if key not in row_verdicts:
            row_verdicts[key] = "GOOD"
        if r["verdict"] == "BAD":
            row_verdicts[key] = "BAD"
        elif r["verdict"] == "UNCERTAIN" and row_verdicts[key] != "BAD":
            row_verdicts[key] = "UNCERTAIN"

    row_counts = Counter(row_verdicts.values())
    rows_no_annotation = SAMPLE_N - len(row_verdicts)
    print(f"\n  Per-row verdict (all annotations must be GOOD):")
    print(f"    GOOD      : {row_counts.get('GOOD', 0)}")
    print(f"    BAD       : {row_counts.get('BAD', 0)}")
    print(f"    UNCERTAIN : {row_counts.get('UNCERTAIN', 0)}")
    print(f"    No tool call (single-pass) : {rows_no_annotation}")
    row_accept = row_counts.get("GOOD", 0) / SAMPLE_N
    print(f"  Per-row acceptance rate : {row_accept:.1%}")

    # ── SECTION 4: FAILURE MODES ──────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("4. COMMON FAILURE MODES")
    print("─" * 72)

    bbox_fail_modes = Counter()
    for r in bbox_records:
        if r["verdict"] != "GOOD":
            reason = r["reason"]
            if "whole-image bbox" in reason or "near-full" in reason:
                mode = "Whole-image bbox (no targeted crop)"
            elif "degenerate" in reason:
                mode = "Degenerate bbox (too small)"
            elif "zero/negative" in reason:
                mode = "Invalid bbox (zero/negative dims)"
            elif "full/near-full" in reason:
                mode = "Near-full bbox (model acknowledged no landmark)"
            elif "large bbox" in reason:
                mode = "Large bbox (>60%) without region rationale"
            else:
                mode = reason.split("(")[0].strip()[:60]
            bbox_fail_modes[mode] += 1

    useful_fail_modes = Counter()
    for r in useful_records:
        if r["verdict"] != "GOOD":
            reason = r["reason"]
            if "tagged []" in reason and "geo results exist" in reason:
                mode = "Empty tag despite clear geo results (false negative)"
            elif "missed" in reason and "clearly-geo" in reason:
                mode = "Severe false negatives (missed many geo results)"
            elif "non-geo results" in reason:
                mode = "False positives (non-geo results tagged useful)"
            elif "no search results" in reason:
                mode = "No search results to verify against"
            elif "could not parse" in reason:
                mode = "Could not parse search results"
            elif "minor false negatives" in reason:
                mode = "Minor false negatives (1-3 missed geo results)"
            elif "minor false positive" in reason:
                mode = "Minor false positive (1-2 non-geo tagged useful)"
            elif "tool returned error" in reason:
                mode = "Tool error / API failure"
            else:
                mode = reason[:60]
            useful_fail_modes[mode] += 1

    print("  Bbox failure modes:")
    if bbox_fail_modes:
        for mode, cnt in bbox_fail_modes.most_common():
            print(f"    {cnt:4d}x  {mode}")
    else:
        print("    (none)")

    print("  Useful-tag failure/uncertain modes:")
    for mode, cnt in useful_fail_modes.most_common():
        print(f"    {cnt:4d}x  {mode}")

    # ── SECTION 5: KEY FINDINGS ───────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("5. KEY FINDINGS & RECOMMENDATIONS")
    print("─" * 72)

    if total_b > 0:
        whole_img_bad = sum(1 for r in bbox_records if r["verdict"] == "BAD"
                            and "whole-image bbox" in r["reason"])
        whole_img_unc = sum(1 for r in bbox_records if r["verdict"] == "UNCERTAIN"
                            and "full/near-full" in r["reason"])
        targeted_good = bbox_counts["GOOD"]
        targeted_unc_large = sum(1 for r in bbox_records if r["verdict"] == "UNCERTAIN"
                                 and "large bbox" in r["reason"])
        degenerate = sum(1 for r in bbox_records if r["verdict"] == "BAD"
                         and ("degenerate" in r["reason"] or "zero/negative" in r["reason"]))
        print(f"\n  BBOX:")
        print(f"  • {whole_img_bad}/{total_b} ({whole_img_bad/total_b:.0%}) BAD: whole-image bbox without rationale")
        print(f"  • {whole_img_unc}/{total_b} ({whole_img_unc/total_b:.0%}) UNCERTAIN: whole-image (model acknowledged no distinct landmark)")
        print(f"  • {targeted_good}/{total_b} ({targeted_good/total_b:.0%}) GOOD: targeted crop (area <80%)")
        if degenerate:
            print(f"  • {degenerate} BAD: degenerate/invalid bbox")

    if total_u > 0:
        empty_bad = sum(1 for r in useful_records
                        if r["verdict"] == "BAD" and "tagged []" in r["reason"])
        missed_bad = sum(1 for r in useful_records
                         if r["verdict"] == "BAD" and "missed" in r["reason"])
        fp_bad = sum(1 for r in useful_records
                     if r["verdict"] == "BAD" and "non-geo" in r["reason"])
        print(f"\n  USEFUL TAGS:")
        print(f"  • {empty_bad} tags annotated as [] despite clear geo results (critical errors)")
        print(f"  • {missed_bad} tags missed most geo results (severe false negatives)")
        print(f"  • {fp_bad} tags contain clearly non-geo results (false positives)")
        print(f"  • {sum(1 for r in useful_records if r['verdict'] == 'UNCERTAIN')} "
              f"uncertain (minor misses/FP, or unverifiable)")

    print("\n" + "=" * 72)
    print("END OF REPORT")
    print("=" * 72)


def normalize_bbox(bbox):
    """Normalize fractional (0-1) bboxes to pixel (0-1000) space."""
    if not bbox:
        return bbox
    if max(bbox) <= 1.0:
        return [v * 1000 for v in bbox]
    return list(bbox)


if __name__ == "__main__":
    analyze()
