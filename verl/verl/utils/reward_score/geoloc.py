"""
Reward scoring for geo-localization tasks.

Expected answer format: <answer>Country, City, Latitude, Longitude</answer>
e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>

Reward = w_acc × acc_reward + w_fmt × format_reward + w_tool × tool_reward
  (w_acc=0.6, w_fmt=0.1, w_tool=0.3 — combined in limited.py)

This module computes acc_reward and format_reward independently.
Tool reward is computed in the tool execute() methods and combined in limited.py.

acc_reward is based on haversine distance between predicted and GT coordinates,
using piecewise-linear interpolation through anchor points:
    (1 km, 1.00)
    (25 km, 0.80)
    (200 km, 0.60)
    (750 km, 0.40)
    (2500 km, 0.20)
    d > 2500 km → 0.00
"""

import math
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Format check
# ---------------------------------------------------------------------------
# Base format (0.8): <think>...</think> + <answer>...</answer>
# Bonus (0.2): <useful>[...]</useful> present after a search tool response
#   → encourages model to output useful tag, aligned with SFT coldstart supervision

_PATTERN_BASE = re.compile(r"<think>.*</think>.*<answer>.*</answer>.*", re.DOTALL)
_PATTERN_USEFUL = re.compile(r"<useful>\s*\[.*?\]\s*</useful>", re.DOTALL)

def format_reward(predict_str: str) -> float:
    """
    0.0 — missing <think> or <answer>
    0.8 — has <think>...</think> + <answer>...</answer>
    1.0 — also has <useful>[...]</useful> (bonus for search result discrimination)
    """
    if not re.fullmatch(_PATTERN_BASE, predict_str):
        return 0.0
    if _PATTERN_USEFUL.search(predict_str):
        return 1.0
    return 0.8


# ---------------------------------------------------------------------------
# Coordinate parsing  — handles both \boxed{lat, lon, ...} and bare "lat, lon"
# ---------------------------------------------------------------------------

_FLOAT = r"[-+]?\d+(?:\.\d+)?"

# e.g. "Latitude: 40.71, Longitude: -74.01"  or  "lat=-23.5, lon=46.6"
_LATLON_KW = re.compile(
    rf"lat(?:itude)?\s*[=:]\s*({_FLOAT})\s*[,;]?\s*lon(?:gitude)?\s*[=:]\s*({_FLOAT})",
    re.IGNORECASE,
)

# bare pair in parentheses / brackets: "(40.71, -74.01)"
_LATLON_PAREN = re.compile(
    rf"[([]?\s*({_FLOAT})\s*,\s*({_FLOAT})\s*[)\]]?"
)


def _parse_coords(text: str) -> Optional[tuple[float, float]]:
    """Return (lat, lon) or None if not parseable.

    Priority: <answer>...</answer> > \\boxed{lat, lon, ...} > lat=.../lon=... keyword > bare "lat, lon" pair.

    New format: Country, City, Latitude, Longitude (lat/lon are last two fields).
    Legacy format: Latitude, Longitude, Country, City (lat/lon are first two fields).
    Both are handled: we try to find a valid (lat, lon) pair in the extracted text.
    """
    # 1. Extract from <answer>...</answer> first
    answer_m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_m:
        search_text = answer_m.group(1).strip()
    else:
        # 2. Fall back to \boxed{...} for legacy compatibility
        boxed_m = re.search(r"\\boxed\{([^}]*)\}", text)
        search_text = boxed_m.group(1) if boxed_m else text

    # 3. keyword form: lat=... lon=...
    m = _LATLON_KW.search(search_text)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass

    # 4. Try all consecutive float pairs; prefer the pair where both are valid coords.
    #    New format: last two numbers are lat, lon.
    #    Legacy format: first two numbers are lat, lon.
    #    We collect all float pairs and return the last valid (lat, lon) pair first,
    #    then fall back to first.
    matches = list(_LATLON_PAREN.finditer(search_text))
    valid_pairs = []
    for m in matches:
        try:
            lat, lon = float(m.group(1)), float(m.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                valid_pairs.append((lat, lon))
        except ValueError:
            continue
    if valid_pairs:
        return valid_pairs[-1]  # new format: lat/lon at the end
    return None


# ---------------------------------------------------------------------------
# Haversine distance (km)
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Distance → reward
# ---------------------------------------------------------------------------

_KNOTS = [
    (1.0, 1.00),
    (25.0, 0.80),
    (200.0, 0.60),
    (750.0, 0.40),
    (2500.0, 0.20),
]


def distance_to_score(km: float) -> float:
    if km <= _KNOTS[0][0]:
        return _KNOTS[0][1]

    for i in range(1, len(_KNOTS)):
        x0, y0 = _KNOTS[i - 1]
        x1, y1 = _KNOTS[i]
        if km <= x1:
            t = (km - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)

    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _compute_score(
    predict_str: str,
    ground_truth: str,
    format_score: float = 0.1,
) -> float:
    """
    Internal reward computation. Used by SpotSFTGeolocTool.

    Args:
        predict_str:  the model's submitted answer string
        ground_truth: the reference answer ("lat, lon, Country, City" or "Country, City, lat, lon")
        format_score: weight for format reward (has <think> + <answer>)

    Returns:
        Scalar reward in [0, 1].
    """
    gt_coords = _parse_coords(ground_truth)
    pred_coords = _parse_coords(predict_str)

    fmt = format_reward(predict_str)

    if gt_coords is None:
        # No parseable GT — fall back to exact text match
        acc = 1.0 if predict_str.strip() == ground_truth.strip() else 0.0
        return (1.0 - format_score) * acc + format_score * fmt

    if pred_coords is None:
        return format_score * fmt

    dist = haversine_km(pred_coords[0], pred_coords[1], gt_coords[0], gt_coords[1])
    acc = distance_to_score(dist)
    return (1.0 - format_score) * acc + format_score * fmt


# ---------------------------------------------------------------------------
# custom_reward_function compatible wrapper
# Signature required by VERL custom_reward_function:
#   compute_score(data_source, solution_str, ground_truth, extra_info=None)
# Can be used standalone without the tool, by setting in YAML:
#   custom_reward_function:
#     path: verl/utils/reward_score/geoloc.py
#     name: compute_score   # (or leave unset if this is the only one)
# ---------------------------------------------------------------------------

def compute_score(data_source, solution_str, ground_truth, extra_info=None):  # noqa: F811
    """Return a dict with "score" (scalar reward) plus Im2GPS3K threshold accuracy keys.

    The dict format allows limited.py to automatically write all keys into
    reward_extra_info, making acc_1km / acc_25km / acc_200km / acc_750km /
    acc_2500km appear in wandb val-aux metrics (e.g. val-aux/geoloc/acc_25km/mean@N).
    km is also logged (None → written as-is, process_validation_metrics skips None).
    """
    detailed = compute_score_detailed(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    # limited.py reads result["score"] as the scalar reward.
    # "acc" alias ensures val-core routing in ray_trainer._val_metrics_update.
    # Convert km=None → float("nan") so np.mean in metric_utils.py doesn't crash.
    km_val = detailed["km"]
    reward = detailed["reward"]
    return {
        "score":       reward,
        "acc":         reward,   # ray_trainer uses "acc" as the val-core key
        "km":          float("nan") if km_val is None else float(km_val),
        "acc_1km":     detailed["acc_1km"],
        "acc_25km":    detailed["acc_25km"],
        "acc_200km":   detailed["acc_200km"],
        "acc_750km":   detailed["acc_750km"],
        "acc_2500km":  detailed["acc_2500km"],
    }


def compute_score_detailed(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    format_score: float = 0.1,
) -> dict:
    """
    Returns acc_reward and format_reward independently.
    The final weighted combination (0.6×acc + 0.1×fmt + 0.3×tool) is done in limited.py.

    The "reward" field here = acc_reward only (for backward compat with val metrics).

    Returns:
        {
          "reward":        float  — acc_reward (haversine score, used as "score" in val metrics)
          "acc_reward":    float  — raw haversine distance score ∈ [0, 1]
          "format_reward": float  — format check score ∈ {0, 0.8, 1.0}
          "km":            float | None
          "acc_1km" .. "acc_2500km": 0 or 1
        }
    """
    gt_coords   = _parse_coords(ground_truth)
    pred_coords = _parse_coords(solution_str)
    fmt         = format_reward(solution_str)

    if gt_coords is None:
        acc = 1.0 if solution_str.strip() == ground_truth.strip() else 0.0
        return {
            "reward":     acc,
            "format_reward": fmt,
            "acc_reward":    acc,
            "km":         None,
            "acc_1km":    0,
            "acc_25km":   0,
            "acc_200km":  0,
            "acc_750km":  0,
            "acc_2500km": 0,
        }

    if pred_coords is None:
        return {
            "reward":     0.0,
            "format_reward": fmt,
            "acc_reward":    0.0,
            "km":         None,
            "acc_1km":    0,
            "acc_25km":   0,
            "acc_200km":  0,
            "acc_750km":  0,
            "acc_2500km": 0,
        }

    km     = haversine_km(pred_coords[0], pred_coords[1], gt_coords[0], gt_coords[1])
    acc    = distance_to_score(km)
    return {
        "reward":     acc,
        "format_reward": fmt,
        "acc_reward":    acc,
        "km":         round(km, 3),
        "acc_1km":    int(km <= 1),
        "acc_25km":   int(km <= 25),
        "acc_200km":  int(km <= 200),
        "acc_750km":  int(km <= 750),
        "acc_2500km": int(km <= 2500),
    }
