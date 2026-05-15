#!/bin/bash
# =============================================================================
# run_granularity_ablation.sh  —  4-level granularity SFT ablation
# =============================================================================
# Launches 4 SFT jobs with different label granularities, using the same
# 200k images sampled from MP16-Pro (4-level-complete subset, seed=42).
#
# Label formats:
#   l0  \boxed{Latitude, Longitude}
#   l2  \boxed{Country, City, Latitude, Longitude}
#   l3  \boxed{Country, Region, City, Latitude, Longitude}
#   l4  \boxed{Country, Region, City, Neighbourhood, Latitude, Longitude}
#
# Data:   /mnt/sh/mmvision/home/jonahli/data/MP16-Pro/sft_granularity_parquet/<version>/train.parquet
# Ckpts:  /mnt/sh/mmvision/home/jonahli/save/tusou/sft/mp16pro-geo-qwen3-4b-<version>-fsdp-fsdp2-sp1/
#
# USAGE (run on each node, with WORLD_SIZE / RANK / MASTER_ADDR set by scheduler):
#
#   # Run a single version:
#   VERSION=l0 bash run_granularity_ablation.sh
#
#   # Run all 4 versions sequentially (single-node, for testing):
#   bash run_granularity_ablation.sh
# =============================================================================

GRANULARITY_DATA_BASE="/mnt/sh/mmvision/home/jonahli/data/MP16-Pro/sft_granularity_parquet"
DIST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If VERSION is set, run only that one; otherwise run all 4
if [ -n "${VERSION:-}" ]; then
    VERSIONS=("$VERSION")
else
    VERSIONS=(l0 l2 l3 l4)
fi

for ver in "${VERSIONS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Starting SFT: version=$ver"
    echo "============================================================"

    DATA_DIR="${GRANULARITY_DATA_BASE}/${ver}" \
    DATA_VERSION="${ver}" \
    bash "${DIST_DIR}/run_sft_dist.sh"

    echo "============================================================"
    echo "  Done: version=$ver"
    echo "============================================================"
done
