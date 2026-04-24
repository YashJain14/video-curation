#!/bin/bash
# Download the official Kinetics-400 validation split CSV.
# Run this on the login node before submitting the PBS job.

SCRATCH_DIR="${SCRATCH_DIR:-$HOME/scratch/video-curation}"
mkdir -p "$SCRATCH_DIR"
CSV="$SCRATCH_DIR/kinetics400_val.csv"

if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt 1000 ]; then
    echo "Already have $CSV ($(wc -l < "$CSV") lines)"
    exit 0
fi

echo "Downloading Kinetics-400 val CSV → $CSV"

# Try the cvdfoundation mirror (plain CSV, no tarball)
wget -q -O "$CSV" \
    "https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/k400/annotations/val.csv"

LINES=$(wc -l < "$CSV")
if [ "$LINES" -gt 1000 ]; then
    echo "Downloaded val CSV: $LINES lines"
    echo "First row: $(head -2 "$CSV" | tail -1)"
    exit 0
fi

# Fallback: OpenDataLab mirror
echo "First mirror failed, trying fallback..."
wget -q -O "$CSV" \
    "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400_val.csv"

LINES=$(wc -l < "$CSV")
if [ "$LINES" -gt 1000 ]; then
    echo "Downloaded val CSV: $LINES lines"
    echo "First row: $(head -2 "$CSV" | tail -1)"
    exit 0
fi

echo "Both mirrors failed. Download manually:"
echo "  https://github.com/cvdfoundation/kinetics-dataset"
echo "  Place the val CSV at: $CSV"
exit 1
