#!/bin/bash
# Download Kinetics-400 annotation CSVs directly from CVDF S3.
# Run on the login node before submitting the PBS job.

SCRATCH_DIR="${SCRATCH_DIR:-$HOME/scratch/video-curation}"
mkdir -p "$SCRATCH_DIR"

for SPLIT in val train; do
    CSV="$SCRATCH_DIR/kinetics400_${SPLIT}.csv"
    if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt 1000 ]; then
        echo "Already have $CSV ($(wc -l < "$CSV") lines)"
        continue
    fi
    echo "Downloading kinetics400_${SPLIT}.csv from S3..."
    wget -q -O "$CSV" "https://s3.amazonaws.com/kinetics/400/annotations/${SPLIT}.csv"
    echo "  Lines: $(wc -l < "$CSV")  |  First row: $(head -2 "$CSV" | tail -1)"
done

echo "CSVs in $SCRATCH_DIR:"
ls -lh "$SCRATCH_DIR"/*.csv
