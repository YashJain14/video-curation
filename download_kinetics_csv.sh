#!/bin/bash
# Download the official Kinetics-400 validation split CSV from DeepMind.
# Run this on the login node before submitting the PBS job.

SCRATCH_DIR="${SCRATCH_DIR:-$HOME/scratch/video-curation}"
mkdir -p "$SCRATCH_DIR"
CSV="$SCRATCH_DIR/kinetics400_val.csv"

if [ -f "$CSV" ]; then
    echo "Already have $CSV ($(wc -l < $CSV) lines)"
    exit 0
fi

echo "Downloading Kinetics-400 val CSV ..."
wget -q -O "$CSV" \
    "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz"

# The above is a tarball — extract the val CSV
if file "$CSV" | grep -q "gzip"; then
    mv "$CSV" data/kinetics400.tar.gz
    tar -xzf data/kinetics400.tar.gz -C data/
    # Find the val CSV inside the extracted dir
    VAL=$(find data/ -name "validate.csv" | head -1)
    if [ -n "$VAL" ]; then
        cp "$VAL" "$CSV"
        echo "Extracted val CSV → $CSV"
    else
        echo "Could not find validate.csv in archive. Check data/ manually."
    fi
fi

echo "Lines in CSV: $(wc -l < $CSV)"
echo "First row:    $(head -2 $CSV | tail -1)"
