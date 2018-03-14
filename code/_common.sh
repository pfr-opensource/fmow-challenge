#!/bin/bash
set -eu
FMOW_RGB="`dirname "$1"`"
TEST_DIR="$2"
OUTPUT_FILE="$3".txt
MODE="$4"

MODELS=models/*.json

if [ "$MODE" != "--train" -a "$MODE" != "--test" ]; then
    echo You must specify a mode.
    exit 1
fi

if [ "$1" != "$FMOW_RGB/train" ]; then
    echo Error: first argument should be of the form .../train
    exit 1
fi

echo Will use training data from: "$FMOW_RGB/train" and "$FMOW_RGB/val"

if [ `ls -l $MODELS | wc -l` != 12 ]; then
    echo "models/*.json should not be removed"
    exit 1
fi

if [ "$MODE" == "--train" ]; then
    echo "Step 1. Parse JSON metadata."
    python3 code/step1_parse_json.py trainval-rgb "$FMOW_RGB"

    echo "Step 2. Add new boxes."
    python3 code/step2_add_boxes.py

    echo "Step 3. Build crops."
    time python3 code/step3_build_crops.py training

    echo "Step 4. Train models."
    for model in $MODELS; do
        time python3 code/step4_train.py "$model"
    done

    echo "Training complete."
fi

if [ "$MODE" == "--test" ]; then
    echo Will use test data from: "$TEST_DIR"
    echo Will output prediction to: "$OUTPUT_FILE"

    echo "Run baseline model first."
    time bash code/run_baseline.sh "$FMOW_RGB" "$TEST_DIR"

    echo "Step 1. Parse JSON metadata."
    python3 code/step1_parse_json.py test-rgb "$TEST_DIR"

    echo "Step 3. Build crops."
    time python3 code/step3_build_crops.py "test"

    echo "Step 5. Evaluate models."
    for model in $MODELS; do
        time python3 code/step5_test.py "$model"
    done

    echo "Step 6. Build final prediction."
    python3 code/step6_ensemble.py "$OUTPUT_FILE"

    echo "Testing complete."
    wc -l "$OUTPUT_FILE"
fi
