#!/usr/bin/env bash
set -euo pipefail

# Script para executar detectBatchv3.py variando batch_size e workers
# Ajuste as arrays abaixo conforme necessário.

# Configurações padrões
WEIGHTS="best.pt"
SOURCE="datasets/CSDD_det/images/test2017/"
IMG_SZ=2048

# Valores a testar
BATCH_SIZES=(4)
WORKERS=(0 1 4)


OUT_LOGS="./run_logs"
mkdir -p "$OUT_LOGS"

echo "Starting runs: $(date)"
for b in "${BATCH_SIZES[@]}"; do
  for w in "${WORKERS[@]}"; do
    # map workers -> subset percentage
    case "$w" in
      0) SUBSET_PCT="0.11" ;;
      1) SUBSET_PCT="0.33" ;;
      4) SUBSET_PCT="1" ;;
      *) SUBSET_PCT="1" ;; # default
    esac

    echo "\n--- Running: batch_size=$b workers=$w subset-pct=$SUBSET_PCT ---"
    ts=$(date +%Y%m%d_%H%M%S)
    logf="$OUT_LOGS/run_bs${b}_w${w}_$ts.log"

    # Execute the command and capture exit status; logs go to per-run file
    echo "Command: python yolov5/detectBatchv3.py --weights $WEIGHTS --source $SOURCE --imgsz $IMG_SZ --batch-size $b --workers $w --subset-pct $SUBSET_PCT" | tee -a "$logf"
    python yolov5/detectBatchv3.py --weights "$WEIGHTS" --source "$SOURCE" --imgsz "$IMG_SZ" --batch-size "$b" --workers "$w" --subset-pct "$SUBSET_PCT" >> "$logf" 2>&1 || echo "Run FAILED for batch=$b workers=$w (see $logf)" | tee -a "$logf"

    rm -rf yolov5/runs/detect/*
    echo "Finished run: batch_size=$b workers=$w; log=$logf"
    # small pause to let resources settle
    sleep 1
  done
done

echo "All runs finished: $(date)"
