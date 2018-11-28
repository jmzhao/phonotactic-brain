#!/bin/bash
DATASET=./data/english_biglist

## prepare dataset
python scripts/data_proc/parse_pscripts.py \
  --input "${DATASET}.jsonlines" \
  --output-dir "${DATASET}/"
## train a linear hole-filling model
PYTHONPATH='.' python scripts/train_linear.py \
  --input-train "${DATASET}/transcripts.list(list(id)).txt" \
  --output-dir "${DATASET}/holefilling/linear/" \
  --emb-dim 10 \
  --lr 0.001 --momentum 0.0 \
  --n-epoch 20
