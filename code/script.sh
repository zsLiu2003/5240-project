#baseline1
python code/train_ecfp_ridge.py \
    --data-dir data \
    --results-dir results

#baseline2
python code/train.py \
    --experiment full_finetune \
    --data-dir data \
    --results-dir results

#baseline3
python code/train.py \
    --experiment from_scratch \
    --data-dir data \
    --results-dir results

#main experiment (two-stage: freeze head -> unfreeze backbone)
python code/train.py --experiment main \
    --data-dir data \
    --results-dir results
#  Optional: limit Stage-2 to last N transformer layers if unstable:
#  python code/train.py --experiment main --unfreeze-layers 4 ...