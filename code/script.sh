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
