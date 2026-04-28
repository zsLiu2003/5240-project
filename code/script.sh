#baseline1
python code/train_ecfp_ridge.py \
    --data-dir /hdd2/zesen/daily/5240/5240-project/data \
    --results-dir /hdd2/zesen/daily/5240/5240-project/results

#baseline2
python train.py \
    --experiment full_finetune \
    --data-dir /hdd2/zesen/daily/5240/5240-project/data \
    --results-dir /hdd2/zesen/daily/5240/5240-project/results

#baseline3
python train.py \
    --experiment from_scratch \
    --data-dir /hdd2/zesen/daily/5240/5240-project/data \
    --results-dir /hdd2/zesen/daily/5240/5240-project/results
