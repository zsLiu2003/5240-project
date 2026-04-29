python train.py --experiment main --cv --cv-folds 5

python code/train.py \
    --experiment main \
    --cv --cv-folds 5 \
    --data-dir data \
    --results-dir results

python code/train.py \
    --experiment stage1_only \
    --augment --aug-prob 0.5 \
    --cv --cv-folds 5 \
    --data-dir data \
    --results-dir results

python code/train.py \
    --experiment stage2_only \
    --augment --aug-prob 0.5 \
    --cv --cv-folds 5 \
    --data-dir data \
    --results-dir results
                                                                                              
     