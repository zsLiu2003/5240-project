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


# extra analysis
python code/smiles_robustness_probe.py \
    --no-aug-checkpoint results/main_method_seed42_fold0/best_model.pt \
    --aug-checkpoint results/main_method_aug0.5_seed42_fold0/best_model.pt \
    --seed 42 \
    --n-variants 8 \
    --output-dir results/smiles_robustness_probe

