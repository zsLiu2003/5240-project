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

# new
cd /hdd2/zesen/daily/5240/5240-project/code                                                 
                                                                                              
  # 运行main method（默认配置，3个seeds）                                                     
  python train.py --experiment main                                                           
                                                                                              
  # 如果想使用交叉验证（更稳健）                                                              
  python train.py --experiment main --cv --cv-folds 5                                         
                                                                                              
  # 如果想调整超参数                                                                          
  python train.py --experiment main --max-epochs 50 --batch-size 32 --lr 1e-3                 
                                                                                              
  # 如果只想解冻最后4层transformer                                                            
  python train.py --experiment main --unfreeze-layers 4                                       
                                                                                              
  其他实验类型：  
  # 消融实验：只有Stage 1                                                                     
  python train.py --experiment stage1_only
                                                                                              
  # 消融实验：只有Stage 2                                                                     
  python train.py --experiment stage2_only                                                    
                                                                                              
  # 消融实验：完全微调（无两阶段）                                                            
  python train.py --experiment full_finetune                                                  
                                                                                              
  # 消融实验：从头训练                                                                        
  python train.py --experiment from_scratch            