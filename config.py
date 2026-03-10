import os

class Config:
    BATCH_SIZE = 512  
    NUM_WORKERS = 16  
    EPOCHS = 200         
    LR = 1e-3         
    WEIGHT_DECAY = 0.1   
    NUM_CLASSES = 100
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP_NORM = 1.0  
    EMA_DECAY = 0.999    
    DATA_DIR = './data'
    SAVE_DIR_BASE = './lightweight_saved'

    # Mixup 配置参数
    MIXUP_ALPHA = 0.8
    CUTMIX_ALPHA = 1.0
    PROB = 0.5       # 1.0 -> 0.5 (允许模型看一半干净样本，加快收敛)
    SWITCH_PROB = 0.5

os.makedirs(Config.SAVE_DIR_BASE, exist_ok=True)
