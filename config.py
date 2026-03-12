import os

class Config:
    """
    训练超参数配置中心。
    注意: num_classes 不再硬编码，由 datasets.py 的 Dataset Registry 动态提供。
    """
    BATCH_SIZE = 256 
    NUM_WORKERS = 16  
    EPOCHS = 500         
    LR = 1e-3         
    WEIGHT_DECAY = 0.05   
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP_NORM = 1.0  
    EMA_DECAY = 0.9995    
    DATA_DIR = './data'
    SAVE_DIR_BASE = './lightweight_saved'

    # Mixup 配置参数
    MIXUP_ALPHA = 0.5
    # 由于原始分辨率仅 32x32, 强烈的 CutMix 破坏性极强，降低其介入概率或 Alpha
    CUTMIX_ALPHA = 0.0
    PROB = 0.5       # 1.0 -> 0.5 (允许模型看一半干净样本，加快收敛)
    SWITCH_PROB = 0.0

os.makedirs(Config.SAVE_DIR_BASE, exist_ok=True)
