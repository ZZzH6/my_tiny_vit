import os

class Config:
    """
    训练超参数配置中心。
    注意: num_classes 不再硬编码，由 datasets.py 的 Dataset Registry 动态提供。
    """
    BATCH_SIZE = 256 
    NUM_WORKERS = 16  
    SEED = 42
    EPOCHS = 360
    LR = 8e-4
    WEIGHT_DECAY = 0.025
    LABEL_SMOOTHING = 0.03
    GRAD_CLIP_NORM = 1.0  
    EMA_DECAY = 0.999
    DATA_DIR = './data'
    SAVE_DIR_BASE = './lightweight_saved'

    # 学习率调度参数
    WARMUP_EPOCHS = 10
    HOLD_EPOCHS = 0

    # 知识蒸馏参数
    KD_TEMPERATURE = 3.0
    KD_ALPHA = 0.4
    KD_DECAY_START_RATIO = 0.6

    # Mixup 配置参数
    MIXUP_ALPHA = 0.15
    # 由于原始分辨率仅 32x32, 强烈的 CutMix 破坏性极强，降低其介入概率或 Alpha
    CUTMIX_ALPHA = 0.0
    PROB = 0.15
    SWITCH_PROB = 0.0

    # 最后阶段做 clean fine-tune，去掉蒸馏和混合增强
    FINETUNE_EPOCHS = 40
    FINETUNE_LR = 5e-5
    FINETUNE_LABEL_SMOOTHING = 0.0


_DEFAULT_CONFIG_VALUES = {
    key: value
    for key, value in vars(Config).items()
    if key.isupper()
}


def _ensure_runtime_dirs():
    os.makedirs(Config.SAVE_DIR_BASE, exist_ok=True)


def reset_runtime_config():
    for key, value in _DEFAULT_CONFIG_VALUES.items():
        setattr(Config, key, value)
    _ensure_runtime_dirs()


def apply_runtime_overrides(overrides: dict):
    unknown_keys = [key for key in overrides if key not in _DEFAULT_CONFIG_VALUES]
    if unknown_keys:
        raise KeyError(f"Unknown config override keys: {unknown_keys}")

    for key, value in overrides.items():
        setattr(Config, key, value)
    _ensure_runtime_dirs()


def get_active_config():
    return {
        key: getattr(Config, key)
        for key in _DEFAULT_CONFIG_VALUES
    }


_ensure_runtime_dirs()
