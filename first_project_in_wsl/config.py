import os

# 强行将 Hugging Face 流量指向国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class Config:
    # 训练超参数
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    EPOCHS = 50
    LR = 5e-4
    WEIGHT_DECAY = 0.05
    NUM_CLASSES = 100
    LABEL_SMOOTHING = 0.1
    SAVE_PATH = './best_model.pth'
    
    # 默认数据路径
    DATA_DIR = './data'
