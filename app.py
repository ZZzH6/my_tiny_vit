import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import os
from models import CustomMobileViT

# 模型和配置常量
MODEL_PATH = './models_saved/best_model.pth'
NUM_CLASSES = 100
ATTENTION_TYPE = 'coord'
DATA_DIR = './data'

# 页面配置
st.set_page_config(page_title="定制版 MobileViT 图像检测仪", page_icon="🔍")
st.title("🔍 定制版 MobileViT 图像检测仪 (CIFAR-100)")
st.write("欢迎来到轻量化图像分类检测平台！请上传一张图像，看看模型识别的结果吧。")

@st.cache_resource
def load_class_names():
    # 借助 dataset 获取 CIFAR-100 类名
    dataset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=True)
    return dataset.classes

@st.cache_resource
def load_model():
    model = CustomMobileViT(num_classes=NUM_CLASSES, pretrained=False, attention_type=ATTENTION_TYPE)
    if os.path.exists(MODEL_PATH):
        # 兼容当前 CPU 和 GPU 推理
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        st.error(f"未找到模型文件：{MODEL_PATH}，请确保模型已正确训练并保存。")
        return None

# 加载类别名称和模型权重
with st.spinner("正在加载类别标签和模型权重..."):
    class_names = load_class_names()
    model = load_model()

# 图像预处理流水线（须与训练时的验证预处理保持一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

st.sidebar.header("参数设置")
top_k = st.sidebar.slider("显示 Top-K 预测结果个数", min_value=1, max_value=10, value=5)

uploaded_file = st.file_uploader("请上传一张待检测图像 (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示上传图像并进行预处理
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='已上传的图像', use_container_width=True)
    
    st.write("---")
    st.write("⏳ **模型正在推理中...**")
    
    try:
        input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
        
        if model is not None:
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
            # 获取概率最高的 Top-K 结果
            top_prob, top_catid = torch.topk(probabilities, top_k)
            
            st.success("✅ **推理完成！**")
            st.subheader(f"Top-{top_k} 预测结果：")
            
            for i in range(top_prob.size(0)):
                idx = top_catid[i].item()
                prob = top_prob[i].item()
                class_name = class_names[idx]
                st.write(f"**{i+1}.** `{class_name}` (预测概率: **{prob*100:.2f}%**)")
                st.progress(prob)
                
    except Exception as e:
        st.error(f"推理过程中发生错误: {e}")
