import torch
import json
from PIL import Image
from transformers import CLIPProcessor
from train import CLIPAdapterModel  # 确保能引用到你的模型定义
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_pro():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./clip-vit-base-patch32"
    
    # 1. 加载模型与权重
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPAdapterModel(num_classes=8, model_path=model_path).to(device)
    model.load_state_dict(torch.load("bridge_clip_adapter_best.pth"))
    model.eval()

    # 2. 加载测试集索引
    with open('test_metadata.json', 'r', encoding='utf-8') as f:
        content = json.load(f)
        samples = content['data'] if 'data' in content else content['samples']
        mapping = content['mapping']

    all_preds = []
    all_labels = []

    print("开始深度评估...")
    with torch.no_grad():
        for item in samples:
            image = Image.open(item['image_path']).convert('RGB')
            label = item['label_id']
            
            inputs = processor(images=image, return_tensors="pt")['pixel_values'].to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1).item()
            
            all_preds.append(pred)
            all_labels.append(label)

    # 3. 产出报告 A：每类详细指标 (用于填表)
    target_labels = [0, 1, 2, 3, 4, 5, 6, 7]
    target_names = [mapping[str(i)] for i in target_labels]

    report = classification_report(
    all_labels, 
    all_preds, 
    labels=target_labels,    # 强制包含所有 ID
    target_names=target_names, 
    zero_division=0          # 没出现的类得分记为 0，不报错
)
    print("\n" + "="*20 + " 详细分类报告 " + "="*20)
    print(report)

    # 4. 产出报告 B：混淆矩阵图 (用于论文配图)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[mapping[str(i)] for i in range(8)],
                yticklabels=[mapping[str(i)] for i in range(8)])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Bridge Disease Detection')
    plt.savefig('confusion_matrix.png', dpi=300)
    print("\n✅ 混淆矩阵图已保存至: confusion_matrix.png")

if __name__ == "__main__":
    evaluate_pro()