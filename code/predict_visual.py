import torch
import json
import random
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPProcessor
from train import CLIPAdapterModel
import os

def predict_visual():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./clip-vit-base-patch32"
    
    # 1. 加载模型与最优权重
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPAdapterModel(num_classes=8, model_path=model_path).to(device)
    model.load_state_dict(torch.load("bridge_clip_adapter_best.pth"))
    model.eval()

    # 2. 加载测试数据索引
    with open('test_metadata.json', 'r', encoding='utf-8') as f:
        content = json.load(f)
        samples = content['data'] if 'data' in content else content['samples']
        mapping = content['mapping']

    # 3. 随机抽取 5 张测试图进行可视化
    selected_samples = random.sample(samples, 5)
    
    if not os.path.exists('results'): os.makedirs('results')

    print("正在生成可视化预测图...")
    with torch.no_grad():
        for i, item in enumerate(selected_samples):
            img_path = item['image_path']
            true_label = item['label_name']
            
            image = Image.open(img_path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt")['pixel_values'].to(device)
            
            # 模型推理
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred_id = outputs.argmax(dim=1).item()
            pred_label = mapping[str(pred_id)]
            confidence = probs[0][pred_id].item()

            # 绘图
            draw = ImageDraw.Draw(image)
            text = f"True: {true_label}\nPred: {pred_label} ({confidence:.2f})"
            # 在左上角画个背景框方便看字
            draw.rectangle([10, 10, 250, 70], fill="black")
            draw.text((15, 15), text, fill="white")
            
            # 保存结果
            image.save(f"results/prediction_{i}.jpg")
            print(f"图片 {i} 已保存: 真实-{true_label} | 预测-{pred_label}")

if __name__ == "__main__":
    predict_visual()