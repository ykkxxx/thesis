import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import f1_score, accuracy_score
import os
import logging

# ================= 全局配置 =================
# 设置镜像，确保下载不超时
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
logging.basicConfig(level=logging.ERROR)

# ================= 1. 数据集加载类 =================
class BridgeDataset(Dataset):
    def __init__(self, json_path, processor):
        # build_dataset.py 生成的 JSON 包含 "mapping" 和 "samples"
        with open(json_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            self.samples = content['samples']
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = item['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 遇到坏图返回黑块，防止训练中断
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        label = item['label_id']
        # 预处理图像
        pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        return pixel_values, torch.tensor(label)

# ================= 2. 模型定义 (CLIP + Adapter) =================
class CLIPAdapterModel(nn.Module):
    def __init__(self, num_classes=8, model_path="./clip-vit-base-patch32"):
        super().__init__()
        # 加载本地 CLIP 模型
        self.clip = CLIPModel.from_pretrained(model_path)
        # 冻结 CLIP 主干，不参与训练
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # 适配器微调结构 (Adapter)
        self.adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 512)
        )
        # 最终分类头
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, pixel_values):
        # 提取视觉特征
        vision_feat = self.clip.get_image_features(pixel_values=pixel_values)
        # 残差适配器计算
        adapt_feat = vision_feat + self.adapter(vision_feat)
        logits = self.classifier(adapt_feat)
        return logits

# ================= 3. 训练主函数 =================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # 模型本地路径
    model_path = "./clip-vit-base-patch32"
    if not os.path.exists(model_path):
        print(f"⚠️ 本地未发现模型文件夹，将尝试在线下载...")
        model_path = "openai/clip-vit-base-patch32"

    # 1. 实例化处理器和模型 (顺序已修正)
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPAdapterModel(num_classes=8, model_path=model_path).to(device)

    # 2. 准备数据
    train_ds = BridgeDataset('train_metadata.json', processor)
    test_ds = BridgeDataset('test_metadata.json', processor)
    
    # AutoDL 环境下显存较大，batch_size 设为 64，num_workers 设为 4 加速读取
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    # 3. 筛选出需要训练的参数 (Adapter 和 Classifier)
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print(f"✅ 可训练参数量: {sum(p.numel() for p in trainable_params):,}")
    
    # 4. 定义优化器和损失函数
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    best_f1 = 0.0

    print("\n开始训练...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step % 20 == 0:
                print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}", end='\r')
            
        # 验证集评估
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Macro-F1: {f1:.4f}")
        
        # 保存最优权重
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "bridge_clip_adapter_best.pth")
            print(f"🌟 检测到性能提升，已保存最佳权重 (F1: {best_f1:.4f})")
        
        torch.save(model.state_dict(), "bridge_clip_adapter_latest.pth")

    print(f"\n🎉 训练全部完成！最佳 Macro-F1: {best_f1:.4f}")

if __name__ == "__main__":
    train()