import os
import json
import random
import collections
from pathlib import Path

# ================= 1. 配置信息 =================
# 确保路径指向你解压后的图片根目录
DATA_ROOT = '../data/images'  
TRAIN_RATIO = 0.8

# 严格对应你的 8 个文件夹名称
CLASS_NAMES = [
    "bottom_shrinkage_crack", 
    "concrete_void", 
    "corrosion", 
    "crack",
    "degraded_concrete", 
    "moist", 
    "pavement_deterioration", 
    "shrinkage_crack"
]

# 对应每一类病害的学术级英文提示词 (用于 CLIP 训练)
DISEASE_PROMPTS = {
    "bottom_shrinkage_crack": "Structural shrinkage cracks occurring at the bottom base of bridge components.",
    "concrete_void": "Empty voids or internal hollow spaces within the concrete bridge structure.",
    "corrosion": "Surface corrosion with visible rust stains and oxidation on bridge steel elements.",
    "crack": "Structural linear cracks on the surface of reinforced concrete bridge components.",
    "degraded_concrete": "Deteriorated concrete surface showing loss of material and exposed aggregates.",
    "moist": "Wet surfaces or water seepage marks on the concrete bridge structure.",
    "pavement_deterioration": "Severe wear, potholes or asphalt degradation on the bridge pavement surface.",
    "shrinkage_crack": "Superficial drying shrinkage cracks on the bridge concrete surface."
}

def build_dataset():
    train_samples = []
    test_samples = []
    root = Path(DATA_ROOT).resolve()
    
    # 固定随机种子，确保每次运行脚本生成的划分结果完全一致
    random.seed(42) 

    print(f"开始处理数据集，根目录: {root}\n")

    for idx, name in enumerate(CLASS_NAMES):
        folder = root / name
        if not folder.exists():
            print(f"⚠️ 警告: 找不到文件夹 {name}，已跳过。")
            continue

        prompt = DISEASE_PROMPTS[name]
        
        # --- 步骤 1: 按病害实例前缀进行分组 ---
        # 键: '587_jpg', 值: 该实例下所有增强后的图片路径列表
        groups = collections.defaultdict(list)
        for img_p in folder.glob('*.jpg'):
            # 提取前缀，例如从 587_jpg.rf.xxx.jpg 中提取出 587_jpg
            prefix = img_p.name.split('.rf.')[0]
            groups[prefix].append(str(img_p))
        
        # --- 步骤 2: 对“实例”进行随机打乱和划分 ---
        prefixes = list(groups.keys())
        random.shuffle(prefixes)
        
        # 计算划分点
        split_idx = int(len(prefixes) * TRAIN_RATIO)
        
        # 特殊处理：如果该类只有 1 个病害实例，必须分给训练集，否则模型没法学
        if len(prefixes) == 1:
            split_idx = 1
            
        train_p = prefixes[:split_idx]
        test_p = prefixes[split_idx:]
        
        # --- 步骤 3: 统计并分发数据 ---
        # 准确统计该类别实际分入的图片总数（修复之前的显示Bug）
        train_img_count = sum(len(groups[p]) for p in train_p)
        test_img_count = sum(len(groups[p]) for p in test_p)
        
        # 将图片信息存入列表
        for p in train_p:
            for img_path in groups[p]:
                train_samples.append({
                    "image_path": img_path, 
                    "label_id": idx, 
                    "label_name": name, 
                    "text_prompt": prompt
                })
        
        for p in test_p:
            for img_path in groups[p]:
                test_samples.append({
                    "image_path": img_path, 
                    "label_id": idx, 
                    "label_name": name, 
                    "text_prompt": prompt
                })
                
        # 打印当前类别的详细划分情况
        print(f"类别 {name:25} | 实例数: {len(prefixes):4} | 训练集图片: {train_img_count:4} | 测试集图片: {test_img_count:4}")

    # 类别映射表
    mapping = {idx: name for idx, name in enumerate(CLASS_NAMES)}

    # --- 步骤 4: 保存为 JSON 文件 ---
    with open("train_metadata.json", "w", encoding='utf-8') as f:
        json.dump({"mapping": mapping, "samples": train_samples}, f, indent=4, ensure_ascii=False)
    
    with open("test_metadata.json", "w", encoding='utf-8') as f:
        json.dump({"mapping": mapping, "samples": test_samples}, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 严谨划分完成！")
    print(f"总计训练图片: {len(train_samples)}")
    print(f"总计测试图片: {len(test_samples)}")
    print(f"已生成: train_metadata.json, test_metadata.json")

if __name__ == "__main__":
    build_dataset()