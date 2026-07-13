import os
import json
import random
from pathlib import Path

# ================= 配置 =================
DATA_ROOT = '../data/images'  
TRAIN_RATIO = 0.8

CLASS_NAMES = [
    "bottom_shrinkage_crack", "concrete_void", "corrosion", "crack",
    "degraded_concrete", "moist", "pavement_deterioration", "shrinkage_crack"
]

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
    all_entries = []
    root = Path(DATA_ROOT)
    
    # 直接遍历定义的类别顺序
    for idx, name in enumerate(CLASS_NAMES):
        folder = root / name
        prompt = DISEASE_PROMPTS[name]
        
        # 仅搜索 jpg 文件
        images = list(folder.glob('*.jpg'))
        print(f"ID {idx} | {name}: {len(images)} 张图片")
        
        for img_path in images:
            all_entries.append({
                "image_path": str(img_path),
                "label_id": idx,
                "label_name": name,
                "text_prompt": prompt
            })

    # 打乱并切分
    random.seed(42)
    random.shuffle(all_entries)
    split_point = int(len(all_entries) * TRAIN_RATIO)
    
    mapping = {idx: name for idx, name in enumerate(CLASS_NAMES)}

    # 写入 JSON
    for filename, data in [("train_metadata.json", all_entries[:split_point]), 
                           ("test_metadata.json", all_entries[split_point:])]:
        with open(filename, "w", encoding='utf-8') as f:
            json.dump({"mapping": mapping, "samples": data}, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 完成！训练集: {split_point} | 测试集: {len(all_entries) - split_point}")

if __name__ == "__main__":
    build_dataset()