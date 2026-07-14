1. 核心结论 (用于结论章节)
最高指标：Macro-F1 = 0.38 (Report) / 0.43 (Train Log)。
改进点：引入了 Class Weights（类别权重补偿），成功激活了小样本类别的识别能力。
最大突破：pavement_deterioration (路面劣化) 召回率达到 100%。
2. 实验环境与参数 (用于方法章节)
模型底座：OpenAI CLIP-ViT-B/32 (Frozen)。
适配器：2层全连接 Bottleneck Adapter (512->256->512)。
损失函数：Weighted CrossEntropyLoss。
权重设置：[100, 1.7, 1.0, 4.2, 2.3, 1.0, 80, 150] (对应 0-7 类)。
划分策略：基于实例前缀的独立划分 (Instance-based Split) —— 这是你论文防御“数据泄露”质疑的最强盾牌。
3. 结果图片深度解析 (用于结果分析章节)
图片编号	结果状态	深度诊断 (写论文用)
prediction_0	成功 (Match)	验证了模型对 moist 类大样本的稳定性。
prediction_1	失败 (Crack -> Void)	典型误判。反映了细长纹理在低空间分辨率下易与坑洞阴影混淆的局限性。
prediction_2	成功 (Match)	证明模型抓取到了 degraded_concrete 粗糙表面的语义特征。
prediction_3	失败 (Degraded -> Corrosion)	语义重叠。退化混凝土表面常伴随钢筋锈蚀颜色，导致模型多标签判别困难。
prediction_4	失败 (Corrosion -> Moist)	光影干扰。锈蚀的黄褐色与某些渗水区域的变色在 CLIP 空间中距离过近。
