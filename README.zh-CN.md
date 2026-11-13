# HighwayAccident-7K

论文《基于时序建模的视频事故检测与碰撞类型分类》的研究配套资料。
本集合包含多个来源的研究二次标注，不是完全自主拍摄的数据集。

```bash
python -m pip install -r requirements-evaluation.txt
python 03_evaluation/reproduce.py --check
python 05_figures/render_all.py
```

结果复算不需要原始视频、GPU 或权重。它核对末轮指标、冻结特征对照、排除敏感性、
来源映射、聚类区间及已记录的外部视频聚合结果。英文 README 说明各表的复算范围。

原完整清单场景 BA 为 77.61%，排除两个内容候选记录后为 77.60%，FAR 均为 11.90%。
类型的有标签准确率与激活子集准确率使用不同分母；真实来源子集结果另列。
不能把混合集表现直接描述为真实部署能力。

模型通过 Git LFS 保存，`git lfs pull` 可下载。末轮和第 8 轮分别提供，模型张量不变，
优化器及私人运行元数据已移除。标注可用 `tools/restore_labels.py` 恢复到新空目录。

本仓库不重新托管原始媒体，也不承诺仅凭仓库就能完整重建全部训练数据。
代码及模型遵循 AGPL-3.0；作者的数据／文档贡献遵循 CC BY-NC 4.0，第三方原有权利不变。
详细说明见 README.md、LICENSE-DATA.md、THIRD_PARTY_NOTICES.md 和各子目录说明。
