# 银行营销定期存款预测分析

本项目分析银行电话营销数据，识别更可能订购定期存款的客户，并将结果转化为可执行的营销建议。

## 项目目标

- 理解客户、联系活动和宏观经济因素与订购结果的关系。
- 建立并评估客户订购定期存款的预测模型。
- 提供客户筛选与营销资源分配建议。

## 目录说明

```text
data/raw/        原始数据；禁止在此目录直接修改文件
data/interim/    清洗与转换过程中的中间数据
data/processed/  分析、建模所用的最终数据集
notebooks/       按分析流程编号的 Jupyter Notebook
src/             可复用的数据、特征、建模和绘图函数
figures/         输出图表（探索性、建模、最终版本）
reports/         最终分析报告与演示文稿
tests/           src 中关键函数的自动化测试
```

## 建议的分析顺序

1. `01_data_understanding.ipynb`：数据字典、质量检查和目标变量概览。
2. `02_data_cleaning.ipynb`：缺失值、类别编码及清洗规则。
3. `03_eda.ipynb`：客户特征、活动表现与订购行为探索。
4. `04_feature_engineering.ipynb`：特征构造及训练/测试集划分。
5. `05_modeling.ipynb`：基线模型与候选模型训练。
6. `06_evaluation.ipynb`：指标、阈值、业务收益与最终建议。

## 复现

```bash
pip install -r requirements.txt
jupyter notebook
```

原始文件位于 `data/raw/`；所有 Notebook 应从项目根目录运行。