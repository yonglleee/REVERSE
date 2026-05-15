## Streamlit 可视化（60.jsonl）

### 依赖安装
```bash
pip install streamlit pandas
```

### 启动方式
```bash
cd /mnt/sh/mmvision/home/jonahli/projects/tusou/script/visualization
streamlit run streamlit_viewer.py
```

### 使用说明
- 默认读取同目录下 `60.jsonl`，也可在左侧输入任意 `jsonl` 路径。
- 支持字段筛选/排序、分页、统计分布与详情面板。
- 若文本中包含图片 URL/本地路径（如 `.png/.jpg`），会自动渲染为图片。
