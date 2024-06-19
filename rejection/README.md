
各目录/文件说明：

```txt
.
├── actual_github_fix_1024      # 20240617 旧版豆哥在不同 chunksize 下的 F1
├── bce                         # bce 模型运行结果
├── bge_v1.5                    # bge 模型测试结果
├── chunk_size_chinese_splitter # 使用 ChineseTextSplitter 测的不同 chunksize 下的 F1
├── chunk_size_recursive_text_splitter  # 使用 RecursiveTextSplitter 测的不同 chunksize 下的 F1
├── gt_bad.txt  # 人工标注的负例
├── gt_good.txt # 人工标注的正例
├── input.jsonl # 2302 条测试输入，来自 HuixiangDou 真实运行
├── plot.py    # 在 splitter 这类目录绘制 3D 对比曲线；或者 2D。或辅助画统计结果
└── splitters  # 不同 splitter 在 768 chunksize 上测试
```