# 退化手写文档二值化

这个作业想把有阴影、污渍的扫描图变成黑字白底的图。实验用了 105 张真实图像，其中 36 张用来选参数，49 张手写图和 20 张跨材料图用来测试。

## 怎么运行

在本目录打开终端，使用 Python 3.12。Windows 使用系统自带的 `tar.exe` 解包，Linux 需先安装提供 `bsdtar` 的 `libarchive-tools`。

```sh
python -m pip install -r requirements.txt
python benchmark_data.py
python benchmark.py tune
python benchmark.py test --save-images
```

第一步安装依赖。之后依次下载并配对原图和标注、在开发集上定参数、在测试集上算分。`--save-images` 会保存二值结果图。下载的数据放在 `data/`，重跑的结果放在 `outputs/benchmark/`，本次实测结果已放在 `results/`。

## 文件

- `benchmark_data.py` 下载官方数据，检查原图和标注是否配对。
- `binarization.py` 实现阈值分割、背景校正、中值滤波和页边处理。
- `benchmark.py` 选参数、运行七组对照、计算逐图分数。
- `results/summary.csv` 看汇总，`results/per_image.csv` 查单张图。`paired.csv` 是同图对比，两个 JSON 分别记录配对清单和冻结参数。

49 张手写图上，直接 Otsu 的平均 F1 是 82.76%，做背景校正后是 88.42%。各方法的对比图和效果不理想的例子都在 Word 报告里。发布代码后，把 GitHub 仓库链接填到报告第一节。
