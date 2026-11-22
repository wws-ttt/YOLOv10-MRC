# 原核与卵裂球检测对照实验

在同一批带 YOLO 边界框标注的测试图像上评价 Cellpose-SAM、原版 Meta SAM 和训练完成的 YOLO-MRC。两个分割基线各自从训练集的掩膜候选框训练一个 HOG + 多项逻辑回归分类器，类别为卵裂球、原核和背景。

## 安装与数据

在能导入此项目 Ultralytics 分支的 Python 3.10/3.11 环境中运行以下命令。GPU 环境先安装相互兼容的 PyTorch 和 torchvision。现有环境若已安装 opencv-python，请避免同时装两个 OpenCV 发行包，必要时改用隔离环境。需要安装 Git 才能从官方仓库安装 SAM。

```powershell
Set-Location E:/guhtp/SAM
python -m pip install -r requirements-benchmark.txt
Copy-Item benchmark.example.yaml benchmark.yaml
```

编辑 benchmark.yaml，填入训练集、测试集和权重的实际路径。YOLO 标签文件必须与图像相对路径同名，格式为 class_id cx cy width height，坐标归一化。无目标图像配空标签文件。项目 coco.yaml 的 o、p 对应 ID 0、1，若数据集另有定义，须核对配置与权重。数据划分应以胚胎或患者为单位，避免同一对象的不同时间帧跨训练和测试。脚本会检查两集合间完全相同的图像文件。

- Cellpose 使用 4.0 系列 CPSAM。首次可自动获取默认权重，也可以指定已下载的本地文件。[Cellpose 官方文档](https://cellpose.readthedocs.io/en/latest/faq.html)
- 原版 SAM 默认使用 ViT-B。下载权重并填入配置，模型类型和权重需匹配。[Meta SAM 官方仓库及权重](https://github.com/facebookresearch/segment-anything#model-checkpoints)
- YOLO-MRC 使用项目中实际训练得到的 best.pt。对 RT-DETR 解码头使用 rtdetr。训练配置、训练数据及权重应与论文所述版本对应。
- 如需仅评价指定的 10 张图，在测试图片目录之外保存一份清单，每行填写相对于 test.images 的路径，然后将清单设为 test.manifest。三个模型会使用同一清单。省略清单时使用整个测试目录。

## 运行

```powershell
python benchmark.py benchmark.yaml --out runs/comparison-01
```

该命令依次生成训练集候选框、训练并保存分类器，再对固定测试集做推理和计分。每个输出目录须为新目录。

- metrics.json 记录图像数、分割分类头训练候选数、每类 AP50 和两类 mAP50。
- test-images.txt 记录实际评估的图像清单。
- 各模型子目录的 predictions.json 记录图像路径、类别、置信度及 xyxy 坐标。分割模型子目录另有 classifier.joblib。
- 分割掩膜使用最小外接框。训练时 IoU ≥ 0.50 为正例，IoU < 0.30 为背景，其余忽略。每个分割模型以分类概率给候选框打分，并对同类候选应用 IoU 0.70 的 NMS。
- AP50 采用逐图同类一对一匹配和 101 点插值，每张图最多保留 100 个检测。某个前景类在测试集没有标注时，该类 AP 和两类 mAP50 显示为 null。

请将论文中的数字替换为一次实际运行产生的结果，并注明测试样本数和范围。若希望训练 YOLO-MRC 本身，在项目根目录使用该分支的 RTDETR 接口加载组合 YAML，并用同一训练划分训练。benchmark.yaml 的 detector.checkpoint 填训练得到的权重。

### 训练 YOLO-MRC（需要重新训练时）

在项目根目录执行。将模型配置路径和数据配置路径换成实际文件。YOLO 数据配置需要 train、val 和与本实验一致的两个类别。请使用论文实验的训练参数，下面只展示调用形式。

```powershell
Set-Location E:/guhtp
python -c "from ultralytics import RTDETR; RTDETR(r'path/to/yolov10n-MRC.yaml').train(data=r'path/to/embryo-data.yaml', imgsz=640)"
```

训练完成后，把实际 best.pt 路径填入 benchmark.yaml，再运行统一评估。正式复现实验时保存原始权重、测试清单、配置、软件版本和 metrics.json。
