# # from ultralytics import YOLOv10
# #
# #
# # def main():
# #     # 加载模型，split='test'利用测试集进行测试
# #     model = YOLOv10(r"runs/train/MLLA-Arelu2/weights/best.pt")
# #     model.val(data="coco.yaml", split='test', imgsz=640, batch=16, device=0, workers=0)  # 模型验证
# #
# #
# # if __name__ == "__main__":
# #     main()
#
# #
# # from ultralytics import RTDETR
# #
# #
# # def main():
# #     # 加载模型，split='test'利用测试集进行测试
# #     model = RTDETR(r"runs/train/MLLA-RTDETR-clo/weights/best.pt")
# #     model.val(data="coco.yaml", split='test', imgsz=640, batch=16, device=0, workers=0)  # 模型验证
# #
# #
# # if __name__ == "__main__":
# #     main()
# import cv2
# from ultralytics import YOLOv10
# #
# # def main():
# #     # 加载模型
# #     model = YOLOv10(r"runs/train/MLLA-RTDETR-clo/weights/best.pt")
# #
# #     # 加载测试数据集
# #     model.val(data="coco.yaml", split='test', imgsz=640, batch=16, device=0, workers=0)
# #
# #     # 加载测试图片
# #     test_images = []  # 假设你已经加载了测试图片的路径列表
# #     for img_path in test_images:
# #         img = cv2.imread(img_path)
# #         if img is None:
# #             print(f"无法加载图片: {img_path}")
# #             continue
# #
# #         # 使用模型进行预测
# #         results = model(img, imgsz=640)
# #
# #         # 自定义可视化
# #         for det in results.xyxy[0]:
# #             # det格式: [x1, y1, x2, y2, confidence, class]
# #             x1, y1, x2, y2, conf, cls = det
# #             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
# #             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #             cv2.putText(img, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# #
# #         # 显示图片（不显示图片名称）
# #         cv2.imshow("Detection", img)  # 使用固定的标题 "Detection"
# #         cv2.waitKey(0)
# #         cv2.destroyAllWindows()
# #
# # if __name__ == "__main__":
# #    main()
#
# from ultralytics import YOLOv10
# import cv2
# import numpy as np
#
#
# def custom_plot_results(results, img, names):
#     """
#     自定义绘制函数，只显示标签、置信度和边界框，去掉背景颜色。
#     """
#     for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
#         x1, y1, x2, y2 = map(int, xyxy)
#         label = f'{names[int(cls)]} {conf:.2f}'
#         color = (255, 255, 255)  # 白色文字
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # 绘制边界框
#         cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 绘制标签和置信度
#
#
# def main():
#     # 加载模型，split='test'利用测试集进行测试
#     model = YOLOv10(r"runs/train/yolov10n-NWD-SPPCSPC'/weights/best.pt")
#
#     # 模型验证
#     results = model.val(data="coco.yaml",
#                         split='test',
#                         imgsz=640,
#                         batch=16,
#                         device=0,
#                         workers=0,
#                         project="runs/detect/NWD-SPPCSPC"
#                         )
#
#     # 获取类别名称
#     names = model.names
#
#     # 遍历测试集中的图像并绘制结果
#     for img_path in results.paths:
#         img = cv2.imread(img_path)
#         results = model(img)
#
#         # 清除默认的可视化信息
#         img = np.copy(img)  # 创建图像的副本，避免修改原始图像
#
#         # 自定义绘制检测结果
#         custom_plot_results(results, img, names)
#
#         # 显示图像
#         cv2.imshow('YOLOv10 Detection', img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()

from ultralytics import YOLOv10
import cv2
import numpy as np
import os


def custom_plot_results(results, img, names):
    """
    论文标准的检测结果绘制函数
    特点：
    1. 字体：使用无衬线字体（论文常用），抗锯齿处理
    2. 字号：标准化大小（0.6），线宽（2）符合论文配图规范
    3. 颜色：白色文字+黑色描边（提升可读性，论文图常用）
    4. 坐标：边界框精准，标签位置不越界
    5. 无多余背景，仅保留边界框和文字标注
    """
    # 论文标准参数配置
    font = cv2.FONT_HERSHEY_SIMPLEX  # 无衬线字体，论文常用
    font_scale = 0.6  # 标准化字号（论文配图建议0.5-0.7）
    font_thickness = 2  # 字体线宽（保证打印清晰）
    box_thickness = 2  # 边界框线宽
    text_color = (255, 255, 255)  # 白色文字（对比鲜明）
    text_bg_color = (0, 0, 0)  # 黑色描边（提升文字可读性）
    offset = 10  # 文字离边界框的距离

    # 遍历检测结果
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = map(int, xyxy)
        cls_idx = int(cls)
        class_name = names[cls_idx]
        # 论文标准标签格式：类别 置信度（保留2位小数）
        label = f'{class_name} {conf:.2f}'

        # 1. 绘制边界框（论文级精准绘制）
        cv2.rectangle(img, (x1, y1), (x2, y2), text_color, box_thickness)

        # 2. 计算文字位置（避免越界）
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_x = x1
        text_y = y1 - offset if (y1 - offset) > text_h else y1 + text_h + offset

        # 3. 绘制文字描边（论文图常用，提升文字辨识度）
        # 先画黑色描边
        cv2.putText(img, label, (text_x, text_y), font, font_scale, text_bg_color, font_thickness + 1, cv2.LINE_AA)
        # 再画白色文字（抗锯齿）
        cv2.putText(img, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)


def main():
    # 修复原代码路径引号错误（单引号转义问题）
    model_path = r"runs/train/yolov10n-NWD-SPPELAN-MLLA/weights/best.pt"
    # 加载YOLOv10模型
    model = YOLOv10(model_path)

    # 模型验证（生成测试集评估结果）
    val_results = model.val(
        data="coco.yaml",
        split='test',
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        project="runs/detect/ALL",
        save=False  # 关闭默认保存，避免冗余文件
    )

    # 获取类别名称（论文级标注需要准确的类别名）
    class_names = model.names

    # 创建保存论文图的目录（可选，建议保存而非仅显示）
    save_dir = "runs/paper_figures"
    os.makedirs(save_dir, exist_ok=True)

    # 遍历测试集图像并绘制论文级结果
    for idx, img_path in enumerate(val_results.paths):
        # 读取图像（确保无损读取）
        img = cv2.imread(img_path)
        if img is None:
            print(f"跳过无法加载的图像: {img_path}")
            continue

        # 模型推理
        detect_results = model(img, imgsz=640, verbose=False)

        # 深拷贝图像，避免修改原图
        img_copy = np.copy(img)

        # 应用论文标准的可视化
        custom_plot_results(detect_results[0], img_copy, class_names)

        # 1. 显示图像（调试用）
        cv2.imshow('YOLOv10 Paper-Grade Detection', img_copy)
        cv2.waitKey(0)  # 按任意键切换下一张

        # 2. 保存论文级图像（建议保存为PNG，无损压缩）
        img_name = os.path.basename(img_path)
        save_path = os.path.join(save_dir, f"paper_{img_name}")
        cv2.imwrite(save_path, img_copy)
        print(f"论文级图像已保存: {save_path}")

    # 释放窗口资源
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
