
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('/home/hgd-bme-li/wws/yolov10-main (1)/ultralytics/cfg/models/v10/yolov10n.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='coco.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                close_mosaic=0,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD AdamW
                project='runs/train',
                name='yolov10n-NWD-SPPCSPC',
                )
#
#
# from ultralytics import RTDETR
#
# if __name__ == '__main__':
#     model = RTDETR('cfg/models/v10/yolov10nRTDETR.yaml ')
#     model.train(data='coco.yaml',
#                 cache=False,
#                 imgsz=640,
#                 epochs=200,
#                 batch=16,
#                 close_mosaic=0,
#                 workers=8,
#                 device='0',
#                 optimizer='SGD',  # using SGD AdamW
#                 project='runs/train',
#                 name='MLLA-RTDETR-clo',
#                 )