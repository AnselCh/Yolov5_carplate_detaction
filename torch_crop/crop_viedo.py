import torch
import cv2
from PIL import Image
# Model
# or yolov5m, yolov5l, yolov5x, custom
model = torch.hub.load(r'C:\Users\ansel_chen\VScode_Ansel\pytorch_practice\yolov5', 'custom',
                       path=r'C:\Users\ansel_chen\VScode_Ansel\pytorch_practice\torch_crop\best.pt', source='local', force_reload=True)

# Images
# or file, Path, PIL, OpenCV, numpy, list

viedo = cv2.VideoCapture(
    r'C:\Users\ansel_chen\VScode_Ansel\pytorch_practice\torch_crop\test1.mov')
# Inference
while True:
    ok, show = viedo.read()
    if ok:
        # results = model(frame)
        # show = results.crop()
        img = cv2.resize(show, (540, 300))

        cv2.imshow('detect plate', model(img))

    else:
        print("Cannot receive frame")
        break

    if cv2.waitKey(5) == ord('q'):
        break    # 按下 q 鍵停止
