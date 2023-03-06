import torch
import cv2
from PIL import Image

model = torch.hub.load(r'C:\Users\ansel_chen\VScode_Ansel\pytorch_practice\yolov5', 'custom',
                       path=r'C:\Users\ansel_chen\VScode_Ansel\pytorch_practice\model\moto.pt', source='local', force_reload=True)
# 從cv2得到畫面
# img_path = r'C:\Users\ansel_chen\VScode_Ansel\pytorch_practice\torch_crop\01.JPG'
# img_cv2 = cv2.imread(img_path)

viedo = cv2.VideoCapture(
    r'C:\Users\ansel_chen\VScode_Ansel\pytorch_practice\torch_crop\test.mov')
num = 0
while True:
    num += 1
    ok, img_cv2 = viedo.read()
    if ok:
        img = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        results = model(img)
        plate = results.pandas().xyxy[0]
        cv2.imshow('detect plate', img_cv2)
        # img = img.rotate(-90, expand=True)
        # 取得車牌座標並擷取後存檔
        if plate.index.array.size > 0:
            print('detect plate')
            for i in range(plate.index.array.size):
                x = plate['xmin'][i]
                y = plate['ymin'][i]
                x2 = plate['xmax'][i]
                y2 = plate['ymax'][i]
                # print(x, y, x2, y2)
                cropped = img.crop((x, y, x2, y2))
                cropped.save(
                    f'C:/Users/ansel_chen/VScode_Ansel/pytorch_practice/torch_crop/test/palte{num}_{i}.jpg')
                print(f'save palte{num}_{i}.jpg success')
        else:
            print('no plate in img')

    else:
        print("Cannot receive frame")
        break

    if cv2.waitKey(5) == ord('q'):
        break    # 按下 q 鍵停止

##############################################################
