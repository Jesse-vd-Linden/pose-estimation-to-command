import torch
import torchvision
import cv2
import numpy as np
from pprint import pprint

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# pprint(vars(model))
# pprint(dir(model))
# Images
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False



# Inference
while rval:
    results = model(frame)
    img = np.squeeze(np.array(results.ims))
    cv2.imshow("Object detection", img)
    rval, frame = vc.read()

    key = cv2.waitKey(1)
    if key == 27: # exit on ESC
        break

# img = np.squeeze(np.array(results.ims))
# print(img.shape)
# cv2.imwrite('image_test.jpg', img)
# # cv2.waitKey(10000)

vc.release()
cv2.destroyWindow("preview")

