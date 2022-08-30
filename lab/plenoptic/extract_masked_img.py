import cv2
import numpy as np
import pickle

img = cv2.imread("./data/video_capture/taekgyun_jpg/000.jpg")
cv2.imshow("original img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()