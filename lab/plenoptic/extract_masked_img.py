import cv2
import numpy as np
import pickle
import torch
import io

img = cv2.imread("./data/video_capture/taekgyun_jpg/000.jpg")

# -----------------------------------
#      display original image
# -----------------------------------

# cv2.imshow("original img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# -----------------------------------
#           load pickle
# -----------------------------------

path = './data/model_output/'
output_path = path + 'transfiner_taekgyun_output/000.pickle'

# mask = pickle.load(open(output_path, "rb"))

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

with open(output_path, "rb") as f:
    contents = CPU_Unpickler(f).load()
print(contents)