from hloc.extractors.netvlad import NetVLAD

import cv2
import numpy as np
import torch
test_img_path = '/home/guo/l.jpg'

img=cv2.imread(test_img_path)

# cv2.imshow('img', img)
# cv2.waitKey(10000)
print(img.shape)
img=img.transpose([2, 0, 1])
img=np.expand_dims(img, axis=0)
img = img/255.0
print(img.shape, img.dtype)
img_tensor = torch.tensor(img.astype('float32'))
data = {'image':img_tensor}
netvlad_conf = {
  'model_name': 'VGG16-NetVLAD-Pitts30K',
  'whiten': True
}
netvlad = NetVLAD(netvlad_conf)
desc = netvlad(data)

print(desc)
print(desc['global_descriptor'].shape)