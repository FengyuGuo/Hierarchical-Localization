from hloc.extractors.netvlad import NetVLAD
from hloc.extractors.megaloc import MegaPlaces
import cv2
import numpy as np
import torch
test_img_path = '/media/guo/fs/Hierarchical-Localization/office_dog_mynteye/frame000628.png'

img=cv2.imread(test_img_path)

# cv2.imshow('img', img)
# cv2.waitKey(10000)
print(img.shape)
img=img.transpose([2, 0, 1])
img=np.expand_dims(img, axis=0)
img = img/255.0
print(img.shape, img.dtype)
img_tensor = torch.tensor(img.astype('float32')).to('cuda')
data = {'image':img_tensor}
netvlad_conf = {
  'model_name': 'VGG16-NetVLAD-Pitts30K',
  'whiten': True
}
netvlad = NetVLAD(netvlad_conf).eval().to('cuda')
desc = netvlad(data)



print(desc)
print(desc['global_descriptor'].dtype)
print(desc['global_descriptor'].shape)

mega = MegaPlaces(MegaPlaces.default_conf).eval().to('cuda')

desc = mega(data)
print(desc['global_descriptor'].cpu().detach().numpy()[0])
print(desc['global_descriptor'].dtype)
print(desc['global_descriptor'].shape)
