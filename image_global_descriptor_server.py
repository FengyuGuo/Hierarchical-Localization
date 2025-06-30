# running in lightglue conda env
from hloc.extractors.megaloc import MegaPlaces
import cv2
import numpy as np
import torch
import zmq
import time

mega = MegaPlaces(MegaPlaces.default_conf).eval().to('cuda')

context = zmq.Context()
socket = context.socket(zmq.REP)
zmq_port = "ipc:///tmp/zmq_global_descriptor"
socket.bind(zmq_port)

print("image global descriptor node start listenning!")

while True:
  
  msg = socket.recv_json()

  shape = msg["shape"]
  dtype = np.dtype(msg["dtype"])
  data = socket.recv()
  start = time.time()
  img = np.frombuffer(data, dtype=dtype).reshape(shape)

  print(img.shape)
  img=img.transpose([2, 0, 1])
  img=np.expand_dims(img, axis=0)
  img = img/255.0
  print(img.shape, img.dtype)
  img_tensor = torch.tensor(img.astype('float32')).to('cuda')
  data = {'image':img_tensor}

  desc = mega(data)

  desc = desc['global_descriptor'].cpu().detach().numpy()
  reply_header = {"shape": desc.shape, "dtype": str(desc.dtype)}
  socket.send_json(reply_header, zmq.SNDMORE)
  reply_data = desc.tobytes()
  socket.send(reply_data)
  end = time.time()
  print(desc)
  print(desc.min(), desc.max())
  print(f"运行时间: {end - start:.6f} 秒")