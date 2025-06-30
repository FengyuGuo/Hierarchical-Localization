import numpy as np
import cv2
from hloc.extractors import superpoint
from hloc.matchers import superglue
import torch
import zmq
import time

sp = superpoint.SuperPoint(superpoint.SuperPoint.default_conf).eval().to('cuda')
sg = superglue.SuperGlue(superglue.SuperGlue.default_conf).eval().to('cuda')

context = zmq.Context()
socket = context.socket(zmq.REP)
zmq_port = "ipc:///tmp/zmq_spsg"
socket.bind(zmq_port)

print(f"image sp sg node start listen on port: {zmq_port}")

from camera.camera_model import EquiDistCamera, RadTanCamera
from dataset.mynteye_left import load_mynteye

cam_param = load_mynteye()
cam_model = EquiDistCamera(np.array(cam_param['cam_mat'],dtype='float32'), np.array(cam_param['distortion'], dtype='float32'), np.array(cam_param['img_size_wh']))

last_img = None
last_tensor = None
last_desc = None

while True:
  msg = socket.recv_json()
  print('got json from zmq: {}'.format(msg))
  shape = msg["shape"]
  dtype = np.dtype(msg["dtype"])
  data = socket.recv()
  
  print('got image data from zmq {}'.format(len(data)))
  start = time.time()
  img = np.frombuffer(data, dtype=dtype).reshape(shape)


  q_tensor = torch.from_numpy(img).float() / 255.0
  q_tensor = q_tensor.unsqueeze(0).unsqueeze(0)
  # print(img_tensor.shape)
  q_desc = sp({"image": q_tensor.to('cuda', non_blocking=True)})
  # print(q_desc['keypoints'], q_desc['scores'], q_desc['descriptors'])
  if last_img is None:
    last_img = img
    last_tensor = q_tensor
    last_desc = q_desc
    socket.send_string("got image")
    continue
  viz = np.hstack((img, last_img))
  viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)
  print(img.shape)
  print(img.shape, img.dtype)
  # print(q_desc['keypoints'][0])
  # print(q_desc.keys())

  t_tensor = last_tensor
  t_desc = last_desc
  kps0 = q_desc['keypoints'][0].cpu().numpy()
  kps1 = t_desc['keypoints'][0].cpu().numpy()
  match_input = {
    'image0': q_tensor,
    'keypoints0': q_desc['keypoints'][0].unsqueeze(0),
    'scores0': q_desc['scores'][0].unsqueeze(0),
    'descriptors0': q_desc['descriptors'][0].unsqueeze(0),
    'image1': t_tensor,
    'keypoints1': t_desc['keypoints'][0].unsqueeze(0),
    'scores1': t_desc['scores'][0].unsqueeze(0),
    'descriptors1': t_desc['descriptors'][0].unsqueeze(0)
  }
  match = sg(match_input)
  last_tensor = q_tensor
  last_desc = q_desc
  last_img = img
  mt0 = match['matches0'].cpu().numpy()[0]
  # print(mt0)
  pts0 = []
  pts1 = []
  for idx1, idx2 in enumerate(mt0):
    if idx2 == -1:
      continue
    pts0.append(kps0[idx1])
    pts1.append(kps1[idx2])
  raw_match = len(pts0)
  pts0 = np.array(pts0, dtype='float32').reshape(-1, 1, 2)
  pts1 = np.array(pts1, dtype='float32').reshape(-1, 1, 2)
  print(pts0.shape)
  if pts0.shape[0] > 0:
    pts0_n = cam_model.undistort_points(pts0)
    pts1_n = cam_model.undistort_points(pts1)
    F, mask = cv2.findFundamentalMat(pts0_n, pts1_n, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99, maxIters = 200)
    good = 0
    mask = np.squeeze(mask)
    print(mask.shape)
    good_match = np.count_nonzero(mask)
    print('good match: {}, raw match: {}'.format(good_match, raw_match))
    if good_match > 0:
      for i, m in enumerate(mask.tolist()):
        if m == 0:
          continue
        good += 1
        pt0 = pts0.astype('int')[i][0]
        pt1 = pts1.astype('int')[i][0]
        cv2.circle(viz, pt0, 2, (255, 0, 0), 2)
        cv2.circle(viz, (pt1[0] + img.shape[1], pt1[1]), 2, (255, 0, 0), 2)
        cv2.line(viz, pt0, (pt1[0] + img.shape[1], pt1[1]), (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('match', viz)
    k = cv2.waitKey(1)
    if k == 'q':
      break
  socket.send_string("got image")