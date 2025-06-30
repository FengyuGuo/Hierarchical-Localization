import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import os.path as osp
import cv2
from hloc.extractors import superpoint
from hloc.matchers import superglue
import torch
import argparse

from camera.camera_model import EquiDistCamera, RadTanCamera
from dataset.mynteye_left import load_mynteye
from dataset.dog import load_param


# example:
# python visualize_netvlad_result.py --result_txt /media/guo/fs/Hierarchical-Localization/dog_match_megaloc_top20.txt --query_img_path /media/guo/fs/Hierarchical-Localization/office_dog_real --train_img_path /media/guo/fs/Hierarchical-Localization/office_dog_d435 --dataset d435i --try_match --start_idx 100

parser = argparse.ArgumentParser()

parser.add_argument(
  '--result_txt',
  type=str,
  required=True,
  help='path to the result with match pairs and scores'
)

parser.add_argument(
  '--query_img_path',
  type=str,
  required=True,
  help='root dir of query images'
)

parser.add_argument(
  '--train_img_path',
  type=str,
  required=True,
  help='root dir of train images'
)

parser.add_argument(
  '--dataset',
  type=str,
  help='dataset param to do ransac'
)

parser.add_argument(
  '--try_match',
  default=False,
  action='store_true',
  help='extract and match feature points'
)

parser.add_argument(
  '--start_idx',
  default=0,
  type=int,
  help='start index of query index'
)

args = parser.parse_args()

# result_path = '/media/guo/fs/Hierarchical-Localization/night_netvlad_top_10.txt'
# query_root = '/media/guo/fs/Hierarchical-Localization/office_night_query'
# train_root = '/media/guo/fs/Hierarchical-Localization/office'

result_path = args.result_txt
query_root = args.query_img_path
train_root = args.train_img_path

if args.dataset == 'mynteye':
  cam_param = load_mynteye()
  cam_model = EquiDistCamera(np.array(cam_param['cam_mat'],dtype='float32'), np.array(cam_param['distortion'], dtype='float32'), np.array(cam_param['img_size_wh']))
elif args.dataset == 'd435i':
  cam_param = load_param()
  cam_model = RadTanCamera(np.array(cam_param['cam_mat'],dtype='float32'), np.array(cam_param['distortion'], dtype='float32'), np.array(cam_param['img_size_wh']))
sp = superpoint.SuperPoint(superpoint.SuperPoint.default_conf).eval().to('cuda')
sg = superglue.SuperGlue(superglue.SuperGlue.default_conf).eval().to('cuda')

df = pd.read_csv(result_path, delimiter=' ', names=['query', 'train', 'score'])

print(df.head(10))

query_imgs = pd.unique(df['query'].sort_values())
# print(query_imgs)
for i in range(args.start_idx, len(query_imgs)):
  print('index: ', i)
  q=query_imgs[i]
  print('q:', q)
  t0=df[df['query'] == q]['train'].iloc[0]
  t1=df[df['query'] == q]['train'].iloc[1]
  t2=df[df['query'] == q]['train'].iloc[2]
  score = df[df['query'] == q]['score'].iloc[0]
  score1 = df[df['query'] == q]['score'].iloc[1]
  score2 = df[df['query'] == q]['score'].iloc[2]
  print('t_max:', t0)
  q_path=osp.join(query_root, q)
  t0_path=osp.join(train_root, t0)

  q_img = cv2.imread(q_path, cv2.IMREAD_GRAYSCALE)
  t_img = cv2.imread(t0_path, cv2.IMREAD_GRAYSCALE)

  t1_path=osp.join(train_root, t1)
  t1_img = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
  t2_path=osp.join(train_root, t2)
  t2_img = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)

  viz = np.hstack((q_img, t_img))

  bt = np.hstack((t1_img, t2_img))

  viz = np.vstack((viz, bt))
  viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)

  if args.try_match:
    q_tensor = torch.from_numpy(q_img).float() / 255.0
    q_tensor = q_tensor.unsqueeze(0).unsqueeze(0)
    # print(img_tensor.shape)
    q_desc = sp({"image": q_tensor.to('cuda', non_blocking=True)})
    # print(q_desc['keypoints'][0])
    # print(q_desc.keys())
    t_tensor = torch.from_numpy(t_img).float() / 255.0
    t_tensor = t_tensor.unsqueeze(0).unsqueeze(0)
    t_desc = sp({"image": t_tensor.to('cuda', non_blocking=True)})
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
    mt0 = match['matches0'].cpu().numpy()[0]
    # print(mt0)
    pts0 = []
    pts1 = []
    for idx1, idx2 in enumerate(mt0):
      if idx2 == -1:
        continue
      pts0.append(kps0[idx1])
      pts1.append(kps1[idx2])
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
      if good_match > 0:
        for i, m in enumerate(mask.tolist()):
          if m == 0:
            continue
          good += 1
          pt0 = pts0.astype('int')[i][0]
          pt1 = pts1.astype('int')[i][0]
          cv2.circle(viz, pt0, 2, (255, 0, 0), 2)
          cv2.circle(viz, (pt1[0] + q_img.shape[1], pt1[1]), 2, (255, 0, 0), 2)
          cv2.line(viz, pt0, (pt1[0] + q_img.shape[1], pt1[1]), (0, 0, 255), 1, cv2.LINE_AA)
        good_match = np.count_nonzero(mask)
        print(good, good_match)
        cv2.putText(viz, 'sp sg match after ransac: '+str(good_match), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(viz, 'sp sg raw match: '+str(pts0.shape[0]), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
  cv2.putText(viz, str(score), (10 + q_img.shape[1], 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
  cv2.putText(viz, str(score1), (10 , 30 + q_img.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
  cv2.putText(viz, str(score2), (10 + q_img.shape[1], 30 + q_img.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
  cv2.putText(viz, 'query', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
  cv2.imshow('pair', viz)
  k = cv2.waitKey(100000)
  if k == 113: #'q'
    break
