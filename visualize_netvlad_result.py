import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import os.path as osp
import cv2
result_path = '/media/guo/fs/Hierarchical-Localization/night_netvlad_top_10.txt'
query_root = '/media/guo/fs/Hierarchical-Localization/office_night_query'
train_root = '/media/guo/fs/Hierarchical-Localization/office'

df = pd.read_csv(result_path, delimiter=' ', names=['query', 'train', 'score'])

print(df.head(10))

query_imgs = pd.unique(df['query'].sort_values())
# print(query_imgs)
for i in range(1000):
  # q=random.choice(query_imgs)
  q=query_imgs[i]
  print('q:', q)
  t0=df[df['query'] == q]['train'].iloc[0]
  t1=df[df['query'] == q]['train'].iloc[1]
  t2=df[df['query'] == q]['train'].iloc[2]
  score = df[df['query'] == q]['score'].iloc[0]
  score1 = df[df['query'] == q]['score'].iloc[1]
  score2 = df[df['query'] == q]['score'].iloc[2]
  # print(score)
  print('t_max:', t0)
  # print(t['train'])
  # print(t['train'].iloc[0])

  q_path=osp.join(query_root, q)
  t0_path=osp.join(train_root, t0)

  q_img = cv2.imread(q_path)
  t_img = cv2.imread(t0_path)

  t1_path=osp.join(train_root, t1)
  t1_img = cv2.imread(t1_path)
  t2_path=osp.join(train_root, t2)
  t2_img = cv2.imread(t2_path)

  viz = np.hstack((q_img, t_img))

  bt = np.hstack((t1_img, t2_img))

  viz = np.vstack((viz, bt))

  cv2.putText(viz, str(score), (10 + q_img.shape[1], 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

  cv2.putText(viz, str(score1), (10 , 30 + q_img.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

  cv2.putText(viz, str(score2), (10 + q_img.shape[1], 30 + q_img.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

  cv2.putText(viz, 'query', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

  # cv2.putText(viz, 'train0', (10 + q_img.shape[1], 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

  cv2.imshow('pair', viz)

  

  k = cv2.waitKey(100000)
  print(k)

  if k == 113: #'q'
    break
