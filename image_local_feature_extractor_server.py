import numpy as np
import cv2
from hloc.extractors import superpoint

import torch
import zmq
import time
import argparse
import sys

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--match_model',
    type=str,
    default='superglue',
    help='feature match model, now support super glue and light glue'
  )
  parser.add_argument(
    '--lightglue_path',
    type=str,
    default='/home/guo/code/LightGlue-ONNX/lightglue',
    help='path to lightglue model'
  )
  parser.add_argument(
    '--model_weight',
    type=str,
    default='/home/guo/code/LightGlue-ONNX/weights/superpoint_lightglue.onnx',
    help='path to weight of matcher. needed by lightglue matcher'
  )
  return parser

class LocalFeatureServer:
  def __init__(self, match_model, ext_path, model_weight, cam_model):
    for p in ext_path:
      sys.path.append(p)
    self.method = match_model
    if match_model == 'superglue':
      from hloc.matchers import superglue
      self.matcher = superglue.SuperGlue(superglue.SuperGlue.default_conf).eval().to('cuda')
    elif match_model == 'lightglue':
      from lightglue import LightGlue
      conf = LightGlue.default_conf
      conf['weights'] = model_weight
      # TODO: how to config the lightglue model
      self.matcher = LightGlue('superpoint').eval().to('cuda')
    
    self.extractor = superpoint.SuperPoint(superpoint.SuperPoint.default_conf).eval().to('cuda')

    self.cam_model = cam_model

    self.last_img = None
    # self.last_tensor = None
    self.last_desc = None
    self.last_trackid = []

    self.track_cnt = 0 #track id for next new feature
    self.img_cnt = 0

    self.track_db = {}

    self.context = zmq.Context()
    self.socket = self.context.socket(zmq.REP)
    zmq_port = "ipc:///tmp/zmq_spsg"
    self.socket.bind(zmq_port)
    print(f"image sp sg node start listen on port: {zmq_port}")

    print('local feature server init finished!')
  
  def update_track_db(self, cur_frame_id, cur_frame_pt, clean_obsolete_track = False):
    for id, pt in zip(cur_frame_id, cur_frame_pt):
      if id in self.track_db:
        self.track_db[id].append(pt)
      else:
        self.track_db[id] = [pt]
    cur_frame_id_set = set(cur_frame_id)
    if clean_obsolete_track:
      delete_track = []
      for k, v in self.track_db.items():
        if k not in cur_frame_id_set:
          delete_track.append(k)
      
      for k in delete_track:
        self.track_db.pop(k)

  def show_tracks(self, cur_frame_img):
    if cur_frame_img.shape[-1] != 3:
      viz = cv2.cvtColor(cur_frame_img, cv2.COLOR_GRAY2BGR)
    else:
      viz = cur_frame_img
    tracks_num = 0
    tracks_sum = 0
    for id, pts in self.track_db.items():
      if len(pts) == 1:
        continue
      for i in range(len(pts) - 1):
        pt0 = pts[i].astype('int')
        pt1 = pts[i+1].astype('int')
        cv2.circle(viz, pt0, 1, (255, 0, 0), 2)
        cv2.line(viz, pt0, pt1, (0, 0, 255), 1, cv2.LINE_AA)
      cv2.circle(viz, pts[-1].astype('int'), 1, (255, 0, 0), 2)
      tracks_sum += len(pts)
      tracks_num += 1
    print('mean track len: {}'.format(tracks_sum / tracks_num))
    cv2.imshow('tracks', viz)
    cv2.waitKey(1)

  def send_feature(self, cur_frame_ids, cur_frame_kps, cur_frame_desc):

    cur_frame_id_float = cur_frame_ids.astype('float32').reshape(-1, 1)

    cur_frame_desc_T = cur_frame_desc.T
    print(cur_frame_id_float.shape, cur_frame_kps.shape, cur_frame_desc_T.shape)
    result = np.hstack((cur_frame_id_float, cur_frame_kps, cur_frame_desc_T))

    result_header = {
      "shape": result.shape,
      "dtype": str(result.dtype)
    }
    self.socket.send_json(result_header, zmq.SNDMORE)
    self.socket.send(result)


    # frame_id_header = {
    #   "shape": cur_frame_ids.shape,
    #   "dtype": str(cur_frame_ids.dtype)
    # }
    # self.socket.send_json(frame_id_header, zmq.SNDMORE)
    # self.socket.send(cur_frame_ids)

    # reply = self.socket.recv_string()

    # frame_kps_header = {
    #   "shape": cur_frame_kps.shape,
    #   "dtype": str(cur_frame_kps.dtype)
    # }
    # self.socket.send_json(frame_kps_header, zmq.SNDMORE)
    # self.socket.send(cur_frame_kps)

    # reply = self.socket.recv_string()

    # frame_desc_header = {
    #   "shape": cur_frame_desc.shape,
    #   "dtype": str(cur_frame_desc.dtype)
    # }
    # self.socket.send_json(frame_desc_header, zmq.SNDMORE)
    # self.socket.send(cur_frame_desc)

    # reply = self.socket.recv_string()

    # self.socket.send_string("feature tracking finished!")

  def run(self):

    while True:
      msg = self.socket.recv_json()
      print('got json from zmq: {}'.format(msg))
      shape = msg["shape"]
      dtype = np.dtype(msg["dtype"])
      data = self.socket.recv()
      
      print('got image data from zmq {}'.format(len(data)))
      start = time.time()
      img = np.frombuffer(data, dtype=dtype).reshape(shape)
      self.img_cnt += 1

      q_tensor = torch.from_numpy(img).float() / 255.0
      q_tensor = q_tensor.unsqueeze(0).unsqueeze(0)
      with torch.no_grad():
        q_desc = self.extractor({"image": q_tensor.to('cuda', non_blocking=True)})
      print(q_desc['keypoints'][0].dtype)
      print(q_desc['descriptors'][0].dtype)

      if self.last_desc is None:
        self.last_img = img
        # self.last_tensor = q_tensor
        self.last_desc = q_desc
        self.last_trackid = [i for i in range(q_desc['keypoints'][0].shape[0])]
        self.track_cnt = len(self.last_trackid) + 1
        # self.socket.send_string("got image")
        self.send_feature(np.array(self.last_trackid, dtype='int'), q_desc['keypoints'][0].cpu().numpy(), q_desc['descriptors'][0].cpu().numpy())

        pts = q_desc['keypoints'][0].cpu().numpy()
        self.update_track_db(self.last_trackid, pts)

        continue
      viz = np.hstack((img, self.last_img))
      viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)
      print(img.shape)
      print(img.shape, img.dtype)
      # print(q_desc['keypoints'][0])
      # print(q_desc.keys())

      # t_tensor = self.last_tensor

      null_img = np.zeros_like(img)
      null_tensor = torch.from_numpy(null_img).float()
      null_tensor = null_tensor.unsqueeze(0).unsqueeze(0)

      t_desc = self.last_desc
      kps0 = q_desc['keypoints'][0].cpu().numpy() # new image
      kps1 = t_desc['keypoints'][0].cpu().numpy() # last image
      if self.method == 'superglue':
        match_input = {
          'image0': null_tensor,
          'keypoints0': q_desc['keypoints'][0].unsqueeze(0),
          'scores0': q_desc['scores'][0].unsqueeze(0),
          'descriptors0': q_desc['descriptors'][0].unsqueeze(0),
          'image1': null_tensor,
          'keypoints1': t_desc['keypoints'][0].unsqueeze(0),
          'scores1': t_desc['scores'][0].unsqueeze(0),
          'descriptors1': t_desc['descriptors'][0].unsqueeze(0)
        }
      elif self.method == 'lightglue':
        match_input = {
          'image0':{
            'keypoints':q_desc['keypoints'][0].unsqueeze(0),
            'descriptors':q_desc['descriptors'][0].transpose(1, 0).unsqueeze(0),
            'image': null_tensor
          },
          'image1':{
            'keypoints':t_desc['keypoints'][0].unsqueeze(0),
            'descriptors':t_desc['descriptors'][0].transpose(1, 0).unsqueeze(0),
            'image': null_tensor
          }
        }
        # print(match_input['image0']['keypoints'].shape)
        # print(match_input['image0']['descriptors'].shape)
      with torch.no_grad():
        match = self.matcher(match_input)
      # self.last_tensor = q_tensor
      self.last_desc = q_desc
      self.last_img = img
      mt0 = match['matches0'].cpu().numpy()[0]
      # print(mt0)
      pts0 = []
      pts1 = []
      remain_id_after_match1 = []
      remain_id_after_match2 = []
      cur_frame_trackid = []
      for idx1, idx2 in enumerate(mt0):
        if idx2 == -1:
          continue
        remain_id_after_match1.append(idx1) # frame id, not track id
        pts0.append(kps0[idx1])
        remain_id_after_match2.append(idx2)
        pts1.append(kps1[idx2])
      raw_match = len(pts0)
      pts0 = np.array(pts0, dtype='float32').reshape(-1, 1, 2)
      pts1 = np.array(pts1, dtype='float32').reshape(-1, 1, 2)
      print(pts0.shape)
      remain_id_after_ransac1 = []
      remain_id_after_ransac2 = []
      if pts0.shape[0] > 0:
        pts0_n = self.cam_model.undistort_points(pts0)
        pts1_n = self.cam_model.undistort_points(pts1)
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
            remain_id_after_ransac1.append(remain_id_after_match1[i])
            remain_id_after_ransac2.append(remain_id_after_match2[i])
            pt0 = pts0.astype('int')[i][0]
            pt1 = pts1.astype('int')[i][0]
            cv2.circle(viz, pt0, 2, (255, 0, 0), 2)
            cv2.circle(viz, (pt1[0] + img.shape[1], pt1[1]), 2, (255, 0, 0), 2)
            cv2.line(viz, pt0, (pt1[0] + img.shape[1], pt1[1]), (0, 0, 255), 1, cv2.LINE_AA)
        
      cur_frame_id_to_last_frame_id = {}
      for id1, id2 in zip(remain_id_after_ransac1, remain_id_after_ransac2):
        # print('{} --> {}'.format(id1, id2))
        cur_frame_id_to_last_frame_id[id1] = id2

      for i in range(len(kps0)):
        if i in cur_frame_id_to_last_frame_id:
          cur_frame_trackid.append(self.last_trackid[cur_frame_id_to_last_frame_id[i]])
        else:
          cur_frame_trackid.append(self.track_cnt)
          self.track_cnt += 1

      self.update_track_db(cur_frame_trackid, kps0, True)
      self.show_tracks(img)
      self.last_trackid = cur_frame_trackid

      self.send_feature(np.array(cur_frame_trackid, dtype='int'), q_desc['keypoints'][0].cpu().numpy(), q_desc['descriptors'][0].cpu().numpy())

      print('{} image got, track number: {}, {} tracked, {} new assigned'.format(self.img_cnt, self.track_cnt, len(cur_frame_id_to_last_frame_id), len(kps0) - len(cur_frame_id_to_last_frame_id)))
      print('{} new feat per image'.format(self.track_cnt / self.img_cnt))
      cv2.imshow('match', viz)
      k = cv2.waitKey(1)


def main():

  parser = get_args()

  args = parser.parse_args()

  from camera.camera_model import EquiDistCamera, RadTanCamera
  from dataset.mynteye_left import load_mynteye

  cam_param = load_mynteye()
  cam_model = EquiDistCamera(np.array(cam_param['cam_mat'],dtype='float32'), np.array(cam_param['distortion'], dtype='float32'), np.array(cam_param['img_size_wh']))

  server = LocalFeatureServer(args.match_model, [args.lightglue_path], args.model_weight, cam_model)

  server.run()

if __name__ == '__main__':
  main()