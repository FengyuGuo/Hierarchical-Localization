import cv2
import numpy as np
import zmq
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import threading
import time
import Queue

context = zmq.Context()
socket = context.socket(zmq.REQ)
zmq_port = "ipc:///tmp/zmq_global_descriptor"
socket.connect(zmq_port)

image_queue = Queue.Queue()

server_idle = True


def worker(stop_event):
    global image_queue
    global server_idle
    while not stop_event.is_set():
      if image_queue.empty():
        time.sleep(0.001)
        continue
      cv_image =image_queue.get()
      while not image_queue.empty():
        image_queue.get()
      if cv_image is not None:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        data = cv_image.tobytes()
        msg = {'shape': cv_image.shape, 'dtype': str(cv_image.dtype)}
        print('image shape: {}'.format(cv_image.shape))
        print("send message to global desc server!")
        
        start = time.time()
        socket.send_json(msg, zmq.SNDMORE)
        socket.send(data)
        print("send msg finished!")
        reply_header = socket.recv_json()
        reply_data = socket.recv()
        desc = np.frombuffer(reply_data, dtype=reply_header["dtype"]).reshape(reply_header["shape"])
        end = time.time()

        print("running time: {}".format(end - start))
        print(desc.shape, desc.dtype)
        print(desc)
        print(desc.min(), desc.max())
        #TODO: publish the image descriptors with 

        server_idle = True
def img_cbk(msg):
  global image_queue

  bridge = CvBridge()
  try:
    recv_image = bridge.compressed_imgmsg_to_cv2(msg)
    image_queue.put(recv_image)
  except CvBridgeError as e:
    print(e)


rospy.init_node('image_global_descriptor', anonymous=True)

image_sub = rospy.Subscriber("/mynteye/left/image_raw/compressed", CompressedImage,img_cbk, queue_size=1)
stop_event = threading.Event()
t = threading.Thread(target=worker, args=(stop_event,))
t.start()

try:
  rospy.spin()
except KeyboardInterrupt:
  print("Shutting down.")

stop_event.set()
t.join()