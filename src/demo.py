# Code taken and adapted from:
# https://github.com/omasaht/headpose-fsanet-pytorch/tree/master/src


import numpy as np
import cv2
import onnxruntime
from pathlib import Path
# local imports
from face_detector import FaceDetector

root_path = str(Path(__file__).absolute().parent.parent)

def process_frame(frame, face_d, sess, sess2):
    face_bb = face_d.get(frame)
    head_poses = []
    for (x1, y1, x2, y2) in face_bb:
        face_roi = frame[y1:y2+1, x1:x2+1]

        # preprocess headpose model input
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = face_roi.transpose((2, 0, 1))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = (face_roi - 127.5) / 128
        face_roi = face_roi.astype(np.float32)

        # get headpose
        res1 = sess.run(["output"], {"input": face_roi})[0]
        res2 = sess2.run(["output"], {"input": face_roi})[0]

        yaw, pitch, roll = np.mean(np.vstack((res1, res2)), axis=0)
        head_poses.append((yaw, pitch, roll))
    
    return head_poses

def process_image(frame):
    face_d = FaceDetector()
    sess = onnxruntime.InferenceSession(f'{root_path}/pretrained/fsanet-1x1-iter-688590.onnx')
    sess2 = onnxruntime.InferenceSession(f'{root_path}/pretrained/fsanet-var-iter-688590.onnx')


    head_poses = process_frame(frame, face_d, sess, sess2)

    return head_poses