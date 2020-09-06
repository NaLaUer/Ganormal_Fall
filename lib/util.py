from tqdm import tqdm
import os
import cv2
import numpy as np
from lib.model import GANomaly3D
import torch



import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)




def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    return np.array(frame).astype(np.uint8)


def load_model (args, checkpoint):
    model = GANomaly3D(args)
    checkpoint = torch.load(
        checkpoint,
        map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()

    return model

def video_reading (model, video):
    cap = cv2.VideoCapture(video)
    retaining = True
    # 该参数是MPEG-4编码类型，后缀名为avi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = os.path.join('output.mp4')
    out = cv2.VideoWriter(output, fourcc, 20.0, (320, 240))
    clip = []
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue

        tmp_ = center_crop(cv2.resize(frame, (64, 64)))
        clip.append(tmp_)
        if len(clip) == 4:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))

            inputs = torch.from_numpy(inputs)
            inputs = inputs.cuda()
            with torch.no_grad():
                z, z_ = model.forward(inputs)

            result = abs(z - z_).view(100,)
            result = torch.mean(result) / 4



            cv2.putText(frame, "Threshold: %.4f" % result, (10, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (138, 43, 226), 2)


            clip.pop(0)

        if retaining == True:
            out.write(frame)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()