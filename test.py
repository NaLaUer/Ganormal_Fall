from lib.util import *
from parse.parse import parse_args
import sys
import os
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()

    print("device is available : ", torch.cuda.is_available())
    print("Start.....")

    checkpoint = os.path.join('img', args.checkpoint)

    model = load_model(args, checkpoint)

    video = os.path.join("input2.avi")

    video_reading(model, video)
    print("Finish")


if __name__ == '__main__':
    main()
