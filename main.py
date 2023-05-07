import os
import sys

import cv2
import time
import torch
import numpy as np
from PIL import ImageFont
from loguru import logger
from torchvision import transforms
from argparse import ArgumentParser
from utils.datasets import letterbox
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, output_to_dict
from trainer.plank_trainer import planks_assessment, plot_planks_skeleton


def parse_args():
    """
    Parse the arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of model to use", required=True,
                        choices=['yolov7-pose', 'mediapipe'])
    parser.add_argument("--workout", type=str, help="Name of workout for posture assessment", required=True,
                        choices=['planks'])
    args = parser.parse_args()
    return args


def select_model(model):
    if model == 'yolov7-pose':
        model_path = 'models/yolov7-w6-pose.pt'
        return model_path


def main(args):
    data: str = 'data'
    device = select_device()
    logger.info(f"Selected {device} for inference")
    model_path: str = select_model(args.model)
    logger.info(f"Loading model from {model_path}")
    model = attempt_load(weights=model_path, map_location=device)
    _ = model.eval()
    data_workout: str = os.path.join(data, args.workout)
    all_videos: list = [os.path.join(data_workout, x) for x in os.listdir(data_workout) if
                        os.path.splitext(x)[1] in ['.mp4']]
    for input_path in all_videos:
        logger.info(f"Performing inference on {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error('Error while trying to read video. Please check path again')
            sys.exit()
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        vid_write_image = letterbox(original_img=cap.read()[1], new_shape=frame_width, stride=64, auto=True)
        resize_height, resize_width = vid_write_image[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"{os.path.basename(os.path.splitext(input_path)[0])}_output.mp4", fourcc, 30, (resize_width, resize_height))

        frame_count, total_fps = 0, 0
        fontpath = "data/sfpro.ttf"
        font = ImageFont.truetype(fontpath, 32)
        font1 = ImageFont.truetype(fontpath, 160)
        frame_list: list = []
        while cap.isOpened:
            logger.debug(f"Frame {frame_count} Processing")
            ret, frame = cap.read()
            if ret:
                orig_image = frame
                # Preprocess Image
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                image = letterbox(original_img=image, new_shape=frame_width,
                                  stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))

                image = image.to(device)
                image = image.float()
                start_time = time.time()

                with torch.no_grad():
                    output, _ = model(image)
                output = non_max_suppression_kpt(prediction=output, conf_thres=0.5, iou_thres=0.65, nc=model.yaml['nc'],
                                                 nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)

                # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if output.shape[0] > 1:
                    logger.warning("Make sure you don't have any other person in the camera view")
                    is_correct, response = False, "Multiple Person on the view"
                    result = None
                    pass
                elif output.shape[0] == 0:
                    logger.warning("No key points could be detected. Make sure camera is in a good position.")
                    is_correct, response = False, "Poor camera angle"
                    result = None
                    pass
                else:
                    output_dict = output_to_dict(frame_no=frame_count, output=output)
                    result = output[0, 7:].T
                    if args.workout == "planks":
                        is_correct, response = planks_assessment(result=output_dict)
                    else:
                        is_correct, response = False, "Please Select your workout"
                plot_planks_skeleton(image=img, result=result, is_correct=is_correct, response=response)

                end_time = time.time()
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                out.write(img)
            else:
                break
        out.release()
        # with VideoFileClip(f"{out_video_name}_result4.mp4", fps=total_fps) as clip:
        #     clip.write_videofile(f"{out_video_name}_result4_converted.mp4")


if __name__ == '__main__':
    main(args=parse_args())
