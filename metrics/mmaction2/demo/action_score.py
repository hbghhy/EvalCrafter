# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from operator import itemgetter
from typing import Optional, Tuple

from mmengine import Config, DictAction

from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.visualization import ActionVisualizer
import pandas as pd

import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import time
import logging
# import wandb
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
from torchvision.transforms import Resize
from torchvision.utils import save_image
from diffusers import StableDiffusionXLPipeline
import requests
from transformers import AutoProcessor, Blip2ForConditionalGeneration


###### action score ######
def get_output(
    video_path: str,
    out_filename: str,
    data_sample: str,
    labels: list,
    fps: int = 30,
    font_scale: Optional[str] = None,
    font_color: str = 'white',
    target_resolution: Optional[Tuple[int]] = None,
) -> None:
    """Get demo output using ``moviepy``.

    This function will generate video file or gif file from raw video or
    frames, by using ``moviepy``. For more information of some parameters,
    you can refer to: https://github.com/Zulko/moviepy.

    Args:
        video_path (str): The video file path.
        out_filename (str): Output filename for the generated file.
        datasample (str): Predicted label of the generated file.
        labels (list): Label list of current dataset.
        fps (int): Number of picture frames to read per second. Defaults to 30.
        font_scale (float): Font scale of the text. Defaults to None.
        font_color (str): Font color of the text. Defaults to ``white``.
        target_resolution (Tuple[int], optional): Set to
            (desired_width desired_height) to have resized frames. If
            either dimension is None, the frames are resized by keeping
            the existing aspect ratio. Defaults to None.
    """

    if video_path.startswith(('http://', 'https://')):
        raise NotImplementedError

    # init visualizer
    out_type = 'gif' if osp.splitext(out_filename)[1] == '.gif' else 'video'
    visualizer = ActionVisualizer()
    visualizer.dataset_meta = dict(classes=labels)

    text_cfg = {'colors': font_color}

    visualizer.add_datasample(
        out_filename,
        video_path,
        data_sample,
        draw_pred=True,
        draw_gt=False,
        text_cfg=text_cfg,
        fps=30,
        out_type=out_type,
        out_path=osp.join('demo', out_filename),
        target_resolution=target_resolution)


def calculate_action_score(video_path, action_model):
    pred_result = inference_recognizer(action_model, video_path)

    pred_scores = pred_result.pred_scores.item.tolist()
    score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
    top5_label = score_sorted[:1]

    label = '../../../metrics/mmaction2/tools/data/kinetics/label_map_k400.txt'
    labels = open(label).readlines()
    labels = [x.strip() for x in labels]
    results = [(labels[k[0]], k[1]) for k in top5_label]

    print('The top-5 labels with corresponding scores are:')
    confidence = []
    action_pred = []
    for result in results:
        print(f'{result[0]}: ', result[1])
        action_pred.append(result[0])
        confidence.append(result[1])

    return action_pred[0], confidence[0]


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_videos", type=str, default='', help="Specify the path of generated videos")
    parser.add_argument("--metric", type=str, default='celebrity_id_score', help="Specify the metric to be used")
    parser.add_argument('--output_path', help='output directory')
    args = parser.parse_args()

    dir_videos = args.dir_videos
    metric = args.metric
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)    
    out_path = args.output_path + '/{}.tsv'.format(metric)

   
    video_paths = [os.path.join(dir_videos, x) for x in os.listdir(dir_videos)]

     # Create the directory if it doesn't exist
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"../../../results", exist_ok=True)
    # Set up logging
    log_file_path = f"../../../results/{metric}_record.txt"
    # Delete the log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # File handler for writing logs to a file
    file_handler = logging.FileHandler(filename=f"../../../results/{metric}_record.txt")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    # Stream handler for displaying logs in the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)

    # Load pretrained models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # action_model 
    config = '../../../metrics/mmaction2/configs/recognition/videomaev2/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400.py'
    checkpoint = '../../../checkpoints/VideoMAE/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth'
    cfg = Config.fromfile(config)
    # Build the recognizer from a config file and checkpoint file/url
    action_model = init_recognizer(cfg, checkpoint, device=device)
    # get the videos' basenames list action_vid  for recognition
    # Load the JSON data from the file

    # Calculate SD scores for all video-text pairs
    scores = []
    results_list = []

    for i in tqdm(range(len(video_paths))):
        video_path = video_paths[i]
        # get the videos' basenames list action_vid  for recognition
        basename = os.path.basename(video_path)[:4]
        act, score = calculate_action_score(video_path, action_model)
        print(act, score)
        if score is not None:
            scores.append(score)
            average_score = sum(scores) / len(scores)
            logging.info(f"Vid: {os.path.basename(video_path)[:4]},  Current {metric}: {score}, Current avg. {metric}: {average_score}")
                # wandb.log({
                #     f"Current {metric}": score,
                #     f"Average {metric}": average_score,
                # })
            
            results_list.append({
                "video_path": video_path.split("/")[-1],
                "{}".format(metric): score,
                'action_pred': act
            })

    # Calculate the average SD score across all video-text pairs
    df = pd.DataFrame(results_list)
    df.to_csv(out_path, sep='\t', index=False)