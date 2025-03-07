import argparse
import os.path as osp
from operator import itemgetter
from typing import Optional, Tuple
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk import edit_distance
# from paddleocr import PaddleOCR, draw_ocr
import numpy as np
from PIL import Image
import time
import logging
from deepface import DeepFace
import tempfile


##### celebrity id score #####
def calculate_celebrity_id_score(video_path, img_paths):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Extract frames from the video
    frames = []
    face_verify_score_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame)
        frames.append(frame)
    cap.release()

    face_images = img_paths
    # Save the frame as a temporary file
    for i in range(len(frames)):
        frame = frames[i]
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_frame_file:
            frame_pil = Image.fromarray(frame)
            frame_pil.save(temp_frame_file.name)
            temp_frame_file.flush()  # Make sure the file is written to disk

        distance = []
        for j in range(len(face_images)): 
            face_gt = face_images[j]
            # Calculate the distance using DeepFace.verify() with the temporary file paths
            distance.append(DeepFace.verify(img1_path=face_gt, img2_path=temp_frame_file.name, enforce_detection = False)['distance'])
        face_verify_score_frames.append(min(distance))
        # Delete the temp_frame_file from the local disk
        os.remove(temp_frame_file.name)
    face_verify_score_frames = np.array(face_verify_score_frames)
    face_verify_score_avg = np.mean(face_verify_score_frames).item()
    
    return face_verify_score_avg

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_videos", type=str, default='', help="Specify the path of generated videos")
    parser.add_argument("--metric", type=str, default='celebrity_id_score', help="Specify the metric to be used")
    parser.add_argument("--output_path", type=str, default='../../results/', help="output_path")


    args = parser.parse_args()
    print(args)

    dir_videos = args.dir_videos
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    out_path = args.output_path + '/{}.tsv'.format(args.metric)
    metric = args.metric

    #dir_prompts =  '../../prompts/'
   
    video_paths = [os.path.join(dir_videos, x) for x in os.listdir(dir_videos)]
    #prompt_paths = [os.path.join(dir_prompts, os.path.splitext(os.path.basename(x))[0]+'.txt') for x in video_paths]

    # Load pretrained models
    device =  "cpu"


    image_paths = {}

 
    # Calculate SD scores for all video-text pairs
    scores = []
    # Create the directory if it doesn't exist
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"../../results", exist_ok=True)
    # Set up logging
    log_file_path = f"../../results/{metric}_record.txt"
    # Delete the log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # File handler for writing logs to a file
    file_handler = logging.FileHandler(filename=f"../../results/{metric}_record.txt")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    # Stream handler for displaying logs in the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)

    results_list = []

    for i in tqdm(range(len(video_paths))):
        video_path = video_paths[i]
        #prompt_path = prompt_paths[i]
        basename = os.path.basename(video_path)[:4]
        score = calculate_celebrity_id_score(video_path, image_paths[basename])

        results_list.append({
            "video_path": video_path.split("/")[-1],
            "celebrity_id_score": score,
        })


    df = pd.DataFrame(results_list)
    df.to_csv(out_path, sep='\t', index=False)

            
    # Calculate the average SD score across all video-text pairs

