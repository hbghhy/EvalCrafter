import os
import sys
import pandas as pd
import argparse
import numpy as np

def get_tsv_files(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tsv')]

def merge_tsv_files(tsv_files):
    dataframes = [pd.read_csv(f, sep='\t') for f in tsv_files]
    print(f"Merging {len(dataframes)} TSV files")
    print(f'all tsv files: {tsv_files}')
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on='video_path', how='outer')
    return merged_df

def post_process(merged_df):
    quality_weights = np.array([0.03004555, 0.02887537])*5
    quality_intercept = 0.08707462696457707*5
    quality = merged_df['aesthetic_score'] * quality_weights[0]/ 100 + merged_df['technical_score'] * quality_weights[1]/ 100 + quality_intercept
    merged_df['quality_score'] = quality * 100

    temporal_weights = np.array([2.92492244, 0.45475678, 0.17561504])*5
    temporal_intercept = -3.42274050899774*5
    temporal_metrics = merged_df['clip_temp_score'] * temporal_weights[0] + (1 - merged_df['warping_error']) * temporal_weights[1] + merged_df['face_consistency_score'] * temporal_weights[2] + temporal_intercept
    merged_df['temporal'] = temporal_metrics * 100

    motion_weights = np.array([-0.01641512, -0.01340959, -0.10517075])*5
    motion_intercept = 0.1297562020899355*5
    motion_metrics = merged_df['action_score'] * motion_weights[0] + merged_df['flow_score'] / 100 * motion_weights[2] + motion_intercept
    merged_df['motion'] = motion_metrics * 100

    t2v_align_weights = np.array([-0.0701577, 0.02561424, 0.05566109, 0.0173974, -0.020954, 0.03069167, 0.00372351, 0.22686202]) * 5
    t2v_align_intercept = -0.30683181901390977

    t2v_align_metrics = merged_df['clip_score'] * t2v_align_weights[0] + merged_df['blip_bleu'] * t2v_align_weights[1] + merged_df['sd_score'] * t2v_align_weights[2] + t2v_align_intercept
    merged_df['t2v_alighn'] = t2v_align_metrics * 100

    return merged_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge TSV files in a folder")
    parser.add_argument("--input_folder", type=str, help="Path to the folder containing TSV files")
    parser.add_argument("--output_file", type=str, help="Path to the output merged TSV file")
    args = parser.parse_args()

    folder_path = args.input_folder
    output_path = args.output_file
    tsv_files = get_tsv_files(folder_path)

    if not tsv_files:
        print(f"No TSV files found in the folder: {folder_path}")
        sys.exit(1)

    merged_df = merge_tsv_files(tsv_files)
    post_process(merged_df)
    merged_df.to_csv(output_path, sep='\t', index=False)
    print(f"Merged TSV file saved to {output_path}")