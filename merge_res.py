import os
import sys
import pandas as pd
import argparse

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
    merged_df.to_csv(output_path, sep='\t', index=False)
    print(f"Merged TSV file saved to {output_path}")