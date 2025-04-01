EC_path=$1 
dir_videos=$2
prompt_dir=$3
eval_result_dir=$4
dimensions=$5  # New parameter for dimensions, split by ';'

echo "Starting evaluation script..."
echo "EC_path: $EC_path"
echo "dir_videos: $dir_videos"
echo "prompt_dir: $prompt_dir"
echo "eval_result_dir: $eval_result_dir"
echo "dimensions: $dimensions"

# Convert dimensions into an array
IFS=';' read -r -a dimension_array <<< "$dimensions"

# Function to log and execute a command
run_command() {
    echo "Running: $*"
    eval "$@"
    if [ $? -ne 0 ]; then
        echo "Error: Command failed - $*"
        exit 1
    fi
}

# # VQA_A and VQA_T
if [[ " ${dimension_array[@]} " =~ " VQA " ]]; then
    echo "Running VQA_A and VQA_T..."
    cd $EC_path
    cd ./metrics/DOVER
    run_command python3 evaluate_dover.py --dir_videos $dir_videos --output_path $eval_result_dir
fi

# CLIP-Score
if [[ " ${dimension_array[@]} " =~ " CLIP-Score " ]]; then
    echo "Running CLIP-Score..."
    cd $EC_path
    cd ./metrics/Scores_with_CLIP
    run_command python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'clip_score' --output_path $eval_result_dir --prompt_file $prompt_dir
fi

# Face Consistency
if [[ " ${dimension_array[@]} " =~ " Face-Consistency " ]]; then
    echo "Running Face Consistency..."
    cd $EC_path
    cd ./metrics/Scores_with_CLIP
    run_command python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'face_consistency_score' --output_path $eval_result_dir --prompt_file $prompt_dir
fi

# SD-Score
if [[ " ${dimension_array[@]} " =~ " SD-Score " ]]; then
    echo "Running SD-Score..."
    cd $EC_path
    cd ./metrics/Scores_with_CLIP
    run_command python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'sd_score' --output_path $eval_result_dir --prompt_file $prompt_dir
fi

# BLIP-BLUE
if [[ " ${dimension_array[@]} " =~ " BLIP-BLUE " ]]; then
    echo "Running BLIP-BLUE..."
    cd $EC_path
    cd ./metrics/Scores_with_CLIP
    run_command python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'blip_bleu' --output_path $eval_result_dir --prompt_file $prompt_dir
fi

# CLIP-Temp
if [[ " ${dimension_array[@]} " =~ " CLIP-Temp " ]]; then
    echo "Running CLIP-Temp..."
    cd $EC_path
    cd ./metrics/Scores_with_CLIP
    run_command python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'clip_temp_score' --output_path $eval_result_dir --prompt_file $prompt_dir
fi

# Action Score
if [[ " ${dimension_array[@]} " =~ " Action-Score " ]]; then
    echo "Running Action Score..."
    cd $EC_path
    cd ./metrics/mmaction2/demo
    run_command python3 action_score_fix.py --dir_videos $dir_videos --metric 'action_score' --output_path $eval_result_dir
fi

# Flow-Score
if [[ " ${dimension_array[@]} " =~ " Flow-Score " ]]; then
    echo "Running Flow-Score..."
    cd $EC_path
    cd ./metrics/RAFT
    run_command python3 optical_flow_scores_fix.py --dir_videos $dir_videos --metric 'flow_score' --output_path $eval_result_dir
fi

# Warping Error
if [[ " ${dimension_array[@]} " =~ " Warping-Error " ]]; then
    echo "Running Warping Error..."
    cd $EC_path
    cd ./metrics/RAFT
    run_command python3 optical_flow_scores_fix.py --dir_videos $dir_videos --metric 'warping_error' --output_path $eval_result_dir
fi

echo "Evaluation script completed successfully."