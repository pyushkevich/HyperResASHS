#!/bin/bash

function main()
{
    # Set these variables according to your configuration
    # NOTE: The following three variables (INR_PATH, EXP_NUM, MODEL_NAME) will be automatically
    # replaced by prepare_inr.py based on your config file. Do not modify the {PLACEHOLDER} format.
    INR_PATH="{INR_PATH}"  # Will be replaced with INR_PATH from config
    EXP_NUM="{EXP_NUM}"    # Will be replaced with EXP_NUM from config
    MODEL_NAME="{MODEL_NAME}"  # Will be replaced with MODEL_NAME from config
    INR_REPO_PATH="{INR_REPO_PATH}"  # Will be replaced with submodule path automatically
    
    exp_name="${EXP_NUM}${MODEL_NAME}"
    base_dir="${INR_PATH}/${exp_name}/training_preparation"
    start=0
    count=60

    # Get subfolder names as a sorted list
    all_folders=($(find "$base_dir" -mindepth 1 -maxdepth 1 -type d | sort))

    # Slice the list
    selected_folders=("${all_folders[@]:$start:$count}")

    for subfolder in "${selected_folders[@]}"; do
        name=$(basename "$subfolder")
        echo "Running command for: $name"
        python "${INR_REPO_PATH}/main.py" --config "${INR_PATH}/${exp_name}/preprocess/${name}/config.yaml" --logging
    done

}


if [[ $1 ]]; then
    command=$1
    command=$1
    echo $1
    shift
    $command $@
else
    main
fi

