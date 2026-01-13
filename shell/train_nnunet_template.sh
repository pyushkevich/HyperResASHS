#!/bin/bash
#$ -S /bin/bash
set -e  -x

function main()
{
    # NOTE: The following placeholders will be automatically replaced by preprocessing.py
    # {EXP_NUM} will be replaced with EXP_NUM from config
    # {TRAINER} will be replaced with TRAINER from config
    for ((i=0;i<5;i++)); do
        nnUNetv2_train {EXP_NUM} 3d_fullres $i -tr {TRAINER}
    done
}

if [[ $1 ]]; then
    command=$1
    echo $1
    shift
    $command $@
else
    main
fi

