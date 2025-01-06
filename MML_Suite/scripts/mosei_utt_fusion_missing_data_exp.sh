#!/bin/bash

set - e

missing_25="/home/jmg/code/phd/MML_Suite/MML_Suite/configs/mosei/centralised/experiments_chapter_three/with_missing_data/utt_fusion_baseline_train_25.yaml"
missing_50="/home/jmg/code/phd/MML_Suite/MML_Suite/configs/mosei/centralised/experiments_chapter_three/with_missing_data/utt_fusion_baseline_train_50.yaml"
missing_75="/home/jmg/code/phd/MML_Suite/MML_Suite/configs/mosei/centralised/experiments_chapter_three/with_missing_data/utt_fusion_baseline_train_75.yaml"
missing_90="/home/jmg/code/phd/MML_Suite/MML_Suite/configs/mosei/centralised/experiments_chapter_three/with_missing_data/utt_fusion_baseline_train_90.yaml"

bash run_n.sh 3 $missing_25

if [ $? -ne 0 ]; then
    echo "Error in running the experiment"
    exit 1
fi

bash run_n.sh 3 $missing_50

if [ $? -ne 0 ]; then
    echo "Error in running the experiment"
    exit 1
fi

bash run_n.sh 3 $missing_75

if [ $? -ne 0 ]; then
    echo "Error in running the experiment"
    exit 1
fi

bash run_n.sh 3 $missing_90

if [ $? -ne 0 ]; then
    echo "Error in running the experiment"
    exit 1
fi
