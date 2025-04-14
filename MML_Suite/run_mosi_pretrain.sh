#!/bin/bash
echo "Starting MOSI encoder pretraining"

echo "Step 1: Training audio encoder"
python train_monomodal_mosi.py --config configs/mosi/mono/mosi_audio_encoder.yaml --run_id 1

echo "Step 2: Training video encoder"
python train_monomodal_mosi.py --config configs/mosi/mono/mosi_video_encoder.yaml --run_id 1

echo "Step 3: Training text encoder"
python train_monomodal_mosi.py --config configs/mosi/mono/mosi_text_encoder.yaml --run_id 1

echo "Step 4: Training multimodal model with pretrained encoders"
python train_multimodal.py --config configs/mosi/centralised/utt_fusion_pretrained.yaml --run_id 1

echo "Pretraining complete!" 