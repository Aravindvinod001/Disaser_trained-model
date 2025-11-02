# Drone Disaster Detection Module

This module allows a drone to detect disasters (like fire, flood, or smoke) in real-time using a trained YOLOv8 TFLite model.

## Folder Structure

Disaster_Detection/
- Models/
   best_float16.tflite        # Trained TFLite model
- Docs/  
   README.md                  # This file
- requirements.txt               # Python dependencies

Optional:
sample_images/                 # For testing with static images
  test1.jpg

## Dependencies

Install the following Python libraries on Raspberry Pi:

```bash
sudo apt update
sudo apt install python3-opencv -y
pip install numpy opencv-python
