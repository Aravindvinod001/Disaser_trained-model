# Drone Disaster Detection Module

This module allows a drone to detect disasters (like fire, flood, or smoke) in real-time using a trained YOLOv8 TFLite model.

## Folder Structure

Disaster_Detection/
│
├── Models/
│   └── best_float16.tflite        # Trained TFLite model
│
├── Code/
│   └── run_disaster_camera.py     # Script to run the model on live camera feed
│
├── Docs/
│   └── README.md                  # This file
│
└── requirements.txt               # Python dependencies

Optional:
├── sample_images/                 # For testing with static images
    └── test1.jpg

## Dependencies

Install the following Python libraries on Raspberry Pi:

```bash
sudo apt update
sudo apt install python3-opencv -y
pip install numpy opencv-python
Install TFLite Runtime
The TFLite runtime is required to run the .tflite model:

pip install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.14.0-cp311-cp311-linux_aarch64.whl
Make sure the .whl file matches your Python version and Raspberry Pi architecture.

How to Run
1.Connect the Pi camera or USB camera to the Raspberry Pi.
2.Open a terminal and navigate to the Code/ folder.
3.Run the script:

python3 run_disaster_camera.py
A window will open showing the live camera feed with disaster detection results.

Press q to exit the program.

Notes
The TFLite model used here is optimized for Raspberry Pi (float16 version).

You do not need the original PyTorch .pt files for deployment.

For testing, you can place static images in sample_images/ and modify the script to read them instead of the live feed.
