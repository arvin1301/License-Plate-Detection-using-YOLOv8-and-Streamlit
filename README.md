# License-Plate-Detection-using-YOLOv8-and-Streamlit

This project uses YOLOv8 and Streamlit to detect license plates in images, videos, or in real-time via webcam. It also provides a simple way to train YOLOv8 on a custom license plate dataset using Roboflow.

## Features
- Detect license plates in uploaded images and videos.
- Real-time detection with your device's camera.
- Train the YOLOv8 model on a custom dataset.


## Setup

### 1. Clone the Repository
git clone https://github.com/Arijit1080/Licence-Plate-Detection-using-YOLO-V8.git
cd Licence-Plate-Detection-using-YOLO-V8


2. Install Requirements
pip install -r requirements.txt
pip install roboflow


3. Download Dataset
Use Roboflow to download a labeled dataset. Replace the API key with your own if necessary:

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("mochoye").project("license-plate-detector-ogxxg")
dataset = project.version(2).download("yolov8")


4. Train the YOLOv8 Model
Run this command to train the YOLOv8 model:
python /content/Licence-Plate-Detection-using-YOLO-V8/ultralytics/yolo/v8/detect/train.py model=yolov8n.pt data=/content/Licence-Plate-Detection-using-YOLO-V8/License-Plate-Detector-2/data.yaml epochs=100


Running the App
Set the Model Path: Make sure the best.pt model file is in the correct path in the code:
model_path = 'E:/Data science project/License_plate_detection_yolov8/best.pt'


Launch Streamlit:
streamlit run app.py


App Options:
Upload Image: Choose an image file to detect license plates.
Upload Video: Select a video file for frame-by-frame detection.
Start Camera: Activate webcam for real-time detection.
Visualize Training Results


To see training outcomes, use the following code to view images:
from PIL import Image
import matplotlib.pyplot as plt


# Display training results
plt.imshow(Image.open('/path/to/results.png'))
plt.axis('off')
plt.show()


Project Structure

license-plate-detection/
├── app.py                     # Main application file
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
└── ultralytics/
    └── yolo/
        └── v8/
            └── detect/        # YOLOv8 training and detection code



Acknowledgments
Ultralytics for YOLOv8.
Roboflow for dataset management.
Streamlit for easy web app deployment.


Screenshot
![05 11 2024_18 39 08_REC](https://github.com/user-attachments/assets/b2a5cfa1-5cb5-474a-9b43-5db89c895948)
![05 11 2024_18 41 39_REC](https://github.com/user-attachments/assets/78de0756-fd0a-4f13-aa99-7a94ab232bb9)

