# Object Detection UI

This project is a Flask-based web application designed for object detection, specifically aimed at infrastructure components like insulators, crossarms, and poles. It utilizes YOLO and UNet models for detection and instance segmentation.

## Features

- **Object Detection & Classification**: Identifies insulators, crossarms, and poles using YOLO models.
- **Conductor Segmentation**: Uses a custom UNet model (ResNet34 backbone) to segment conductors and measure their thickness.
- **Rule Engine**: Includes a highly specific pipeline that estimates voltage, tracks pole orientation (lean angle), and identifies crossarm types, calculating bounding box orientations.
- **User Authentication**: Simple login system distinguishing between 'admin' and 'user' roles.
- **Admin Dashboard**: Admins can review tasks submitted by workers and export detailed analysis reports to CSV.
- **Mobile Access**: Uses ngrok to expose the local server for remote or mobile access.

## Prerequisites

- Python 3.8+ (Python 3.9 recommended)
- Optional but recommended: A CUDA-enabled GPU for faster model inference.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sreeya-ux/object_detection_ui.git
cd object_detection_ui
```

### 2. Set up a Virtual Environment

It is highly recommended to run this project inside a virtual environment to manage dependencies locally.

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

You will need the following key libraries installed. You can install them manually using pip:

```bash
pip install flask werkzeug requests ultralytics opencv-python numpy torch torchvision segmentation-models-pytorch scikit-image
```

*(Note: PyTorch installation might vary depending on whether you have a CUDA GPU. Visit [PyTorch's website](https://pytorch.org/get-started/locally/) for the exact command for your system).*

### 4. Initialize the Database

The project uses SQLite for user and task management. If `database.db` is missing or you want to start fresh, run the initialization script:

```bash
python init_db.py
```

This will create `database.db` and insert two default users:
- **Admin**: Username: `admin`, Password: `admin123`
- **User/Worker**: Username: `Worker-Alpha`, Password: `worker`

### 5. Running the Application

Ensure your virtual environment is activated, then run the Flask app:

```bash
python app.py
```

The application will start, usually on `http://127.0.0.1:5000/`.

### Optional: Setting up Ngrok for Mobile Access

To allow access from other devices (like a mobile phone for testing the camera), install and configure `ngrok`.

1. Download Ngrok from [ngrok.com](https://ngrok.com/).
2. Run it in a separate terminal window:
   ```bash
   ngrok http 5000
   ```
3. The Flask app automatically attempts to find the active ngrok tunnel and will display it on the login page.

## Project Structure Highlights

- `app.py`: Main Flask application handling routing, API endpoints, and orchestrating image processing.
- `pipeline.py`: The master rule-engine pipeline handling the complex logic for component detection and classification.
- `init_db.py`: Script to initialize the SQLite database.
- `models/` & `dry_backup/`: Directories storing the YOLO `.pt` files.
- `best_cable_unet.pth`: The UNet model used for conductor segmentation.
- `templates/`: HTML files for the frontend UI.
- `static/`: CSS and JS assets.

## Usage Guide

1. **Login**: Go to the web address. Use either the Admin or Worker credentials.
2. **Worker View**: Upload an image. The system will process it, run the detections via the Rule Engine and UNet, and return annotated images and confidence scores. Click "Confirm" to save the task.
3. **Admin View**: Log in as `admin`. Go to the Admin Dashboard to view all submitted tasks, review specific task details (labels, confidences, bounding boxes), and export all task data to a CSV.
