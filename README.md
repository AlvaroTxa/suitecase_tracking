# Suitcase Tracking and Monitoring System

This project implements a system to track and monitor people and suitcases using YOLOv5 for object detection and SORT (Simple Online and Realtime Tracking) for object tracking.
<br>The system aims to detect and associate suitcases with people, and it monitors when a person drops a suitcase, capturing images of the individual for further analysis.</br>

## Features

- **Object Detection**: Utilizes YOLOv5 to detect objects in video frames, specifically looking for "person" and "suitcase".
- **Object Tracking**: Implements the SORT (Simple Online and Realtime Tracking) algorithm to keep track of detected objects across video frames.
- **Person-Suitcase Association**: Associates suitcases with the closest person if within a 70-pixel range for more than 5 seconds. (Adjust pixels according to distance from the camera to the place. More distance, less pixel range)
- **Disassociation Detection**: Detects when a person drops a suitcase by moving more than 70 pixels away from it for more than 5 seconds.
- **Capture Images**: Captures three images of the bounding box of the person who dropped the suitcase and saves them in a folder named with the person's and suitcase's IDs.
- **Visual Feedback**: Displays bounding boxes around detected objects with different colors (green for normal, red for disassociated) in the video feed.

## How It Works

### 1. Initialization:
- Load the YOLOv5 model for object detection.
- Initialize the SORT tracker.
### 2. Object Detection:
- Detect objects in each video frame using YOLOv5.
- Filter detections to only include persons and suitcases with confidence scores above a threshold.
### 3. Object Tracking:
- Update the SORT tracker with the filtered detections to maintain consistent object identities.
### 4. Association and Disassociation:
- Check and associate persons with nearby suitcases.
- Detect when a person drops a suitcase and capture images of the person.
### 5. Visualization:
- Draw bounding boxes around detected objects.
- Change the color of the bounding box to red when a person drops a suitcase.
- Display the frame with the drawn bounding boxes.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/AlvaroTxa/suitcase-tracking.git
    cd suitcase-tracking
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Clone YOLOv5 repository:
    ```sh
    git clone https://github.com/ultralytics/yolov5.git
    ```

4. Clone SORT repository:
    ```sh
    git clone https://github.com/abewley/sort.git
    ```

## Usage

1. Run the main script:
    ```sh
    python main.py
    ```

2. The system will start capturing video from the default camera (camera index 0). It will display the video feed with bounding boxes drawn around detected objects.

3. When a person is detected carrying a suitcase, the system will associate them. If the person drops the suitcase, the bounding box color will change to red, and three images of the person will be captured and saved in the `Outputs` folder.

## Project Structure

- `main.py`: The main script to run the tracking system.
- `utilities.py`: Contains the `ObjectTracker` class that implements the detection, tracking, association, and disassociation logic.
- `Outputs/`: Directory where captured images of disassociated persons are saved.

## Example Output

When a person is detected dropping a suitcase, the system will output messages like:</br>
<br>[INFO] Person 2 has taken suitcase 8
<br>[INFO] Person 2 has dropped suitcase 8</br>
<br>The bounding box around the person will turn red, and images will be saved in the `Outputs` folder.

## Acknowledgements

- [YOLOv5](https://github.com/ultralytics/yolov5) for object detection.
- [SORT](https://github.com/abewley/sort) for object tracking.