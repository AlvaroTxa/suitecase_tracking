from collections import defaultdict
from sort.sort import Sort
import numpy as np
import torch
import cv2
import os

class ObjectTracker:
    def __init__(self, model_name='yolov5m'):
        """
        Initializes the object tracker with the YOLOv5 model and SORT tracker.

        Params:
            model_name: Name of the YOLOv5 model to load.
        """
        self.model = torch.hub.load('ultralytics/yolov5', model_name)
        self.tracker = Sort()
        self.track_to_class = {}  # Dictionary to map track_id to class
        self.track_to_bbox = {}  # Dictionary to map track_id to bounding box
        self.person_suitcase_pairs = defaultdict(int)  # Dictionary to count frames of proximity
        self.assigned_suitcases = set()  # Set to keep track of assigned suitcases
        self.tracked_associations = {}  # Dictionary to maintain person-suitcase associations
        self.informed_disassociations = set()  # Set to ensure disassociation is informed only once
        
        # Get the class IDs for 'person' and 'suitcase'
        self.person_class_id = None
        self.suitcase_class_id = None
        for cls_id, cls_name in self.model.names.items():
            if cls_name == 'person':
                self.person_class_id = cls_id
            elif cls_name == 'suitcase':
                self.suitcase_class_id = cls_id

    def detect_objects(self, image):
        """
        Detects objects in an image using the YOLOv5 model.

        Params:
            image: Image in which to detect objects.
        Returns object detection results.
        """
        results = self.model(image)
        return results

    def get_detections(self, results):
        """"
        Filters detection results to get only 'person' and 'suitcase' classes with confidence > 0.5.

        Params:
            results: Object detection results.
        Returns array with bounding boxes, confidence, and class.
        """
        detections = []
        for *box, conf, cls in results.xyxy[0]:
            # Filter by class and confidence
            if conf > 0.5 and int(cls) in [self.person_class_id, self.suitcase_class_id]:
                x1, y1, x2, y2 = map(int, box)
                detections.append([x1, y1, x2, y2, conf, int(cls)])
        return np.array(detections)

    def update_tracks(self, detections):
        """
        Updates object tracks using SORT and maps track_id to the corresponding class.

        Params:
            detections: Current detections.
        Returns updated tracks.
        """
        if len(detections) == 0:
            detections = np.empty((0, 5))  # Ensure detections have the correct shape
        tracks = self.tracker.update(detections[:, :5])  # Only pass bbox and conf to SORT
        
        new_track_to_class = {}
        new_track_to_bbox = {}
        for track in tracks:
            track_id = int(track[4])
            bbox = track[:4]
            # Find the class corresponding to this track_id (in the "tracker.update" they may have changed)
            for det in detections:
                # Correspondencia entre objeto detectado y rastreado
                iou = self.compute_iou(bbox, det[:4])
                # Use IoU threshold to determine correspondence
                if iou > 0.5:
                    new_track_to_class[track_id] = int(det[5])
                    new_track_to_bbox[track_id] = bbox
                    break
        self.track_to_class.update(new_track_to_class)
        self.track_to_bbox.update(new_track_to_bbox) 
        return tracks

    def get_class(self, track_id):
        """
        Gets the class corresponding to a track_id.

        Params:
            track_id: Track ID.
        Returns object class.
        """
        return self.track_to_class.get(track_id, -1)

    def compute_iou(self, box1, box2):
        """
        Computes the Intersection over Union (IoU) between two bounding boxes.
        
        Params:
            box1: First bounding box.
            box2: Second bounding box.
        Returns IoU between the two bounding boxes.
        """
        x1, y1, x2, y2 = box1
        x1_b, y1_b, x2_b, y2_b = box2

        # Determine the coordinates of the intersection rectangle
        inter_x1 = max(x1, x1_b)
        inter_y1 = max(y1, y1_b)
        inter_x2 = min(x2, x2_b)
        inter_y2 = min(y2, y2_b)

        # Compute the area of intersection rectangle
        inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

        # Compute the area of both bounding boxes
        box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        box2_area = (x2_b - x1_b + 1) * (y2_b - y1_b + 1)

        # Compute the intersection over union (IoU)
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    def check_person_suitcase_association(self):
        """
        Associates suitcases with persons if they are within 70 pixels for more than 5 seconds.
        """
        # Do not run if there are no suitcases
        if not any(cls == self.suitcase_class_id for cls in self.track_to_class.values()):
            return

        for suitcase_id, suitcase_bbox in self.track_to_bbox.items():
            if self.track_to_class.get(suitcase_id) != self.suitcase_class_id:
                continue
            # Do not run if the suitcase has already been associated
            if suitcase_id in self.assigned_suitcases:
                continue
            suitcase_center_x = self.get_center_x(suitcase_bbox)
            closest_person_id = None
            closest_distance = float('inf')

            for person_id, person_bbox in self.track_to_bbox.items():
                if self.track_to_class.get(person_id) != self.person_class_id:
                    continue

                person_center_x = self.get_center_x(person_bbox)
                distance_x = abs(person_center_x - suitcase_center_x)

                # Distance threshold
                if distance_x < 70 and distance_x < closest_distance:
                    closest_distance = distance_x
                    closest_person_id = person_id

            if closest_person_id is not None:
                self.person_suitcase_pairs[(closest_person_id, suitcase_id)] += 1

                if self.person_suitcase_pairs[(closest_person_id, suitcase_id)] > 5 * 10:  # Umbral de tiempo (5 segundos a 30 FPS)
                    print(f"\n[INFO] Person {closest_person_id} has taken suitcase {suitcase_id}")
                    self.assigned_suitcases.add(suitcase_id)  # Marcar la maleta como asignada
                    self.tracked_associations[suitcase_id] = closest_person_id  # Asociar la maleta a la persona

    def check_suitcase_disassociation(self, cap):
        """
        Detects when a person drops a suitcase and captures images of the person's bounding box.
        
        Params:
            cap: Video capture object.
        """
        for suitcase_id, person_id in self.tracked_associations.items():
            # Only process bags that have already been assigned
            if suitcase_id not in self.assigned_suitcases:
                continue

            suitcase_bbox = self.track_to_bbox.get(suitcase_id)
            person_bbox = self.track_to_bbox.get(person_id)

            # Ensure that both bboxes exist
            if suitcase_bbox is None or person_bbox is None:
                continue
            suitcase_center_x = self.get_center_x(suitcase_bbox)
            person_center_x = self.get_center_x(person_bbox)
            distance_x = abs(person_center_x - suitcase_center_x)

            if distance_x > 70:  # Umbral de distancia para desasociación
                self.person_suitcase_pairs[(person_id, suitcase_id)] -= 1

                if self.person_suitcase_pairs[(person_id, suitcase_id)] < 0:
                    self.person_suitcase_pairs[(person_id, suitcase_id)] = 0

                if self.person_suitcase_pairs[(person_id, suitcase_id)] == 0:
                    if (person_id, suitcase_id) not in self.informed_disassociations:
                        print(f"\n[INFO] Person {person_id} has dropped suitcase {suitcase_id}")
                        self.capture_images(person_id, suitcase_id, cap)
                        self.informed_disassociations.add((person_id, suitcase_id))

    def is_disassociated(self, person_id):
        """
        Checks if a person has already been informed about dropping a suitcase.
        
        Params:
            person_id: ID of the person.
        Returns True if the person has already been informed, False otherwise.
        """
        return any(pair[0] == person_id for pair in self.informed_disassociations)

    def get_center_x(self, bbox):
        """
        Calculates the central x-coordinate of a bounding box.
        
        Params:
            bbox: Bounding box.
        Returns central x-coordinate.
        """
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) // 2
        return center_x

    def capture_images(self, person_id, suitcase_id, cap):
        """
        Captures three images of the bounding box of the person who dropped the suitcase and saves them in a folder.
        
        Params:
            person_id: ID of the person.
            suitcase_id: ID of the suitcase.
            cap: Video capture object.
        """
        folder_name = os.path.join('Outputs', f"{person_id}_{suitcase_id}")
        os.makedirs(folder_name, exist_ok=True)

        for i in range(3):
            person_bbox = self.track_to_bbox.get(person_id)
            if person_bbox is None:
                continue
            x1, y1, x2, y2 = map(int, person_bbox)
            ret, frame = cap.read()
            if not ret:
                break

            # Bounding box is cropped
            img = frame[y1:y2, x1:x2]

            # Check if the image is empty
            if img.size == 0:
                continue
            img_path = os.path.join(folder_name, f"{i+1}.jpg")
            cv2.imwrite(img_path, img)
            cv2.waitKey(1000)  # Wait 1 second before the next capture

    def show_bbox_detections(self, frame, tracks):
        """
        Draws bounding boxes around detected objects and displays the frame.
        
        Params:
            frame: The current video frame.
            tracks: The list of tracked objects.
        Returns True if 'q' is pressed to quit, False otherwise.
        """
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            cls = self.get_class(track_id)
            label = f'ID {track_id}: {self.model.names[cls]}' if cls != -1 else f'ID {track_id}: Unknown'

            # Determine the color of the bounding box
            color = (0, 255, 0)  # Default color (green) for bounding box
            if (cls == self.person_class_id) and self.is_disassociated(track_id):
                color = (0, 0, 255)  # Change to red if the suitcase has been dropped

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the resulting frame
        cv2.imshow('YOLOv5 Detection', frame)
        return True if cv2.waitKey(1) & 0xFF == ord('q') else False