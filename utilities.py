import torch
import numpy as np
from sort.sort import Sort

class ObjectTracker:
    def __init__(self, model_name='yolov5m'):
        self.model = torch.hub.load('ultralytics/yolov5', model_name)
        self.tracker = Sort()
        self.track_to_class = {}  # Diccionario para mapear track_id a la clase
        
        # Obtener los IDs de las clases 'person' y 'suitcase'
        self.person_class_id = None
        self.suitcase_class_id = None
        for cls_id, cls_name in self.model.names.items():
            if cls_name == 'person':
                self.person_class_id = cls_id
            elif cls_name == 'suitcase':
                self.suitcase_class_id = cls_id

    def detect_objects(self, image):
        results = self.model(image)
        return results

    def get_detections(self, results):
        detections = []
        for *box, conf, cls in results.xyxy[0]:
            if conf > 0.5 and int(cls) in [self.person_class_id, self.suitcase_class_id]:  # Filtrar por clase y confianza
                x1, y1, x2, y2 = map(int, box)
                detections.append([x1, y1, x2, y2, conf, int(cls)])
        return np.array(detections)

    def update_tracks(self, detections):
        if len(detections) == 0:
            detections = np.empty((0, 5))  # Asegurar que detections tenga la forma correcta
        tracks = self.tracker.update(detections[:, :5])  # Solo pasar bbox y conf a SORT
        
        new_track_to_class = {}
        for track in tracks:
            track_id = int(track[4])
            # Buscar la clase correspondiente a este track_id
            for det in detections:
                iou = self.compute_iou(track[:4], det[:4])
                if iou > 0.5:  # Usar umbral de IoU para determinar correspondencia
                    new_track_to_class[track_id] = int(det[5])
                    break
        self.track_to_class.update(new_track_to_class)
        return tracks

    def get_class(self, track_id):
        return self.track_to_class.get(track_id, -1)

    def compute_iou(self, box1, box2):
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