import cv2
from utilities import ObjectTracker

def main():
    # Inicializar el rastreador de objetos
    tracker = ObjectTracker()

    # Leer la imagen desde la cámara
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen al formato adecuado
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar objetos en la imagen
        results = tracker.detect_objects(img)
        detections = tracker.get_detections(results)

        # Actualizar el rastreador
        tracks = tracker.update_tracks(detections)

        # Mostrar los resultados
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            cls = tracker.get_class(track_id)
            label = f'ID {track_id}: {tracker.model.names[cls]}' if cls != -1 else f'ID {track_id}: Unknown'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('YOLOv5 Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()