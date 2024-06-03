from utilities import ObjectTracker
import cv2

def main():
    # Initialize the object tracker
    tracker = ObjectTracker()

    # Open the video capture
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to the correct format
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect objects in the frame (YOLO)
        results = tracker.detect_objects(img)

        # Get detections and update tracks (SORT model)
        detections = tracker.get_detections(results)
        tracks = tracker.update_tracks(detections)

        # Check for person-suitcase association and disassociation
        tracker.check_person_suitcase_association()
        tracker.check_suitcase_disassociation(cap)

        # Display the results
        if tracker.show_bbox_detections(frame, tracks):
            break

    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()