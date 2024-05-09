import cv2
from ultralytics import YOLO
import time

class SingleClassDetection:
    def __init__(self, model_path, device='cpu') -> None:
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def run(self, source, save=False, imgsz=640, conf=0.25):
        results = self.model.predict(source=source, save=save, imgsz=imgsz, conf=conf, device=self.device)
        detected_list = []
        for result in results:
            for box in result.boxes:
                name = f"{result.names[int(box.cls[0])]}"
                (x1, y1), (x2, y2) = (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
                detected_list.append({'box': (x1, y1, x2, y2), 'name': str(name)})
        return detected_list

# Initialize each model for a specific class
models = {
    'aprons': SingleClassDetection('aprons_best.pt'),
    'glasses': SingleClassDetection('glasses_best.pt'),
    'gloves': SingleClassDetection('gloves_best.pt'),
    'mask': SingleClassDetection('mask_best.pt'),
    'shoes': SingleClassDetection('shoes_best.pt')
}

# Open the video file
video_path = '12_26_23(2).mp4'  # Replace with your video file path
video_capture = cv2.VideoCapture(video_path)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_video.mp4', fourcc, 25.0, (int(video_capture.get(3)), int(video_capture.get(4))))

skip_frames = 10  # Number of frames to skip
frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        break  # Break the loop if no frame is captured

    frame_count += 1
    if frame_count % skip_frames == 0:  # Process every nth frame
        for class_name, model in models.items():
            detected_items = model.run(frame)
            for item in detected_items:
                x1, y1, x2, y2 = item['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        output_video.write(frame)  # Write frame to output video
        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit if 'q' is pressed
            break

# Release the video capture, writer, and close all OpenCV windows
video_capture.release()
output_video.release()
cv2.destroyAllWindows()
