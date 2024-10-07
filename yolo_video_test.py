import cv2 
from ultralytics import YOLO
from flask import Flask, render_template, Response
from threading import Thread

app = Flask(__name__)

# Load the pre-trained YOLO model
model = YOLO(r"C:\Users\lamaa\Downloads\LPmodel\best (1).pt")

# Traffic detection status
traffic_status = "No Traffic"

def detect_traffic():
    global traffic_status
    video_path = r"C:\Users\lamaa\Downloads\theFainl.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read.")
            break

        # Perform inference on the frame
        results = model(frame)

        # Check if results are not empty
        if results:
            detections = results[0].boxes
            num_detections = len(detections)
            traffic_status = "Traffic" if num_detections > 17 else "No Traffic"
            print(f"Number of detections: {num_detections}")
            print(f"Traffic Status: {traffic_status}")

        # Optional: Add a small delay for processing
        cv2.waitKey(1)

    cap.release()
    print("Video capture released.")

def generate_video_feed():
    video_path = r"C:\Users\lamaa\Downloads\theFainl.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference on the frame
        results = model(frame)
        # You can add code here to draw boxes around detected objects if needed

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame to the browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    global traffic_status
    return render_template('index.html', traffic_status=traffic_status)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    thread = Thread(target=detect_traffic)
    thread.start()
    app.run(debug=True)
