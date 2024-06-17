import cv2
import numpy as np
import threading
import logging
from ultralytics import YOLO
import subprocess
from yaspin import yaspin
from yaspin.spinners import Spinners

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Variables for YOLO detection
model = YOLO("yolov8n.pt")
labels = open("coco.names").read().strip().split("\n")
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
CONFIDENCE = 0.5
font_scale = 1
thickness = 1

# RTSP stream URLs
rtsp_stream = "rtsp://192.168.68.67:8554/cam"
output_rtsp = "rtsp://192.168.68.77:8554/mystream"

# Set up video capture
cap = cv2.VideoCapture(rtsp_stream)
if not cap.isOpened():
    logging.error("Failed to open the RTSP stream")
    exit(1)

# Function to process YOLO object detection
def process_yolo(frame):
    results = model.predict(frame, conf=CONFIDENCE, verbose=False)[0]
    for data in results.boxes.data.tolist():
        xmin, ymin, xmax, ymax, confidence, class_id = data
        xmin, ymin, xmax, ymax, class_id = map(int, [xmin, ymin, xmax, ymax, class_id])

        color = [int(c) for c in colors[class_id]]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)
        text = f"{labels[class_id]}: {confidence:.2f}"
        cv2.putText(frame, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
    return frame

# Function to read frames from the RTSP stream and process them
def process_stream():
    logging.info("Starting to process the stream")
    with yaspin(Spinners.bouncingBall, text="Processing frames...") as spinner:
        while True:
            ret, frame = cap.read()
            if not ret:
                spinner.fail("Failed to grab frame")
                logging.error("Failed to grab frame")
                break
            processed_frame = process_yolo(frame)
            yield processed_frame
        spinner.ok("Processing frames done.")

# Function to relay the processed stream using FFmpeg
def relay_stream():
    logging.info("Starting FFmpeg relay process")
    try:
        process = subprocess.Popen(
            [
                'ffmpeg',
                '-re',
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', '640x480',  # Change this to your stream's resolution
                '-r', '15',  # Change the frame rate to a lower value
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-pix_fmt', 'yuv420p',
                '-b:v', '1M',  # Lower the bitrate
                '-f', 'rtsp',
                '-rtsp_transport', 'tcp',
                output_rtsp
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except Exception as e:
        logging.error(f"Failed to start FFmpeg process: {e}")
        return

    with yaspin(Spinners.dots, text="Relaying processed frames...") as spinner:
        try:
            for frame in process_stream():
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            process.wait()
            spinner.ok("Relaying done.")
        except BrokenPipeError as e:
            spinner.fail("Broken pipe error occurred")
            logging.error(f"Broken pipe error: {e}")
        except Exception as e:
            spinner.fail("An error occurred while relaying frames")
            logging.error(f"Error: {e}")
        finally:
            stdout, stderr = process.communicate()
            logging.error(f"FFmpeg stdout: {stdout}")
            logging.error(f"FFmpeg stderr: {stderr}")

    logging.info("FFmpeg relay process completed")

# Start the processing and relaying in a separate thread
relay_thread = threading.Thread(target=relay_stream)
logging.info("Starting relay thread")
relay_thread.start()

# Keep the main thread running
try:
    while True:
        pass
except KeyboardInterrupt:
    logging.info("Shutting down gracefully")
    cap.release()
    relay_thread.join()
    logging.info("Shutdown complete")
