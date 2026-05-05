import csv
import os
import time
import threading
import math
import cv2

from pymavlink import mavutil
from ultralytics import YOLO



PIXHAWK_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

MODEL_PATH = "/home/quetzal/models/best.pt"  
TARGET_CLASS_NAME = "target-localization"

CAMERA_INDEX = 0
# camera at 90 degrees downwards, pointing down
CAMERA_PITCH_DEG = -90.0  

OUTPUT_DIR = "target_logs"
FRAME_DIR = os.path.join(OUTPUT_DIR, "frames")
CSV_PATH = os.path.join(OUTPUT_DIR, "detections.csv")

CONFIDENCE_THRESHOLD = 0.40


#telemetry part
latest_telemetry = {
    "lat": None,
    "lon": None,
    "altitude_m": None,
    "yaw_deg": None,
    "mode": None,
    "armed": None,
    "gps_fix": None,
    "satellites": None,
}

telemetry_lock = threading.Lock()




# MAVlink connection

def mavlink_reader():
    print("Connecting to Pixhawk...")
    master = mavutil.mavlink_connection(PIXHAWK_PORT, baud=BAUD_RATE)
    master.wait_heartbeat()
    print(f"MAVLink connected: system {master.target_system}, component {master.target_component}")

    while True:
        msg = master.recv_match(blocking=True)
        if msg is None:
            continue

        msg_type = msg.get_type()

        with telemetry_lock:
            if msg_type == "HEARTBEAT":
                latest_telemetry["mode"] = mavutil.mode_string_v10(msg)
                latest_telemetry["armed"] = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )

            elif msg_type == "GPS_RAW_INT":
                latest_telemetry["gps_fix"] = msg.fix_type
                latest_telemetry["satellites"] = msg.satellites_visible

            elif msg_type == "GLOBAL_POSITION_INT":
                latest_telemetry["lat"] = msg.lat / 1e7
                latest_telemetry["lon"] = msg.lon / 1e7
                latest_telemetry["altitude_m"] = msg.relative_alt / 1000.0

            elif msg_type == "ATTITUDE":
                yaw_deg = math.degrees(msg.yaw)
                if yaw_deg < 0:
                    yaw_deg += 360.0
                latest_telemetry["yaw_deg"] = yaw_deg




# log data

def main():
    os.makedirs(FRAME_DIR, exist_ok=True)

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded.")

    print("Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try changing CAMERA_INDEX to 1 or check camera connection.")

    telemetry_thread = threading.Thread(target=mavlink_reader, daemon=True)
    telemetry_thread.start()

    file_exists = os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        if not file_exists:
            writer.writerow([
                "timestamp_unix",
                "timestamp_readable",
                "frame_file",
                "bbox_center_x",
                "bbox_center_y",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "confidence",
                "drone_latitude",
                "drone_longitude",
                "altitude_m",
                "yaw_heading_deg",
                "camera_pitch_deg",
                "mode",
                "armed",
                "gps_fix",
                "satellites"
            ])

        frame_id = 0
        print("Logger running. Press Ctrl+C to stop.")

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Warning: camera frame not received.")
                    time.sleep(0.1)
                    continue

                timestamp_unix = time.time()
                timestamp_readable = time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(timestamp_unix)
                )

                results = model(frame, verbose=False)

                for result in results:
                    class_names = result.names

                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = class_names[class_id]
                        confidence = float(box.conf[0])

                        if class_name != TARGET_CLASS_NAME:
                            continue

                        if confidence < CONFIDENCE_THRESHOLD:
                            continue

                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        bbox_center_x = (x1 + x2) / 2.0
                        bbox_center_y = (y1 + y2) / 2.0

                        frame_filename = f"frame_{frame_id:06d}.jpg"
                        frame_path = os.path.join(FRAME_DIR, frame_filename)
                        relative_frame_path = os.path.join("frames", frame_filename)

                        cv2.imwrite(frame_path, frame)

                        with telemetry_lock:
                            telemetry_snapshot = latest_telemetry.copy()

                        writer.writerow([
                            timestamp_unix,
                            timestamp_readable,
                            relative_frame_path,
                            bbox_center_x,
                            bbox_center_y,
                            x1,
                            y1,
                            x2,
                            y2,
                            confidence,
                            telemetry_snapshot["lat"],
                            telemetry_snapshot["lon"],
                            telemetry_snapshot["altitude_m"],
                            telemetry_snapshot["yaw_deg"],
                            CAMERA_PITCH_DEG,
                            telemetry_snapshot["mode"],
                            telemetry_snapshot["armed"],
                            telemetry_snapshot["gps_fix"],
                            telemetry_snapshot["satellites"]
                        ])

                        csv_file.flush()

                        print(
                            f"Logged target | conf={confidence:.2f} | "
                            f"lat={telemetry_snapshot['lat']} | "
                            f"lon={telemetry_snapshot['lon']} | "
                            f"alt={telemetry_snapshot['altitude_m']} | "
                            f"yaw={telemetry_snapshot['yaw_deg']}"
                        )

                        frame_id += 1

        except KeyboardInterrupt:
            print("\nStopping logger...")

        finally:
            cap.release()
            print(f"Saved CSV to: {CSV_PATH}")
            print(f"Saved frames to: {FRAME_DIR}")


if __name__ == "__main__":
    main()
