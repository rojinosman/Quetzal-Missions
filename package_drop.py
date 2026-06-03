import cv2
import time
import math
import threading
from ultralytics import YOLO
from pymavlink import mavutil


# =========================
# USER SETTINGS
# =========================

MODEL_PATH = "/home/quetzal/models/best.pt"
TARGET_CLASS = "red-target"

CAMERA_INDEX = 0

PIXHAWK_PORT = "/dev/ttyTHS1"
BAUD_RATE = 57600

CONF_THRESHOLD = 0.45
IMGSZ = 320

# Drop safety
MIN_DROP_ALT_M = 2.2      # 6 ft = 1.83 m, using 2.2 m for safety margin
MAX_DROP_ALT_M = 20.0

# Visual alignment
CENTER_TOLERANCE = 0.12   # normalized image error
CENTER_HOLD_TIME = 2.0    # seconds target must stay centered
MAX_ALIGN_SPEED = 0.8     # m/s
KP_X = 0.6                # left/right correction gain
KP_Y = 0.6                # forward/back correction gain

# If movement is reversed, flip these signs
SIDE_SIGN = 1.0
FORWARD_SIGN = 1.0

# Servo release settings
DROP_SERVO_CHANNEL = 9
DROP_PWM_RELEASE = 2000
DROP_PWM_HOLD = 1000
DROP_HOLD_SECONDS = 0.8

# After drop
POST_DROP_MODE = "LOITER"   # options: "LOITER", "RTL", or "MISSION"


# =========================
# SHARED TELEMETRY
# =========================

telemetry = {
    "connected": False,
    "mode": "UNKNOWN",
    "armed": False,
    "lat": None,
    "lon": None,
    "alt_m": None,
    "yaw_deg": None,
}

telemetry_lock = threading.Lock()
master = None


# =========================
# MAVLINK FUNCTIONS
# =========================

def px4_set_mode(mode_name):
    mode_map = {
        "MANUAL": 1,
        "ALTCTL": 2,
        "POSCTL": 3,
        "AUTO": 4,
        "ACRO": 5,
        "OFFBOARD": 6,
        "STABILIZED": 7,
        "RATTITUDE": 8,
    }

    if mode_name == "MISSION":
        main_mode = 4
        sub_mode = 4
    elif mode_name == "LOITER":
        main_mode = 4
        sub_mode = 3
    elif mode_name == "RTL":
        main_mode = 4
        sub_mode = 5
    else:
        main_mode = mode_map[mode_name]
        sub_mode = 0

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        main_mode,
        sub_mode,
        0, 0, 0, 0
    )


def send_body_velocity(vx, vy, vz):
    """
    BODY_NED frame:
    vx = forward m/s
    vy = right m/s
    vz = down m/s
    """
    type_mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
    )

    master.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        type_mask,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )


def release_package():
    print("RELEASING PACKAGE")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        DROP_SERVO_CHANNEL,
        DROP_PWM_RELEASE,
        0, 0, 0, 0, 0
    )

    time.sleep(DROP_HOLD_SECONDS)

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        DROP_SERVO_CHANNEL,
        DROP_PWM_HOLD,
        0, 0, 0, 0, 0
    )


def mavlink_reader():
    global master

    print("Connecting to Pixhawk...")
    master = mavutil.mavlink_connection(PIXHAWK_PORT, baud=BAUD_RATE)
    master.wait_heartbeat()
    print(f"Connected to Pixhawk: system {master.target_system}, component {master.target_component}")

    with telemetry_lock:
        telemetry["connected"] = True

    while True:
        msg = master.recv_match(blocking=True)
        if msg is None:
            continue

        msg_type = msg.get_type()

        with telemetry_lock:
            if msg_type == "HEARTBEAT":
                telemetry["mode"] = mavutil.mode_string_v10(msg)
                telemetry["armed"] = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )

            elif msg_type == "GLOBAL_POSITION_INT":
                telemetry["lat"] = msg.lat / 1e7
                telemetry["lon"] = msg.lon / 1e7
                telemetry["alt_m"] = msg.relative_alt / 1000.0

            elif msg_type == "ATTITUDE":
                yaw = math.degrees(msg.yaw)
                if yaw < 0:
                    yaw += 360
                telemetry["yaw_deg"] = yaw


# =========================
# VISION FUNCTIONS
# =========================

def get_best_red_target(results):
    best = None
    best_conf = 0.0

    for result in results:
        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])

            if cls_name != TARGET_CLASS:
                continue

            if conf < CONF_THRESHOLD:
                continue

            if conf > best_conf:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                best = {
                    "conf": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                }
                best_conf = conf

    return best


def clamp(value, low, high):
    return max(low, min(high, value))


# =========================
# MAIN AUTONOMY
# =========================

def main():
    threading.Thread(target=mavlink_reader, daemon=True).start()

    print("Waiting for MAVLink...")
    while True:
        with telemetry_lock:
            connected = telemetry["connected"]
        if connected:
            break
        time.sleep(0.1)

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("Opening USB camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("Could not open USB camera")

    print("Camera opened.")
    print("Starting autonomous package drop logic.")

    target_centered_since = None
    dropped = False
    offboard_started = False

    last_setpoint_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No camera frame.")
                continue

            h, w = frame.shape[:2]

            with telemetry_lock:
                alt_m = telemetry["alt_m"]
                mode = telemetry["mode"]
                armed = telemetry["armed"]

            if alt_m is None:
                print("Waiting for altitude telemetry...")
                time.sleep(0.2)
                continue

            results = model.predict(
                frame,
                conf=CONF_THRESHOLD,
                imgsz=IMGSZ,
                device=0,
                verbose=False
            )

            target = get_best_red_target(results)

            annotated = results[0].plot()

            if target is None:
                target_centered_since = None

                if offboard_started:
                    send_body_velocity(0.0, 0.0, 0.0)

                cv2.putText(
                    annotated,
                    "SEARCHING FOR red-target",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

            else:
                err_x = (target["cx"] - (w / 2.0)) / (w / 2.0)
                err_y = (target["cy"] - (h / 2.0)) / (h / 2.0)

                centered = abs(err_x) < CENTER_TOLERANCE and abs(err_y) < CENTER_TOLERANCE

                if not offboard_started:
                    print("Target found. Starting OFFBOARD stream.")

                    for _ in range(30):
                        send_body_velocity(0.0, 0.0, 0.0)
                        time.sleep(0.05)

                    px4_set_mode("OFFBOARD")
                    offboard_started = True
                    time.sleep(0.5)

                vy = clamp(SIDE_SIGN * KP_X * err_x, -MAX_ALIGN_SPEED, MAX_ALIGN_SPEED)
                vx = clamp(FORWARD_SIGN * KP_Y * err_y, -MAX_ALIGN_SPEED, MAX_ALIGN_SPEED)
                vz = 0.0

                send_body_velocity(vx, vy, vz)
                last_setpoint_time = time.time()

                if centered:
                    if target_centered_since is None:
                        target_centered_since = time.time()

                    centered_time = time.time() - target_centered_since

                    if (
                        centered_time >= CENTER_HOLD_TIME
                        and not dropped
                        and MIN_DROP_ALT_M <= alt_m <= MAX_DROP_ALT_M
                    ):
                        send_body_velocity(0.0, 0.0, 0.0)
                        time.sleep(0.3)
                        release_package()
                        dropped = True

                        print("Package dropped successfully.")

                        if POST_DROP_MODE in ["LOITER", "RTL", "MISSION"]:
                            px4_set_mode(POST_DROP_MODE)

                        break

                else:
                    target_centered_since = None

                cv2.putText(
                    annotated,
                    f"TARGET conf={target['conf']:.2f} err_x={err_x:.2f} err_y={err_y:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                annotated,
                f"mode={mode} armed={armed} alt={alt_m:.2f}m dropped={dropped}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            cv2.imshow("Autonomous Package Drop", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("Manual script stop requested.")
                break

            if offboard_started and time.time() - last_setpoint_time > 0.3:
                send_body_velocity(0.0, 0.0, 0.0)
                last_setpoint_time = time.time()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Script ended.")


if __name__ == "__main__":
    main()
