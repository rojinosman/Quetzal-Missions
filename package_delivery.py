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

# Delivery altitude behavior
# WARNING: relative_alt is not true AGL unless takeoff point and target ground are similar.
# For real delivery, use a rangefinder if possible.
DELIVERY_ALT_M = 1.2          # about 4 ft
MIN_SAFE_ALT_M = 0.7
APPROACH_ALT_M = 5.0
POST_DELIVERY_CLIMB_ALT_M = 6.0

# Vision alignment
CENTER_TOLERANCE = 0.10
CENTER_HOLD_TIME = 2.0

MAX_ALIGN_SPEED = 0.7
MAX_DESCENT_SPEED = 0.35
MAX_CLIMB_SPEED = 0.6

KP_X = 0.55
KP_Y = 0.55

# Flip these if movement direction is reversed during testing
SIDE_SIGN = 1.0
FORWARD_SIGN = 1.0

# Servo/package release
DELIVERY_SERVO_CHANNEL = 9
PWM_RELEASE = 2000
PWM_HOLD = 1000
RELEASE_HOLD_SECONDS = 1.0

POST_DELIVERY_MODE = "LOITER"  # LOITER or RTL


# =========================
# TELEMETRY
# =========================

telemetry = {
    "connected": False,
    "mode": "UNKNOWN",
    "armed": False,
    "alt_m": None,
    "yaw_deg": None,
    "lat": None,
    "lon": None,
}

telemetry_lock = threading.Lock()
master = None


# =========================
# MAVLINK FUNCTIONS
# =========================

def px4_set_mode(mode_name):
    mode_map = {
        "OFFBOARD": (6, 0),
        "LOITER": (4, 3),
        "RTL": (4, 5),
        "MISSION": (4, 4),
    }

    main_mode, sub_mode = mode_map[mode_name]

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
    BODY_NED:
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
    print("Releasing delivery package...")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        DELIVERY_SERVO_CHANNEL,
        PWM_RELEASE,
        0, 0, 0, 0, 0
    )

    time.sleep(RELEASE_HOLD_SECONDS)

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        DELIVERY_SERVO_CHANNEL,
        PWM_HOLD,
        0, 0, 0, 0, 0
    )


def mavlink_reader():
    global master

    print("Connecting to Pixhawk...")
    master = mavutil.mavlink_connection(PIXHAWK_PORT, baud=BAUD_RATE)
    master.wait_heartbeat()
    print(f"Connected to Pixhawk system {master.target_system}")

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
                    yaw += 360.0
                telemetry["yaw_deg"] = yaw


# =========================
# VISION
# =========================

def clamp(value, low, high):
    return max(low, min(high, value))


def get_best_target(results):
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


# =========================
# MAIN
# =========================

def main():
    threading.Thread(target=mavlink_reader, daemon=True).start()

    print("Waiting for MAVLink...")
    while True:
        with telemetry_lock:
            if telemetry["connected"]:
                break
        time.sleep(0.1)

    print("Loading YOLO...")
    model = YOLO(MODEL_PATH)

    print("Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("Could not open USB camera")

    print("Camera ready.")

    print("Starting OFFBOARD setpoint stream...")
    for _ in range(40):
        send_body_velocity(0.0, 0.0, 0.0)
        time.sleep(0.05)

    px4_set_mode("OFFBOARD")
    print("OFFBOARD requested.")

    centered_since = None
    delivered = False
    state = "SEARCH_ALIGN"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame failed")
                send_body_velocity(0.0, 0.0, 0.0)
                continue

            h, w = frame.shape[:2]

            with telemetry_lock:
                alt_m = telemetry["alt_m"]
                mode = telemetry["mode"]
                armed = telemetry["armed"]

            if alt_m is None:
                print("Waiting for altitude...")
                send_body_velocity(0.0, 0.0, 0.0)
                time.sleep(0.1)
                continue

            results = model.predict(
                frame,
                conf=CONF_THRESHOLD,
                imgsz=IMGSZ,
                device=0,
                verbose=False
            )

            target = get_best_target(results)
            annotated = results[0].plot()

            vx = 0.0
            vy = 0.0
            vz = 0.0

            if target is None:
                centered_since = None
                vx = 0.0
                vy = 0.0
                vz = 0.0
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

                vy = clamp(SIDE_SIGN * KP_X * err_x, -MAX_ALIGN_SPEED, MAX_ALIGN_SPEED)
                vx = clamp(FORWARD_SIGN * KP_Y * err_y, -MAX_ALIGN_SPEED, MAX_ALIGN_SPEED)

                if centered:
                    if centered_since is None:
                        centered_since = time.time()
                    centered_time = time.time() - centered_since
                else:
                    centered_since = None
                    centered_time = 0.0

                if state == "SEARCH_ALIGN":
                    vz = 0.0

                    if centered_time >= CENTER_HOLD_TIME:
                        print("Target centered. Beginning controlled descent.")
                        state = "DESCEND"

                elif state == "DESCEND":
                    if not centered:
                        vz = 0.0
                    elif alt_m > DELIVERY_ALT_M:
                        vz = MAX_DESCENT_SPEED
                    else:
                        print("Delivery altitude reached.")
                        state = "RELEASE"
                        vz = 0.0

                elif state == "RELEASE":
                    send_body_velocity(0.0, 0.0, 0.0)
                    time.sleep(0.5)

                    if alt_m < MIN_SAFE_ALT_M:
                        print("Altitude too low, aborting release.")
                        state = "CLIMB"
                    else:
                        release_package()
                        delivered = True
                        print("Package delivered.")
                        state = "CLIMB"

                elif state == "CLIMB":
                    vx = 0.0
                    vy = 0.0

                    if alt_m < POST_DELIVERY_CLIMB_ALT_M:
                        vz = -MAX_CLIMB_SPEED
                    else:
                        print("Post-delivery climb complete.")
                        send_body_velocity(0.0, 0.0, 0.0)
                        px4_set_mode(POST_DELIVERY_MODE)
                        break

                cv2.putText(
                    annotated,
                    f"state={state} conf={target['conf']:.2f} err_x={err_x:.2f} err_y={err_y:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

            send_body_velocity(vx, vy, vz)

            cv2.putText(
                annotated,
                f"mode={mode} armed={armed} alt={alt_m:.2f}m delivered={delivered}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            cv2.imshow("Autonomous Package Delivery", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("Manual stop requested.")
                break

            time.sleep(0.05)

    finally:
        send_body_velocity(0.0, 0.0, 0.0)
        cap.release()
        cv2.destroyAllWindows()
        print("Script ended.")


if __name__ == "__main__":
    main()
