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
BUCKET_CLASS = "bucket"

CAMERA_INDEX = 0

PIXHAWK_PORT = "/dev/ttyTHS1"
BAUD_RATE = 57600

# Put competition-provided bucket target coordinates here
TARGET_LAT = 35.0000000
TARGET_LON = -118.0000000

CONF_THRESHOLD = 0.45
IMGSZ = 320

# Altitudes
APPROACH_ALT_M = 8.0          # fly to GPS target at this altitude
VISION_ALT_M = 1.22           # 4 ft above ground
PICKUP_ALT_M = 0.45           # final pickup height, tune carefully
RETURN_ALT_M = 5.0

# Safety
MIN_GPS_FIX = 3
GPS_REACHED_RADIUS_M = 2.0
HOME_REACHED_RADIUS_M = 2.0
MAX_FLIGHT_SPEED = 2.0
MAX_ALIGN_SPEED = 0.45
MAX_DESCENT_SPEED = 0.25
MAX_CLIMB_SPEED = 0.6

# Vision alignment
CENTER_TOLERANCE = 0.10
CENTER_HOLD_TIME = 1.5

KP_GPS = 0.45
KP_X = 0.45
KP_Y = 0.45

# Flip these during testing if movement is reversed
SIDE_SIGN = 1.0
FORWARD_SIGN = 1.0

# Gripper servo
GRIPPER_SERVO_CHANNEL = 9
GRIPPER_OPEN_PWM = 1000
GRIPPER_CLOSE_PWM = 2000
GRIPPER_HOLD_SECONDS = 1.5

# Return behavior
RELEASE_AT_HOME = True
POST_MISSION_MODE = "LOITER"


# =========================
# TELEMETRY
# =========================

telemetry = {
    "connected": False,
    "mode": "UNKNOWN",
    "armed": False,
    "lat": None,
    "lon": None,
    "alt_m": None,
    "yaw_deg": None,
    "gps_fix": None,
    "sats": None,
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


def send_local_velocity(vn, ve, vd):
    """
    LOCAL_NED:
    vn = north velocity m/s
    ve = east velocity m/s
    vd = down velocity m/s
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
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        0, 0, 0,
        vn, ve, vd,
        0, 0, 0,
        0, 0
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


def set_gripper(pwm):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        GRIPPER_SERVO_CHANNEL,
        pwm,
        0, 0, 0, 0, 0
    )


def open_gripper():
    print("Opening gripper")
    set_gripper(GRIPPER_OPEN_PWM)
    time.sleep(GRIPPER_HOLD_SECONDS)


def close_gripper():
    print("Closing gripper")
    set_gripper(GRIPPER_CLOSE_PWM)
    time.sleep(GRIPPER_HOLD_SECONDS)


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

            elif msg_type == "GPS_RAW_INT":
                telemetry["gps_fix"] = msg.fix_type
                telemetry["sats"] = msg.satellites_visible

            elif msg_type == "ATTITUDE":
                yaw = math.degrees(msg.yaw)
                if yaw < 0:
                    yaw += 360.0
                telemetry["yaw_deg"] = yaw


# =========================
# NAVIGATION HELPERS
# =========================

def clamp(value, low, high):
    return max(low, min(high, value))


def gps_error_m(current_lat, current_lon, target_lat, target_lon):
    """
    Returns north/east error in meters from current position to target.
    """
    earth_radius = 6378137.0

    lat1 = math.radians(current_lat)
    lat2 = math.radians(target_lat)

    dlat = math.radians(target_lat - current_lat)
    dlon = math.radians(target_lon - current_lon)

    north = dlat * earth_radius
    east = dlon * earth_radius * math.cos((lat1 + lat2) / 2.0)

    return north, east


def distance_m(north, east):
    return math.sqrt(north * north + east * east)


def navigate_to_gps(target_lat, target_lon, target_alt_m):
    with telemetry_lock:
        lat = telemetry["lat"]
        lon = telemetry["lon"]
        alt = telemetry["alt_m"]

    if lat is None or lon is None or alt is None:
        send_local_velocity(0, 0, 0)
        return False

    north, east = gps_error_m(lat, lon, target_lat, target_lon)
    dist = distance_m(north, east)

    vn = clamp(KP_GPS * north, -MAX_FLIGHT_SPEED, MAX_FLIGHT_SPEED)
    ve = clamp(KP_GPS * east, -MAX_FLIGHT_SPEED, MAX_FLIGHT_SPEED)

    alt_error = target_alt_m - alt
    vd = clamp(-0.5 * alt_error, -MAX_CLIMB_SPEED, MAX_DESCENT_SPEED)

    send_local_velocity(vn, ve, vd)

    return dist <= GPS_REACHED_RADIUS_M and abs(alt_error) < 1.0


# =========================
# VISION
# =========================

def get_best_bucket(results):
    best = None
    best_conf = 0.0

    for result in results:
        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])

            if cls_name != BUCKET_CLASS:
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

    print("Waiting for GPS...")
    while True:
        with telemetry_lock:
            gps_fix = telemetry["gps_fix"]
            lat = telemetry["lat"]
            lon = telemetry["lon"]
        if gps_fix is not None and gps_fix >= MIN_GPS_FIX and lat is not None and lon is not None:
            break
        print("Waiting for GPS fix...")
        time.sleep(1.0)

    with telemetry_lock:
        home_lat = telemetry["lat"]
        home_lon = telemetry["lon"]

    print(f"Home captured: {home_lat}, {home_lon}")

    print("Loading YOLO...")
    model = YOLO(MODEL_PATH)

    print("Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("Could not open USB camera")

    open_gripper()

    print("Starting OFFBOARD stream...")
    for _ in range(40):
        send_local_velocity(0, 0, 0)
        time.sleep(0.05)

    px4_set_mode("OFFBOARD")
    print("OFFBOARD requested.")

    state = "GOTO_TARGET_GPS"
    centered_since = None
    picked_up = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame failed")
                send_local_velocity(0, 0, 0)
                continue

            h, w = frame.shape[:2]

            with telemetry_lock:
                alt_m = telemetry["alt_m"]
                mode = telemetry["mode"]
                armed = telemetry["armed"]

            if alt_m is None:
                send_local_velocity(0, 0, 0)
                continue

            results = model.predict(
                frame,
                conf=CONF_THRESHOLD,
                imgsz=IMGSZ,
                device=0,
                verbose=False
            )

            bucket = get_best_bucket(results)
            annotated = results[0].plot()

            if state == "GOTO_TARGET_GPS":
                reached = navigate_to_gps(TARGET_LAT, TARGET_LON, APPROACH_ALT_M)

                if reached:
                    print("Reached GPS target. Descending to vision altitude.")
                    state = "DESCEND_TO_VISION"

            elif state == "DESCEND_TO_VISION":
                if alt_m > VISION_ALT_M:
                    send_local_velocity(0, 0, MAX_DESCENT_SPEED)
                else:
                    print("At 4 ft vision altitude. Starting bucket correction.")
                    state = "VISION_ALIGN"

            elif state == "VISION_ALIGN":
                if bucket is None:
                    send_body_velocity(0, 0, 0)
                    centered_since = None
                    print("Bucket not detected. Holding position.")

                else:
                    err_x = (bucket["cx"] - (w / 2.0)) / (w / 2.0)
                    err_y = (bucket["cy"] - (h / 2.0)) / (h / 2.0)

                    centered = abs(err_x) < CENTER_TOLERANCE and abs(err_y) < CENTER_TOLERANCE

                    vy = clamp(SIDE_SIGN * KP_X * err_x, -MAX_ALIGN_SPEED, MAX_ALIGN_SPEED)
                    vx = clamp(FORWARD_SIGN * KP_Y * err_y, -MAX_ALIGN_SPEED, MAX_ALIGN_SPEED)
                    send_body_velocity(vx, vy, 0.0)

                    if centered:
                        if centered_since is None:
                            centered_since = time.time()

                        if time.time() - centered_since >= CENTER_HOLD_TIME:
                            print("Bucket centered. Descending for pickup.")
                            state = "DESCEND_PICKUP"
                    else:
                        centered_since = None

            elif state == "DESCEND_PICKUP":
                if bucket is not None:
                    err_x = (bucket["cx"] - (w / 2.0)) / (w / 2.0)
                    err_y = (bucket["cy"] - (h / 2.0)) / (h / 2.0)

                    vy = clamp(SIDE_SIGN * KP_X * err_x, -MAX_ALIGN_SPEED, MAX_ALIGN_SPEED)
                    vx = clamp(FORWARD_SIGN * KP_Y * err_y, -MAX_ALIGN_SPEED, MAX_ALIGN_SPEED)
                else:
                    vx = 0.0
                    vy = 0.0

                if alt_m > PICKUP_ALT_M:
                    send_body_velocity(vx, vy, MAX_DESCENT_SPEED)
                else:
                    send_body_velocity(0, 0, 0)
                    time.sleep(0.5)
                    close_gripper()
                    picked_up = True
                    print("Pickup attempted. Climbing.")
                    state = "CLIMB_WITH_BUCKET"

            elif state == "CLIMB_WITH_BUCKET":
                if alt_m < RETURN_ALT_M:
                    send_local_velocity(0, 0, -MAX_CLIMB_SPEED)
                else:
                    print("Return altitude reached. Returning home.")
                    state = "RETURN_HOME"

            elif state == "RETURN_HOME":
                reached_home = navigate_to_gps(home_lat, home_lon, RETURN_ALT_M)

                if reached_home:
                    print("Reached home.")
                    send_local_velocity(0, 0, 0)

                    if RELEASE_AT_HOME:
                        print("Lowering near home before release.")
                        state = "LOWER_AT_HOME"
                    else:
                        px4_set_mode(POST_MISSION_MODE)
                        break

            elif state == "LOWER_AT_HOME":
                if alt_m > 1.2:
                    send_local_velocity(0, 0, MAX_DESCENT_SPEED)
                else:
                    send_local_velocity(0, 0, 0)
                    open_gripper()
                    print("Bucket released at home.")
                    state = "MISSION_COMPLETE"

            elif state == "MISSION_COMPLETE":
                send_local_velocity(0, 0, 0)
                px4_set_mode(POST_MISSION_MODE)
                break

            cv2.putText(
                annotated,
                f"state={state} mode={mode} armed={armed} alt={alt_m:.2f} picked={picked_up}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            if bucket is not None:
                cv2.putText(
                    annotated,
                    f"bucket conf={bucket['conf']:.2f}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Autonomous Package Recovery", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("Manual stop requested.")
                break

            time.sleep(0.05)

    finally:
        send_local_velocity(0, 0, 0)
        cap.release()
        cv2.destroyAllWindows()
        print("Script ended.")


if __name__ == "__main__":
    main()
