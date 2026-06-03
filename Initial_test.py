from pymavlink import mavutil
import time
import math


# =========================
# SETTINGS
# =========================

PIXHAWK_PORT = "/dev/ttyTHS1"
BAUD_RATE = 57600

TAKEOFF_ALT_M = 3.0
HOLD_SECONDS = 10

SERVO_CHANNEL = 9
SERVO_PWM_LOW = 1000
SERVO_PWM_HIGH = 2000


# =========================
# BASIC FUNCTIONS
# =========================

def connect():
    print("Connecting to Pixhawk...")
    master = mavutil.mavlink_connection(PIXHAWK_PORT, baud=BAUD_RATE)
    master.wait_heartbeat()
    print(f"Connected: system={master.target_system}, component={master.target_component}")
    return master


def arm(master):
    print("Sending ARM command...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,
        0, 0, 0, 0, 0, 0
    )


def disarm(master):
    print("Sending DISARM command...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0,
        0, 0, 0, 0, 0, 0
    )


def takeoff(master, altitude_m):
    print(f"Sending TAKEOFF command to {altitude_m} m...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0,
        0,
        0,
        float("nan"),
        0,
        0,
        altitude_m
    )


def land(master):
    print("Sending LAND command...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0,
        0, 0, 0, float("nan"),
        0, 0, 0
    )


def rtl(master):
    print("Sending RTL command...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0,
        0, 0, 0, 0, 0, 0, 0
    )


def set_servo(master, channel, pwm):
    print(f"Setting servo channel {channel} to PWM {pwm}...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        channel,
        pwm,
        0, 0, 0, 0, 0
    )


def print_telemetry(master, seconds=10):
    print(f"Printing telemetry for {seconds} seconds...")
    start = time.time()

    while time.time() - start < seconds:
        msg = master.recv_match(blocking=True, timeout=1)

        if msg is None:
            continue

        msg_type = msg.get_type()

        if msg_type == "HEARTBEAT":
            mode = mavutil.mode_string_v10(msg)
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            print(f"HEARTBEAT | mode={mode} | armed={armed}")

        elif msg_type == "GLOBAL_POSITION_INT":
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = msg.relative_alt / 1000.0
            print(f"POSITION | lat={lat:.7f} | lon={lon:.7f} | rel_alt={alt:.2f} m")

        elif msg_type == "GPS_RAW_INT":
            print(f"GPS | fix={msg.fix_type} | sats={msg.satellites_visible}")

        elif msg_type == "ATTITUDE":
            yaw = math.degrees(msg.yaw)
            if yaw < 0:
                yaw += 360
            print(f"ATTITUDE | yaw={yaw:.1f} deg")


# =========================
# MAIN TEST AREA
# =========================

def main():
    master = connect()

    # =========================================================
    # STEP 1: TELEMETRY ONLY
    # Safe with props on or off.
    # Confirms Jetson can read Pixhawk telemetry.
    # =========================================================

    print_telemetry(master, seconds=10)


    # =========================================================
    # STEP 2: SERVO / RELEASE MECHANISM TEST
    # PROPS OFF.
    # Uncomment this to test gripper/drop servo.
    # =========================================================

    # set_servo(master, SERVO_CHANNEL, SERVO_PWM_HIGH)
    # time.sleep(2)
    # set_servo(master, SERVO_CHANNEL, SERVO_PWM_LOW)
    # time.sleep(2)


    # =========================================================
    # STEP 3: ARM / DISARM TEST
    # PROPS OFF.
    # Confirms Jetson can command the Pixhawk.
    # =========================================================

    # arm(master)
    # time.sleep(5)
    # disarm(master)


    # =========================================================
    # STEP 4: TAKEOFF TEST
    # PROPS ON ONLY IN SAFE OPEN TEST AREA.
    # Safety pilot must be ready to switch modes / RTL.
    # =========================================================

    # arm(master)
    # time.sleep(3)
    # takeoff(master, TAKEOFF_ALT_M)
    # time.sleep(HOLD_SECONDS)
    # land(master)


    # =========================================================
    # STEP 5: TAKEOFF THEN RTL TEST
    # PROPS ON ONLY IN SAFE OPEN TEST AREA.
    # =========================================================

    # arm(master)
    # time.sleep(3)
    # takeoff(master, TAKEOFF_ALT_M)
    # time.sleep(HOLD_SECONDS)
    # rtl(master)


if __name__ == "__main__":
    main()
