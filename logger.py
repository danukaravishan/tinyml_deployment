import logging
import psutil
import os
from datetime import datetime, timedelta
from collections import deque
import threading
import time

# Setup logger with 24-hour timestamp
logging.basicConfig(
    filename='opt/log/earthquake.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # %H = 24-hour clock
)

# global deque to store detection timestamps
detection_times = deque()

def get_cpu_temp():
    """Read CPU temperature from sysfs (Raspberry-Pi style)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_raw = f.read().strip()
        return float(temp_raw) / 1000.0
    except Exception:
        return None
    
def log_system_status(interval=900):
    """Every `interval` seconds: log CPU, MEM, TEMP and count of detections in last interval."""
    cpu = psutil.cpu_percent()
    proc = psutil.Process(os.getpid())
    mem = proc.memory_percent()
    #mem = psutil.virtual_memory().percent
    cpu_temp = get_cpu_temp()
    temp_str = f"{cpu_temp:.1f}°C" if cpu_temp is not None else "N/A"

    now = datetime.now()
    cutoff = now - timedelta(seconds=interval)
    # remove old detection timestamps
    while detection_times and detection_times[0] < cutoff:
        detection_times.popleft()
    count = len(detection_times)

    logging.info(
        f"SYSTEM STATUS: CPU={cpu:.1f}%, MEM={mem:.1f}%, TEMP={temp_str}, "
        f"DETECTIONS_LAST_{interval//60}_MIN= {count}"
    )

def log_detection():
    """Log only the detection time and record it."""
    now = datetime.now()
    detection_times.append(now)
    logging.info(f"EARTHQUAKE DETECTED at {now:%Y-%m-%d %H:%M:%S}")

def log_vibration():
    """Log only the vibration detection time and record it."""
    now = datetime.now()
    logging.info(f"VIBRATION DETECTED at {now:%Y-%m-%d %H:%M:%S}, NOT EARTHQUAKE")


def start_periodic_logging(interval=900):
    """Thread target to call log_system_status every `interval` seconds."""
    while True:
        log_system_status(interval)
        time.sleep(interval)
