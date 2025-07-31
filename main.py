import time
import queue
import threading
import socket
import ast
import threading
import numpy as np
from enum import Enum
from logger import start_periodic_logging, log_detection, log_vibration
from datetime import datetime

SAMPLE_RATE = 100
WINDOW_DURATION = 2
WINDOW_SIZE = SAMPLE_RATE * WINDOW_DURATION
PORT = 8888


data_queue = queue.Queue(maxsize=2000)
detection_count = 0

class STATE(Enum):
    IDLE           = 1
    READ           = 2
    INFERENCE      = 3
    ALARM          = 4

channel_buffers = {
    'ENN': [],
    'ENE': [],
    'ENZ': []
}

EXPECTED_CHANNELS = ['ENN', 'ENE', 'ENZ'] # Omit the EHZ channel

def resample_segment(segment, original_rate=100, target_rate=50):
    segment = np.array(segment)
    if segment.shape[0] != 3:
        segment = segment.T  # convert to (3, N)

    n_samples = segment.shape[1]
    new_samples = int(n_samples * target_rate / original_rate)
    
    # Create new time indices
    old_indices = np.linspace(0, n_samples - 1, n_samples)
    new_indices = np.linspace(0, n_samples - 1, new_samples)
    
    # Interpolate each channel
    resampled = np.array([np.interp(new_indices, old_indices, ch) for ch in segment])
    return resampled  # shape: (3, new_samples)


def normalize_segment_data(data):
    for i in range(data.shape[0]):  # For each channel
        channel = data[i, :]
        min_val = np.min(channel)
        max_val = np.max(channel)
        if max_val - min_val != 0:
            # In-place normalization
            channel -= min_val
            channel /= (max_val - min_val)
            channel *= 2
            channel -= 1
        else:
            channel.fill(0)  # Set to zero if no variation
    return data


def parse_data_packet(packet_str):
    try:
        # Convert string like "{'ENN', 12345678.9, -123, -456, ...}" into Python data
        packet_str = packet_str.replace('{', '[').replace('}', ']')
        parsed = ast.literal_eval(packet_str)
        channel = parsed[0]
        values = parsed[2:]  # Skip timestamp
        return channel, values
    except Exception as e:
        print(f" {datetime.now()} Failed to parse packet: {e}")
        return None, None


def data_reader():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(('', PORT))
    except socket.error as e:
        print(f" {datetime.now()} Socket error: {e}")
        return

    print(f" {datetime.now()} Listening for data on UDP port {PORT}...")

    while True:
        try:
            data, _ = sock.recvfrom(4096)
            decoded = data.decode('utf-8', errors='ignore')
            channel, values = parse_data_packet(decoded)
            if channel in EXPECTED_CHANNELS and values:
                channel_buffers[channel].extend(values)

            # Synchronize when all 3 components have enough samples
            min_len = min(len(channel_buffers[ch]) for ch in EXPECTED_CHANNELS)
            while min_len > 0:
                # Pop one sample from each buffer and form [x, y, z]
                sample = [channel_buffers['ENN'].pop(0),
                          channel_buffers['ENE'].pop(0),
                          channel_buffers['ENZ'].pop(0)]
                data_queue.put(sample)
                min_len -= 1

        except Exception as e:
            print(f" {datetime.now()} Error in receiving/parsing data: {e}")
            continue


# Data processing thread
def data_processor():

    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter # Run in different environment without tflite_runtime

    interpreter = Interpreter(model_path="models/model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    buffer = []
    segment = None
    state = STATE.IDLE

    while True:
        if state == STATE.IDLE:
            detection_count = 0
            buffer = []
            print(f" {datetime.now()} IDLE STATE: Waiting for data...")
            state = STATE.READ

        elif state == STATE.READ:
            segment = None
            try:
                sample = data_queue.get(timeout=1)
                buffer.append(sample)

                if len(buffer) >= WINDOW_SIZE:
                    segment = buffer[:WINDOW_SIZE]
                    buffer = buffer[WINDOW_SIZE // 5:]   # Keep 75% of the buffer, making the overlap
                    segment = np.array(segment)
                    state = STATE.INFERENCE
                else:
                    state = STATE.READ
                    
            except queue.Empty:
                state = STATE.READ
                print(f" {datetime.now()} No data available, waiting...")

        elif state == STATE.INFERENCE:
            if segment is None or segment.shape[1] != 3 or segment.shape[0] != WINDOW_SIZE:
                print(f" {datetime.now()} Invalid segment, resetting")
                state = STATE.READ
                continue

            segment = segment.T
            segment = resample_segment(segment)
            segment = normalize_segment_data(segment)
            inp = segment.reshape(1, 3, 100).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], inp)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]['index'])[0]

            if out[0] > 0.57:
                detection_count += 1
                log_vibration()
                print(f" {datetime.now()} Detection {detection_count}")
            else:
                detection_count = 0

            if detection_count >= 4:
                print(f" {datetime.now()} ALARM triggered!")
                state = STATE.ALARM
                detection_count = 0
                continue
                
            else:
                state = STATE.READ

        elif state == STATE.ALARM:
            print(f" {datetime.now()} ======== ALARM STAGE =========")
            log_detection()
            state = STATE.IDLE


# Main execution
if __name__ == "__main__":
    t1 = threading.Thread(target=data_reader, daemon=True)
    t2 = threading.Thread(target=data_processor, daemon=True)
    t1.start()
    t2.start()

    threading.Thread(target=start_periodic_logging, args=(300,), daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")