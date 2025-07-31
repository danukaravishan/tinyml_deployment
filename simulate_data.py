import socket
import time

INPUT_FILE = "data/sample_input.txt"
TARGET_PORT = 8888
TARGET_IP = "127.0.0.1"
SEND_INTERVAL = 0.1  # seconds between sends (~20 Hz)

def extract_data_line(line):
    prefix = "Decoded data (first 500 chars): "
    if prefix in line:
        start = line.find(prefix) + len(prefix)
        data = line[start:].strip()
        if data.endswith("..."):
            data = data[:-3]  # remove trailing "..."
        return data
    return None

def send_data():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while True:
        try:
            with open(INPUT_FILE, "r") as f:
                for line in f:
                    payload = extract_data_line(line)
                    if payload:
                        sock.sendto(payload.encode("utf-8"), (TARGET_IP, TARGET_PORT))
                        print(f"Sent packet: {payload[:60]}...")  # optional debug
                        time.sleep(SEND_INTERVAL)
        except Exception as e:
            print(f"Error reading file or sending data: {e}")
            time.sleep(1)  # wait before retrying

if __name__ == "__main__":
    send_data()
