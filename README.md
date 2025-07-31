# TinyML Earthquake Detection Deployment

A lightweight Python application that performs real-time earthquake detection on streaming seismic data using a pre-trained TensorFlow Lite model. Designed to run on resource-constrained environments (e.g. Raspberry Pi, edge devices), it ingests UDP packets, buffers and preprocesses multi-axis data, runs on-device inference and logs alerts and system telemetry.

## Features

- UDP listener (`data_reader`) that collects and synchronizes three seismic channels (ENN, ENE, ENZ)
- Sliding-window buffer and overlap strategy for continuous inference
- On-device TFLite inference (`data_processor`) with resampling & normalization
- Threshold-based alarm and detection counter
- Periodic system telemetry logging (CPU, memory, temperature and detection count)
- Simulated data generator (`simulate_data.py`) for local testing

## Repository Structure

```
mode_deployement/
├── main.py               # Core reader + processor threads
├── logger.py             # Logging utilities (system & detection)
├── simulate_data.py      # UDP packet simulator for sample inputs
├── models/
│   └── model.tflite      # TFLite earthquake detection model
├── data/
│   └── sample_input.txt  # Recorded UDP payloads for simulation
└── README.md
```

## Requirements

- Python 3.7+
- numpy
- tensorflow (or tflite-runtime)
- psutil
- (optional) virtualenv or conda for isolated environment

## Installation

Use the provided installer script to set up pyenv, Python, dependencies, and clone the application:

```bash
git clone https://github.com/danukaravishan/tinyml_deployment.git
cd mode_deployement
chmod +x installer.sh
./installer.sh
```

The installer will:
- Install pyenv and Python 3.9.23  
- Create a project folder `ML_DETECTION`  
- Install required Python packages  
- Clone the main repo into `ML_DETECTION/tinyml_deployment`  

## Usage

1. **Launch the application**  
   ```bash
   cd ML_DETECTION/tinyml_deployment
   ./run.sh
   ```

2. **Simulate incoming data**  
   In a separate shell, replay sample data at ~20 Hz:  
   ```bash
   python simulate_data.py
   ```

3. **View logs**  
   - Detection events and system telemetry are appended to `logs/earthquake.log`  
   - Alarms are printed to the console in real time

## Logging Behavior

- **Detection log**: each seismic event that crosses the alarm threshold is timestamped.  
- **System log**: every 15 minutes, CPU usage, memory usage, CPU temperature and the number of detections in the last interval are recorded.

## Customization

- Adjust `WINDOW_DURATION`, `SAMPLE_RATE`, and alert thresholds in `main.py`.  
- Swap in your own TFLite model under `models/model.tflite`.  
- Tweak logging intervals and log file path in `logger.py`.

## License

This project is released under the MIT