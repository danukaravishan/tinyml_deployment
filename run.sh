#!/usr/bin/env bash
# Run main.py as a background daemon, capturing stdout/stderr

# start in background with nohup, redirect output
nohup python main.py > logs/main.out.log 2>&1 &

# record the PID
echo $! > main.pid
echo "main.py started with PID $(cat main.pid)"