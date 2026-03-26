#!/bin/bash

# Create necessary directories
mkdir -p /app/logs /app/backups

# Load cron job and start cron service
crontab /etc/cron.d/update_daily
service cron start

# Start Dash in the background
cd /app
python dashboard.py >> /app/logs/dashboard.log 2>&1 &

# Keep the container running
wait