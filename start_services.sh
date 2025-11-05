#!/bin/bash

# Create necessary directories
mkdir -p /app/logs /app/backups

# Start cron and touch cron to reload /etc/cron.d files
service cron start
touch /etc/cron.d/update_daily

# Start services
cd /app
uvicorn api:app --host 0.0.0.0 --port 8010 &
python dashboard.py