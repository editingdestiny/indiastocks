#!/bin/bash

# Create necessary directories
mkdir -p /app/logs /app/backups

# Load cron job and start cron service
crontab /etc/cron.d/update_daily
service cron start

# Start services
cd /app
python dashboard.py