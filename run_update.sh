#!/usr/bin/env bash
cd /app || exit 1
/usr/local/bin/python update_daily.py >> /app/logs/update_daily.log 2>&1
