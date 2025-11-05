FROM python:3.11-slim

# Install cron and other dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends cron \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install required packages
RUN pip install --no-cache-dir fastapi uvicorn pandas dash plotly yfinance

# Copy all necessary files
COPY docker/update_cron /etc/cron.d/update_daily
COPY run_update.sh /app/run_update.sh
COPY api.py /app/api.py
COPY dashboard.py /app/dashboard.py
COPY start_services.sh /app/start_services.sh
COPY update_daily.py /app/update_daily.py

# Set permissions
RUN chmod 0644 /etc/cron.d/update_daily && \
    chmod +x /app/run_update.sh && \
    chmod +x /app/start_services.sh

# Create runtime directories
RUN mkdir -p /app/logs /app/backups

# Start services
CMD ["/app/start_services.sh"]
