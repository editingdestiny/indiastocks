FROM python:3.11-slim

# Install cron and other dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends cron \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install required packages
RUN pip install --no-cache-dir pandas dash plotly yfinance scikit-learn numpy tensorflow

# Copy all necessary files
COPY docker/update_cron /etc/cron.d/update_daily
COPY run_update.sh /app/run_update.sh
COPY dashboard.py /app/dashboard.py
COPY predictive_analysis.py /app/predictive_analysis.py
COPY lstm_model.py /app/lstm_model.py
COPY backtesting.py /app/backtesting.py
COPY fundamentals.py /app/fundamentals.py
COPY prediction_tracker.py /app/prediction_tracker.py
COPY start_services.sh /app/start_services.sh
COPY update_daily.py /app/update_daily.py
COPY update_fundamentals.py /app/update_fundamentals.py
COPY generate_predictions.py /app/generate_predictions.py

# Set permissions
RUN chmod 0644 /etc/cron.d/update_daily && \
    chmod +x /app/run_update.sh && \
    chmod +x /app/start_services.sh

# Create runtime directories
RUN mkdir -p /app/logs /app/backups

# Start services
CMD ["/app/start_services.sh"]
