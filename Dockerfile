# --------------------------
# WalkSafe Dockerfile
# --------------------------

FROM python:3.11-slim

# Prevent Python from buffering stdout (helps with logs)
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for geopandas/osmnx
RUN apt-get update && apt-get install -y \
    gcc g++ libspatialindex-dev libgeos-dev libproj-dev proj-bin proj-data \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Run app
CMD ["python", "-m", "src.api.app", "--port", "5000"]