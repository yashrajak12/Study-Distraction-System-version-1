# ---------- Base Image ----------
FROM python:3.12-slim

# ---------- Set Working Directory ----------
WORKDIR /app

# ---------- Install System Dependencies ----------
# OpenCV needs these
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---------- Copy Project Files ----------
COPY . .

# ---------- Install Python Dependencies ----------
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Expose Port ----------
EXPOSE 5000

# ---------- Set Environment Variables ----------
ENV FLASK_APP=run.py
ENV FLASK_RUN_HOST=0.0.0.0

# ---------- Run the Application ----------
CMD ["flask", "run"]
