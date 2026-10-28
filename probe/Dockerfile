FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap util-linux procps && rm -rf /var/lib/apt/lists/*
RUN useradd -m probe
USER probe
WORKDIR /home/probe
COPY probe/app.py app.py
CMD ["python", "app.py"]