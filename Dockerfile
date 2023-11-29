from lewangdev/faster-whisper-base:tiny

COPY . /src

WORKDIR /src

CMD ["uvicorn", "openaiapi:app", "--proxy-headers", "--log-config", "log-config.yml", "--host", "0.0.0.0", "--port", "80"]
