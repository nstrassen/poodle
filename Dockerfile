FROM python:3.11-slim

# Only third-party dependency
RUN pip install --no-cache-dir tiktoken

WORKDIR /app

# Copy only the modules the UI server needs
COPY demo/            ./demo/
COPY cost_break_even/ ./cost_break_even/
COPY ui/              ./ui/

EXPOSE 8080

# Bind to 0.0.0.0 so Docker can forward the port to the host
CMD ["python", "ui/server.py", "--host", "0.0.0.0", "--port", "8080"]
