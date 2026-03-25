#!/usr/bin/env bash
set -e

IMAGE="poodle-ui"
PORT="${PORT:-8080}"

# Build the image (cached on subsequent runs)
docker build -t "$IMAGE" .

# Remove a previous container with the same name if it exists
docker rm -f "$IMAGE" 2>/dev/null || true

# Run – mount ui/ so edits to index.html / scenarios.json are live without rebuild
docker run \
  --name "$IMAGE" \
  -p "$PORT:8080" \
  -v "$(pwd)/ui:/app/ui" \
  -v "$(pwd)/demo:/app/demo" \
  "$IMAGE"
