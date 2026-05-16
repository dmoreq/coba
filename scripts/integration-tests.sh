#!/bin/bash

# Integration Test Runner
#
# Ensures backend is running, executes integration tests, and cleans up
#
# Usage:
#   bash scripts/integration-tests.sh              # Run once
#   bash scripts/integration-tests.sh --watch      # Watch mode

set -e

BACKEND_URL="http://localhost:8000"
BACKEND_PID=""
STARTED_BACKEND=false
WATCH_MODE=false

# Parse args
if [ "$1" = "--watch" ]; then
  WATCH_MODE=true
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔍 Checking backend at $BACKEND_URL${NC}"

# Check if backend is running
if curl -s "$BACKEND_URL/docs" > /dev/null 2>&1; then
  echo -e "${GREEN}✓ Backend already running${NC}"
else
  echo -e "${YELLOW}🚀 Starting backend...${NC}"

  # Start backend in background
  cd "$(dirname "$0")/../web/backend"
  python3 -m uvicorn app.main:app --reload --port 8000 > /tmp/backend.log 2>&1 &
  BACKEND_PID=$!
  STARTED_BACKEND=true

  echo -e "${YELLOW}  Backend PID: $BACKEND_PID${NC}"
  echo -e "${YELLOW}  Waiting for backend to start...${NC}"

  # Wait for backend to be ready (max 10 seconds)
  for i in {1..20}; do
    if curl -s "$BACKEND_URL/docs" > /dev/null 2>&1; then
      echo -e "${GREEN}✓ Backend started${NC}"
      sleep 1  # Extra safety margin
      break
    fi
    echo -n "."
    sleep 0.5
  done

  if ! curl -s "$BACKEND_URL/docs" > /dev/null 2>&1; then
    echo -e "${RED}✗ Backend failed to start${NC}"
    echo "Backend logs:"
    cat /tmp/backend.log
    exit 1
  fi
fi

echo -e "\n${YELLOW}🧪 Running integration tests...${NC}"

# Change to frontend directory
cd "$(dirname "$0")/../web/frontend"

# Run integration tests
if [ "$WATCH_MODE" = true ]; then
  echo -e "${YELLOW}  Watch mode enabled${NC}"
  npm run test:integration:watch
else
  npm run test:integration
fi

TEST_EXIT_CODE=$?

echo -e "\n${YELLOW}🧹 Cleanup${NC}"

# Kill backend if we started it
if [ "$STARTED_BACKEND" = true ] && [ ! -z "$BACKEND_PID" ]; then
  echo -e "${YELLOW}  Stopping backend (PID: $BACKEND_PID)...${NC}"
  kill $BACKEND_PID 2>/dev/null || true
  wait $BACKEND_PID 2>/dev/null || true
  echo -e "${GREEN}✓ Backend stopped${NC}"
fi

# Final status
echo -e "\n${YELLOW}════════════════════════════════════════${NC}"
if [ $TEST_EXIT_CODE -eq 0 ]; then
  echo -e "${GREEN}✓ Integration tests passed${NC}"
  echo -e "${YELLOW}════════════════════════════════════════${NC}"
  exit 0
else
  echo -e "${RED}✗ Integration tests failed${NC}"
  echo -e "${YELLOW}════════════════════════════════════════${NC}"
  exit 1
fi
