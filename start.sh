#!/bin/bash
# PriceLens Full Stack Launcher
# Starts both the FastAPI backend and Next.js frontend

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        🔍 PriceLens Full Stack             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}All servers stopped.${NC}"
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Start the FastAPI backend
echo -e "${GREEN}Starting FastAPI backend server...${NC}"
cd "$SCRIPT_DIR"
python scripts/run_api.py &
BACKEND_PID=$!
echo -e "${GREEN}✓ Backend starting on http://localhost:7848${NC}"

# Wait a moment for backend to initialize
sleep 2

# Start the Next.js frontend
echo -e "${GREEN}Starting Next.js frontend...${NC}"
cd "$SCRIPT_DIR/web"
npm run dev &
FRONTEND_PID=$!
echo -e "${GREEN}✓ Frontend starting on http://localhost:7847${NC}"

echo ""
echo -e "${BLUE}════════════════════════════════════════════${NC}"
echo -e "${GREEN}Both servers are running!${NC}"
echo -e "${BLUE}────────────────────────────────────────────${NC}"
echo -e "  📡 Backend API:  ${YELLOW}http://localhost:7848${NC}"
echo -e "  🌐 Frontend:     ${YELLOW}http://localhost:7847${NC}"
echo -e "${BLUE}────────────────────────────────────────────${NC}"
echo -e "  Press ${RED}Ctrl+C${NC} to stop both servers"
echo -e "${BLUE}════════════════════════════════════════════${NC}"
echo ""

# Wait for both processes
wait
