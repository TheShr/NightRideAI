#!/bin/bash

# NightRide AI Launcher Script

echo "Starting NightRide AI..."

# Start backend
cd backend
python main.py &
BACKEND_PID=$!

# Start frontend
cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID