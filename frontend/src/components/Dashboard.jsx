import React, { useState } from "react";
import { Play, Square, AlertTriangle, Car, User, Dog } from "lucide-react";

const Dashboard = ({
  isConnected,
  currentFrame,
  detections,
  hazards,
  alerts,
  fps,
  onStartProcessing,
  onStopProcessing,
}) => {
  const [isProcessing, setIsProcessing] = useState(false);

  const handleStart = () => {
    setIsProcessing(true);
    onStartProcessing();
  };

  const handleStop = () => {
    setIsProcessing(false);
    onStopProcessing();
  };

  const getHazardIcon = (type) => {
    switch (type) {
      case "person":
        return <User className="w-4 h-4" />;
      case "car":
        return <Car className="w-4 h-4" />;
      case "dog":
        return <Dog className="w-4 h-4" />;
      default:
        return <AlertTriangle className="w-4 h-4" />;
    }
  };

  const getHazardColor = (level) => {
    switch (level) {
      case "danger":
        return "bg-red-600";
      case "warning":
        return "bg-yellow-600";
      default:
        return "bg-gray-600";
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Main Camera Feed */}
      <div className="lg:col-span-2">
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Camera Feed</h2>
            <div className="flex items-center space-x-2">
              <span
                className={`w-2 h-2 rounded-full ${isConnected ? "bg-green-500" : "bg-red-500"}`}
              ></span>
              <span className="text-sm">
                {isConnected ? "Connected" : "Disconnected"}
              </span>
            </div>
          </div>

          <div
            className="relative bg-black rounded-lg overflow-hidden"
            style={{ height: "400px" }}
          >
            {currentFrame ? (
              <img
                src={`data:image/jpeg;base64,${currentFrame}`}
                alt="Enhanced camera feed"
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                No feed available
              </div>
            )}

            {/* FPS Counter */}
            <div className="absolute top-2 right-2 bg-black bg-opacity-50 px-2 py-1 rounded text-sm">
              FPS: {fps.toFixed(1)}
            </div>
          </div>

          {/* Controls */}
          <div className="flex justify-center mt-4 space-x-4">
            <button
              onClick={handleStart}
              disabled={!isConnected || isProcessing}
              className="flex items-center space-x-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-4 py-2 rounded"
            >
              <Play className="w-4 h-4" />
              <span>Start Detection</span>
            </button>

            <button
              onClick={handleStop}
              disabled={!isProcessing}
              className="flex items-center space-x-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 px-4 py-2 rounded"
            >
              <Square className="w-4 h-4" />
              <span>Stop</span>
            </button>
          </div>
        </div>
      </div>

      {/* Sidebar */}
      <div className="space-y-6">
        {/* Alerts Panel */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-3 flex items-center">
            <AlertTriangle className="w-5 h-5 mr-2 text-yellow-500" />
            Alerts
          </h3>

          <div className="space-y-2 max-h-32 overflow-y-auto">
            {alerts.length === 0 ? (
              <p className="text-gray-500 text-sm">No active alerts</p>
            ) : (
              alerts.map((alert, index) => (
                <div key={index} className="bg-yellow-900 p-2 rounded text-sm">
                  {alert.message}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Detections Panel */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-3">Detections</h3>

          <div className="space-y-2 max-h-48 overflow-y-auto">
            {detections.length === 0 ? (
              <p className="text-gray-500 text-sm">No detections</p>
            ) : (
              detections.map((detection, index) => (
                <div
                  key={index}
                  className="flex items-center space-x-2 p-2 bg-gray-700 rounded"
                >
                  {getHazardIcon(detection.class)}
                  <span className="text-sm capitalize">{detection.class}</span>
                  <span className="text-xs text-gray-400">
                    {(detection.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Hazards Panel */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-3">Hazards</h3>

          <div className="space-y-2 max-h-48 overflow-y-auto">
            {hazards.length === 0 ? (
              <p className="text-gray-500 text-sm">No hazards detected</p>
            ) : (
              hazards.map((hazard, index) => (
                <div
                  key={index}
                  className={`p-2 rounded text-sm ${getHazardColor(hazard.level)}`}
                >
                  <div className="flex items-center space-x-2">
                    {getHazardIcon(
                      hazard.type === "object" ? hazard.class : "pothole",
                    )}
                    <span className="capitalize">
                      {hazard.type === "object" ? hazard.class : "Pothole"}
                    </span>
                  </div>
                  <div className="text-xs mt-1">
                    Level: {hazard.level.toUpperCase()}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
