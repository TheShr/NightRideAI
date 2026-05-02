import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import Dashboard from "./components/Dashboard";
import "./App.css";

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [detections, setDetections] = useState([]);
  const [potholes, setPotholes] = useState([]);
  const [hazards, setHazards] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [fps, setFps] = useState(0);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await axios.get("http://localhost:8000/health");
      setIsConnected(response.status === 200);
    } catch (error) {
      setIsConnected(false);
      console.error("Backend not available:", error);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });

      const video = videoRef.current;
      if (!video) return;

      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      streamRef.current = stream;

      const playPromise = video.play();
      if (playPromise !== undefined) {
        await playPromise.catch((playError) => {
          console.warn("Video playback could not start automatically:", playError);
        });
      }

      await new Promise((resolve, reject) => {
        const timeout = window.setTimeout(() => {
          video.removeEventListener("canplay", onCanPlay);
          reject(new Error("Camera ready timeout"));
        }, 3000);

        function onCanPlay() {
          window.clearTimeout(timeout);
          video.removeEventListener("canplay", onCanPlay);
          resolve();
        }

        video.addEventListener("canplay", onCanPlay);
      });
    } catch (error) {
      console.error("Error accessing camera:", error);
    }
  };

  const stopCamera = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    setCurrentFrame(null);
    setDetections([]);
    setPotholes([]);
    setHazards([]);
    setAlerts([]);
    setFps(0);
  };

  const captureFrame = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (!canvas || !video || video.videoWidth === 0 || video.videoHeight === 0) {
      return null;
    }

    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    return canvas.toDataURL("image/jpeg", 0.8);
  };

  const processFrame = async () => {
    if (!isConnected || !videoRef.current || videoRef.current.readyState < 2) return;

    try {
      const frameData = captureFrame();
      if (!frameData) return;

      const base64Data = frameData.split(",")[1];
      if (!base64Data) return;

      setCurrentFrame(base64Data);

      const response = await axios.post(
        "http://localhost:8000/predict",
        {
          image: base64Data,
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        },
      );

      setDetections(response.data.detections || []);
      setPotholes(response.data.potholes || []);
      setHazards(response.data.hazards || []);
      setAlerts(response.data.alerts || []);
      setFps(response.data.fps || 0);
      setCurrentFrame(response.data.enhanced_image || base64Data);
    } catch (error) {
      console.error("Frame processing error:", error);
    }
  };

  const startProcessing = async () => {
    await startCamera();

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    intervalRef.current = setInterval(processFrame, 100); // ~10 FPS
    setTimeout(processFrame, 200);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <header className="bg-gray-800 p-4">
        <h1 className="text-2xl font-bold text-center">NightRide AI</h1>
        <p className="text-center text-gray-400">
          Low-Light Hazard Detection for Two-Wheelers
        </p>
      </header>

      <main className="container mx-auto p-4">
        {!isConnected && (
          <div className="bg-red-600 text-white p-4 rounded mb-4">
            Backend not connected. Please start the backend server.
          </div>
        )}

        <Dashboard
          isConnected={isConnected}
          currentFrame={currentFrame}
          detections={detections}
          potholes={potholes}
          hazards={hazards}
          alerts={alerts}
          fps={fps}
          onStartProcessing={startProcessing}
          onStopProcessing={stopCamera}
        />
      </main>

      {/* Hidden video and canvas for capture */}
      <video ref={videoRef} className="hidden" autoPlay muted />
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}

export default App;
