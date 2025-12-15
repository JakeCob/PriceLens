"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
    Scan,
    Camera,
    CameraOff,
    Volume2,
    VolumeX,
    RotateCcw,
    ArrowLeft,
    DollarSign
} from "lucide-react";
import Link from "next/link";

interface Detection {
    bbox: [number, number, number, number];
    name: string;
    card_id: string;
    set: string;
    confidence: number;
    price: number | null;
}

const API_URL = "http://localhost:8080";

export default function ScannerPage() {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isStreaming, setIsStreaming] = useState(false);
    const [detections, setDetections] = useState<Detection[]>([]);
    const [sessionTotal, setSessionTotal] = useState(0);
    const [sessionCards, setSessionCards] = useState<Map<string, Detection>>(new Map());
    const [isMuted, setIsMuted] = useState(false);
    const [fps, setFps] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const frameCountRef = useRef(0);
    const lastFpsTimeRef = useRef(Date.now());
    const streamRef = useRef<MediaStream | null>(null);
    const animationRef = useRef<number | null>(null);

    // Start camera
    const startCamera = async () => {
        try {
            setError(null);

            // Stop any existing stream first
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
            }

            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: "environment"
                }
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                streamRef.current = stream;

                // Wait for video to be ready before playing
                videoRef.current.onloadedmetadata = () => {
                    videoRef.current?.play()
                        .then(() => {
                            setIsStreaming(true);
                            startDetectionLoop();
                        })
                        .catch((err) => {
                            console.log("Play interrupted, retrying...");
                        });
                };
            }
        } catch (err) {
            setError("Could not access camera. Please allow camera permissions.");
            console.error("Camera error:", err);
        }
    };

    // Stop camera
    const stopCamera = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        if (animationRef.current) {
            cancelAnimationFrame(animationRef.current);
            animationRef.current = null;
        }
        setIsStreaming(false);
        setDetections([]);
    };

    // Send frame to backend
    const sendFrame = async () => {
        if (!videoRef.current || !canvasRef.current || !isStreaming) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        // Set canvas size to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw video frame to canvas
        ctx.drawImage(video, 0, 0);

        // Get base64 image
        const imageData = canvas.toDataURL("image/jpeg", 0.8);
        const base64Data = imageData.split(",")[1];

        try {
            const response = await fetch(`${API_URL}/detect-live`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: base64Data }),
            });

            if (response.ok) {
                const data = await response.json();
                setDetections(data.detections || []);

                // Update FPS
                frameCountRef.current++;
                const now = Date.now();
                if (now - lastFpsTimeRef.current >= 1000) {
                    setFps(frameCountRef.current);
                    frameCountRef.current = 0;
                    lastFpsTimeRef.current = now;
                }

                // Add detected cards to session
                data.detections?.forEach((det: Detection) => {
                    if (det.name !== "Scanning..." && det.price !== null) {
                        if (!sessionCards.has(det.card_id)) {
                            const newCards = new Map(sessionCards);
                            newCards.set(det.card_id, det);
                            setSessionCards(newCards);
                            setSessionTotal(prev => prev + (det.price || 0));

                            // Play sound
                            if (!isMuted) {
                                const audio = new Audio("/sounds/success.mp3");
                                audio.volume = 0.3;
                                audio.play().catch(() => { });
                            }
                        }
                    }
                });
            }
        } catch (err) {
            console.error("Detection error:", err);
        }
    };

    // Detection loop
    const startDetectionLoop = () => {
        const loop = async () => {
            await sendFrame();
            animationRef.current = requestAnimationFrame(loop);
        };
        loop();
    };

    // Draw detection boxes
    useEffect(() => {
        if (!canvasRef.current || !videoRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        // Clear and redraw video
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(videoRef.current, 0, 0);

        // Draw detection boxes
        detections.forEach((det) => {
            const [x1, y1, x2, y2] = det.bbox;
            const width = x2 - x1;
            const height = y2 - y1;

            // Box color based on state
            const isConfirmed = det.name !== "Scanning..." && det.price !== null;
            ctx.strokeStyle = isConfirmed ? "#10B981" : "#FFCB05";
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, width, height);

            // Corner markers
            const cornerSize = 15;
            ctx.lineWidth = 4;

            // Top-left
            ctx.beginPath();
            ctx.moveTo(x1, y1 + cornerSize);
            ctx.lineTo(x1, y1);
            ctx.lineTo(x1 + cornerSize, y1);
            ctx.stroke();

            // Top-right
            ctx.beginPath();
            ctx.moveTo(x2 - cornerSize, y1);
            ctx.lineTo(x2, y1);
            ctx.lineTo(x2, y1 + cornerSize);
            ctx.stroke();

            // Bottom-left
            ctx.beginPath();
            ctx.moveTo(x1, y2 - cornerSize);
            ctx.lineTo(x1, y2);
            ctx.lineTo(x1 + cornerSize, y2);
            ctx.stroke();

            // Bottom-right
            ctx.beginPath();
            ctx.moveTo(x2 - cornerSize, y2);
            ctx.lineTo(x2, y2);
            ctx.lineTo(x2, y2 - cornerSize);
            ctx.stroke();

            // Label background
            const label = det.price !== null
                ? `${det.name} - $${det.price.toFixed(2)}`
                : det.name;
            ctx.font = "bold 16px Inter, sans-serif";
            const textWidth = ctx.measureText(label).width;

            ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
            ctx.fillRect(x1, y1 - 28, textWidth + 16, 24);

            // Label text
            ctx.fillStyle = isConfirmed ? "#10B981" : "#FFCB05";
            ctx.fillText(label, x1 + 8, y1 - 10);
        });
    }, [detections]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stopCamera();
        };
    }, []);

    // Clear session
    const clearSession = () => {
        setSessionCards(new Map());
        setSessionTotal(0);
    };

    return (
        <div className="min-h-screen bg-background flex flex-col">
            {/* Top Bar */}
            <header className="h-16 border-b border-border flex items-center justify-between px-4 bg-card">
                <div className="flex items-center gap-4">
                    <Button variant="ghost" size="icon" asChild>
                        <Link href="/dashboard">
                            <ArrowLeft className="w-5 h-5" />
                        </Link>
                    </Button>
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg gradient-primary flex items-center justify-center">
                            <Scan className="w-4 h-4 text-white" />
                        </div>
                        <span className="font-semibold">PriceLens Scanner</span>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    <Badge variant="outline" className="font-mono">
                        {fps} FPS
                    </Badge>
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => setIsMuted(!isMuted)}
                    >
                        {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
                    </Button>
                </div>
            </header>

            {/* Main Content */}
            <main className="flex-1 flex flex-col lg:flex-row gap-4 p-4">
                {/* Camera View */}
                <div className="flex-1 relative">
                    <div className="relative aspect-video bg-slate-900 rounded-2xl overflow-hidden">
                        <video
                            ref={videoRef}
                            className="absolute inset-0 w-full h-full object-cover"
                            playsInline
                            muted
                        />
                        <canvas
                            ref={canvasRef}
                            className="absolute inset-0 w-full h-full"
                        />

                        {/* Overlay when not streaming */}
                        {!isStreaming && (
                            <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/90">
                                <Camera className="w-16 h-16 text-muted-foreground mb-4" />
                                <p className="text-muted-foreground mb-2">Camera Preview</p>
                                {error && (
                                    <p className="text-red-400 text-sm mb-4">{error}</p>
                                )}
                                <Button onClick={startCamera} className="gradient-primary">
                                    <Camera className="w-4 h-4 mr-2" />
                                    Start Camera
                                </Button>
                            </div>
                        )}

                        {/* FPS Counter Overlay */}
                        {isStreaming && (
                            <div className="absolute top-4 right-4 bg-black/60 rounded-lg px-3 py-1.5">
                                <p className="text-sm font-mono text-green-400">{fps} FPS</p>
                            </div>
                        )}
                    </div>

                    {/* Camera Controls */}
                    <div className="flex gap-4 mt-4 justify-center">
                        {isStreaming ? (
                            <Button onClick={stopCamera} variant="destructive" size="lg">
                                <CameraOff className="w-5 h-5 mr-2" />
                                Stop Camera
                            </Button>
                        ) : (
                            <Button onClick={startCamera} size="lg" className="gradient-primary">
                                <Camera className="w-5 h-5 mr-2" />
                                Start Camera
                            </Button>
                        )}
                    </div>
                </div>

                {/* Session Panel */}
                <div className="lg:w-80">
                    <Card className="bg-card border-border h-full">
                        <CardContent className="p-6">
                            <h2 className="font-semibold mb-4 flex items-center gap-2">
                                <DollarSign className="w-5 h-5 text-green-400" />
                                Session Total
                            </h2>

                            <div className="text-center py-6 border-b border-border mb-4">
                                <p className="text-4xl font-bold text-green-400">
                                    ${sessionTotal.toFixed(2)}
                                </p>
                                <p className="text-sm text-muted-foreground mt-1">
                                    {sessionCards.size} cards scanned
                                </p>
                            </div>

                            {/* Scanned Cards List */}
                            <div className="space-y-3 max-h-64 overflow-y-auto mb-4">
                                {Array.from(sessionCards.values()).map((card) => (
                                    <div
                                        key={card.card_id}
                                        className="flex items-center justify-between p-3 bg-muted/50 rounded-lg"
                                    >
                                        <div>
                                            <p className="font-medium">{card.name}</p>
                                            <p className="text-xs text-muted-foreground">{card.set}</p>
                                        </div>
                                        <p className="font-semibold text-green-400">
                                            ${(card.price || 0).toFixed(2)}
                                        </p>
                                    </div>
                                ))}

                                {sessionCards.size === 0 && (
                                    <p className="text-center text-muted-foreground py-8">
                                        No cards scanned yet.<br />
                                        Point camera at a Pokemon card.
                                    </p>
                                )}
                            </div>

                            <Button
                                variant="outline"
                                className="w-full"
                                onClick={clearSession}
                            >
                                <RotateCcw className="w-4 h-4 mr-2" />
                                Clear Session
                            </Button>
                        </CardContent>
                    </Card>
                </div>
            </main>
        </div>
    );
}
