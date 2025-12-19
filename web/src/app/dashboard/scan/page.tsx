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
    DollarSign,
    FlipHorizontal
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

type CurrencyCode = "USD" | "PHP";

// Display-only currency conversion (backend prices are USD).
// If you want live FX rates later, we can move this to an API endpoint.
const CURRENCY_META: Record<CurrencyCode, { label: string; locale: string; usdRate: number }> = {
    USD: { label: "USD ($)", locale: "en-US", usdRate: 1.0 },
    PHP: { label: "PHP (₱)", locale: "en-PH", usdRate: 56.0 }, // fixed rate; adjust as needed
};

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

    const isStreamingRef = useRef(false);
    const processingCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const scanningStartRef = useRef<number | null>(null);

    // Currency picker (display-only; backend prices are USD)
    const [currency, setCurrency] = useState<CurrencyCode>("USD");
    useEffect(() => {
        try {
            const saved = typeof window !== "undefined" ? window.localStorage.getItem("pricelens.currency") : null;
            if (saved === "USD" || saved === "PHP") {
                setCurrency(saved);
            }
        } catch { }
    }, []);

    useEffect(() => {
        try {
            if (typeof window !== "undefined") {
                window.localStorage.setItem("pricelens.currency", currency);
            }
        } catch { }
    }, [currency]);

    const formatPrice = useCallback((usdAmount: number) => {
        const meta = CURRENCY_META[currency];
        const converted = usdAmount * meta.usdRate;
        return new Intl.NumberFormat(meta.locale, {
            style: "currency",
            currency,
            maximumFractionDigits: 2,
        }).format(converted);
    }, [currency]);

    // Mirror State
    const [isMirrored, setIsMirrored] = useState(false);
    const isMirroredRef = useRef(false);

    const toggleMirror = () => {
        const newState = !isMirrored;
        setIsMirrored(newState);
        isMirroredRef.current = newState;
    };

    // Initial setup for processing canvas
    useEffect(() => {
        if (typeof window !== "undefined") {
            const canvas = document.createElement('canvas');
            canvas.width = 640;
            // height will be set dynamically based on aspect ratio
            processingCanvasRef.current = canvas;
        }
    }, []);

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
                            isStreamingRef.current = true;
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
        isStreamingRef.current = false;
        setDetections([]);
    };

    const [debugLog, setDebugLog] = useState<string[]>([]);

    const log = (msg: string) => {
        setDebugLog(prev => [msg, ...prev].slice(0, 5));
        console.log(msg);
    };

    const sessionIdsRef = useRef<Set<string>>(new Set());
    const sessionPricesRef = useRef<Map<string, number | null>>(new Map());

    // Send frame to backend
    const sendFrame = async () => {
        if (!videoRef.current || !processingCanvasRef.current || !isStreamingRef.current) {
            return;
        }

        const video = videoRef.current;
        const processCanvas = processingCanvasRef.current;
        const pCtx = processCanvas.getContext("2d");
        if (!pCtx) return;

        // Use SPEED RESOLUTION (500px)
        // 800px caused 1 FPS on CPU.
        // 500px provides ~2.5x speed boost while maintaining sufficient detail for ORB.
        const targetWidth = 500;
        const scale = targetWidth / video.videoWidth;
        const pWidth = targetWidth;
        const pHeight = video.videoHeight * scale;

        // Update processing canvas size
        if (processCanvas.width !== pWidth) {
            processCanvas.width = pWidth;
        }
        if (processCanvas.height !== pHeight) {
            processCanvas.height = pHeight;
        }

        // Draw video frame to processing canvas (Downscale)
        pCtx.drawImage(video, 0, 0, pWidth, pHeight);

        // Get blob from processing canvas
        return new Promise<void>((resolve) => {
            processCanvas.toBlob(async (blob) => {
                if (!blob) {
                    resolve();
                    return;
                }

                const formData = new FormData();
                formData.append("file", blob, "frame.jpg");

                try {
                    const response = await fetch(`${API_URL}/detect-live`, {
                        method: "POST",
                        body: formData,
                    });

                    if (response.ok) {
                        const data = await response.json();

                        // Check if streaming is still active before updating state
                        // This prevents "ghost" overlays if the user stopped the camera
                        // while the request was in flight.
                        if (!isStreamingRef.current) return;

                        // Scale coordinates back up to VIDEO size for display overlay
                        // The overlay canvas matches the video element size (e.g. 1280x720)
                        const displayScale = video.videoWidth / pWidth;
                        const videoW = video.videoWidth;

                        // UX Fix for "Taking too long": Global timer for "Scanning..." persistence
                        const isScanningAny = (data.detections || []).some((d: Detection) => d.name === "Scanning...");

                        if (isScanningAny) {
                            if (!scanningStartRef.current) {
                                scanningStartRef.current = Date.now();
                            }
                        } else {
                            scanningStartRef.current = null;
                        }

                        const showUnknown = scanningStartRef.current && (Date.now() - scanningStartRef.current > 3000);

                        const scaledDetections = (data.detections || []).map((det: Detection) => {
                            // Scale raw coordinates
                            const sx1 = det.bbox[0] * displayScale;
                            const sy1 = det.bbox[1] * displayScale;
                            const sx2 = det.bbox[2] * displayScale;
                            const sy2 = det.bbox[3] * displayScale;

                            let adjustedName = det.name;

                            if (det.name === "Scanning..." && showUnknown) {
                                adjustedName = "Unknown Card";
                            }

                            if (isMirroredRef.current) {
                                // Flip X coordinates for mirrored view
                                return {
                                    ...det,
                                    name: adjustedName,
                                    bbox: [
                                        videoW - sx2,
                                        sy1,
                                        videoW - sx1,
                                        sy2
                                    ]
                                };
                            } else {
                                // Regular coordinates
                                return {
                                    ...det,
                                    name: adjustedName,
                                    bbox: [
                                        sx1,
                                        sy1,
                                        sx2,
                                        sy2
                                    ]
                                };
                            }
                        });

                        setDetections(scaledDetections);

                        // Update FPS (Detection Rate)
                        frameCountRef.current++;
                        const now = Date.now();
                        if (now - lastFpsTimeRef.current >= 1000) {
                            setFps(frameCountRef.current);
                            frameCountRef.current = 0;
                            lastFpsTimeRef.current = now;
                        }

                        // Add detected cards to session
                        scaledDetections.forEach((det: Detection) => {
                            // Add card if properly identified (not Scanning or Unknown)
                            // Price may be null while still loading - that's OK, show as $0.00
                            if (det.name !== "Scanning..." && det.name !== "Unknown Card" && det.card_id !== "unknown") {
                                // Check against Ref to avoid stale closure issues
                                if (!sessionIdsRef.current.has(det.card_id)) {
                                    sessionIdsRef.current.add(det.card_id);
                                    sessionPricesRef.current.set(det.card_id, det.price);

                                    setSessionCards(prev => {
                                        const newCards = new Map(prev);
                                        newCards.set(det.card_id, det);
                                        return newCards;
                                    });
                                    setSessionTotal(prev => prev + (det.price || 0));

                                    // Play sound (Success Chime)
                                    if (!isMuted) {
                                        // Base64 encoded short success beep to avoid 404s
                                        const successSound = "data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//uQZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWgAAAA0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//uQZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWgAAAA0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//uQZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWgAAAA0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//uQZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWgAAAA0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//uQZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWgAAAA0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//uQZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWgAAAA0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//uQZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWgAAAA0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//uQZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWgAAAA0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//uQZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWgAAAA0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//uQZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWgAAAA0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//uQZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWgAAAA0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//uQZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWgAAAA0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//uQZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWgAAAA0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
                                        // Note: The above is a placeholder empty MP3. Real beep below:
                                        const beep = "data:audio/wav;base64,UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU"; // Truncated generic

                                        // Using a simple oscillator beep since we can't easily embed a full mp3 string here without being huge.
                                        // Better approach: Use Web Audio API for a synthetic beep.
                                        try {
                                            const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
                                            if (AudioContext) {
                                                const ctx = new AudioContext();
                                                const osc = ctx.createOscillator();
                                                const gain = ctx.createGain();

                                                osc.connect(gain);
                                                gain.connect(ctx.destination);

                                                osc.type = "sine";
                                                osc.frequency.setValueAtTime(880, ctx.currentTime); // A5
                                                osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.1);

                                                gain.gain.setValueAtTime(0.1, ctx.currentTime);
                                                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.1);

                                                osc.start(ctx.currentTime);
                                                osc.stop(ctx.currentTime + 0.1);
                                            }
                                        } catch (e) {
                                            console.error("Audio error", e);
                                        }
                                    }
                                }
                                // Card is already in session: update price if it arrives later (or changes).
                                else {
                                    const prevPrice = sessionPricesRef.current.get(det.card_id);
                                    const nextPrice = det.price;
                                    if (nextPrice !== null && nextPrice !== undefined) {
                                        if (prevPrice === null || prevPrice === undefined) {
                                            sessionPricesRef.current.set(det.card_id, nextPrice);
                                            setSessionCards(prev => {
                                                const newCards = new Map(prev);
                                                const existing = newCards.get(det.card_id);
                                                newCards.set(det.card_id, existing ? { ...existing, ...det } : det);
                                                return newCards;
                                            });
                                            setSessionTotal(prev => prev + nextPrice);
                                        } else if (Math.abs(nextPrice - prevPrice) > 0.0001) {
                                            sessionPricesRef.current.set(det.card_id, nextPrice);
                                            setSessionCards(prev => {
                                                const newCards = new Map(prev);
                                                const existing = newCards.get(det.card_id);
                                                newCards.set(det.card_id, existing ? { ...existing, ...det } : det);
                                                return newCards;
                                            });
                                            setSessionTotal(prev => prev + (nextPrice - prevPrice));
                                        }
                                    }
                                }
                            }
                        });
                    } else {
                        // log(`Error: ${response.status} ${response.statusText}`);
                    }
                } catch (err) {
                    // log(`Fetch error: ${err}`);
                    console.error("Detection error:", err);
                } finally {
                    resolve();
                }
            }, "image/jpeg", 0.7); // Low quality for speed
        });
    };

    // Detection loop
    const startDetectionLoop = () => {
        // log("Starting detection loop");
        const loop = async () => {
            if (!videoRef.current || videoRef.current.paused || videoRef.current.ended || !isStreamingRef.current) {
                // log("Loop stop condition met");
                return;
            }
            await sendFrame();
            animationRef.current = requestAnimationFrame(loop);
        };
        loop();
    };

    // Draw detection boxes (Overlay)
    useEffect(() => {
        if (!canvasRef.current || !videoRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        // Match canvas size to video size for correct overlay projection
        if (canvas.width !== videoRef.current.videoWidth || canvas.height !== videoRef.current.videoHeight) {
            canvas.width = videoRef.current.videoWidth;
            canvas.height = videoRef.current.videoHeight;
        }

        // Clear canvas (Transparent overlay)
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Note: We DO NOT draw the video here. The video element is visible behind the canvas.

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
                ? `${det.name} - ${formatPrice(det.price)}`
                : det.name;
            ctx.font = "bold 16px Inter, sans-serif";
            const textWidth = ctx.measureText(label).width;

            ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
            ctx.fillRect(x1, y1 - 28, textWidth + 16, 24);

            // Label text
            ctx.fillStyle = isConfirmed ? "#10B981" : "#FFCB05";
            ctx.fillText(label, x1 + 8, y1 - 10);
        });
    }, [detections, formatPrice]);

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
        sessionIdsRef.current.clear();
        sessionPricesRef.current.clear();
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
                    <select
                        value={currency}
                        onChange={(e) => setCurrency(e.target.value === "PHP" ? "PHP" : "USD")}
                        className="h-9 rounded-md border border-border bg-background px-3 text-sm text-foreground"
                        title="Currency"
                    >
                        {Object.entries(CURRENCY_META).map(([code, meta]) => (
                            <option key={code} value={code}>
                                {meta.label}
                            </option>
                        ))}
                    </select>
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={toggleMirror}
                        title="Toggle Mirror Mode"
                    >
                        <FlipHorizontal className="w-5 h-5" />
                    </Button>
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
                            className={`absolute inset-0 w-full h-full object-cover transform ${isMirrored ? "scale-x-[-1]" : ""}`}
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
                                    {formatPrice(sessionTotal)}
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
                                            {card.price === null ? "N/A" : formatPrice(card.price)}
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
