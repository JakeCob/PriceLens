"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
    Play,
    ArrowRight,
    Zap,
    Target,
    TrendingUp,
    Sparkles
} from "lucide-react";

export function HeroSection() {
    return (
        <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-16">
            {/* Background Effects */}
            <div className="absolute inset-0 -z-10">
                {/* Gradient Orbs */}
                <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-3xl animate-pulse-slow" />
                <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/20 rounded-full blur-3xl animate-pulse-slow" />
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-secondary/10 rounded-full blur-3xl" />

                {/* Grid Pattern */}
                <div
                    className="absolute inset-0 opacity-[0.03]"
                    style={{
                        backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
                        backgroundSize: "50px 50px",
                    }}
                />
            </div>

            <div className="container mx-auto px-4 py-20">
                <div className="grid lg:grid-cols-2 gap-12 items-center">
                    {/* Left Content */}
                    <motion.div
                        initial={{ opacity: 0, x: -50 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.6 }}
                        className="text-center lg:text-left"
                    >
                        {/* Badge */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2 }}
                        >
                            <Badge
                                variant="outline"
                                className="mb-6 px-4 py-2 text-sm border-primary/30 bg-primary/10"
                            >
                                <Sparkles className="w-4 h-4 mr-2 text-secondary" />
                                Powered by YOLO11 AI
                            </Badge>
                        </motion.div>

                        {/* Headline */}
                        <motion.h1
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.3 }}
                            className="text-4xl md:text-5xl lg:text-6xl font-bold leading-tight mb-6"
                        >
                            See Pokemon Card Prices{" "}
                            <span className="text-gradient">in Real-Time</span>
                        </motion.h1>

                        {/* Subheading */}
                        <motion.p
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.4 }}
                            className="text-lg md:text-xl text-muted-foreground mb-8 max-w-xl mx-auto lg:mx-0"
                        >
                            Point your camera at any card and get instant market prices
                            with AI-powered detection. Built for collectors, by collectors.
                        </motion.p>

                        {/* CTA Buttons */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.5 }}
                            className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start"
                        >
                            <Button
                                size="lg"
                                className="gradient-primary text-lg px-8 py-6 group"
                                asChild
                            >
                                <Link href="/dashboard">
                                    Try Demo Free
                                    <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
                                </Link>
                            </Button>
                            <Button
                                size="lg"
                                variant="outline"
                                className="text-lg px-8 py-6 border-muted-foreground/30"
                                asChild
                            >
                                <Link href="#demo">
                                    <Play className="mr-2 w-5 h-5" />
                                    Watch Video
                                </Link>
                            </Button>
                        </motion.div>

                        {/* Stats */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.6 }}
                            className="flex flex-wrap gap-6 mt-10 justify-center lg:justify-start"
                        >
                            <div className="flex items-center gap-2">
                                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                                    <Zap className="w-5 h-5 text-primary" />
                                </div>
                                <div>
                                    <p className="text-2xl font-bold">30+ FPS</p>
                                    <p className="text-xs text-muted-foreground">Real-time</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center">
                                    <Target className="w-5 h-5 text-accent" />
                                </div>
                                <div>
                                    <p className="text-2xl font-bold">95%</p>
                                    <p className="text-xs text-muted-foreground">Accuracy</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-10 h-10 rounded-lg bg-green-500/10 flex items-center justify-center">
                                    <TrendingUp className="w-5 h-5 text-green-500" />
                                </div>
                                <div>
                                    <p className="text-2xl font-bold">10K+</p>
                                    <p className="text-xs text-muted-foreground">Cards</p>
                                </div>
                            </div>
                        </motion.div>
                    </motion.div>

                    {/* Right Content - Hero Visual */}
                    <motion.div
                        initial={{ opacity: 0, x: 50 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.6, delay: 0.2 }}
                        className="relative"
                    >
                        <div className="relative aspect-square max-w-lg mx-auto">
                            {/* Camera Viewfinder Mock */}
                            <div className="absolute inset-0 rounded-3xl bg-card border border-border overflow-hidden shadow-2xl">
                                {/* Viewfinder Content */}
                                <div className="absolute inset-4 rounded-2xl bg-gradient-to-br from-slate-900 to-slate-800 overflow-hidden">
                                    {/* Detection Box */}
                                    <motion.div
                                        initial={{ scale: 0.8, opacity: 0 }}
                                        animate={{ scale: 1, opacity: 1 }}
                                        transition={{ delay: 0.8, duration: 0.5 }}
                                        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-64"
                                    >
                                        {/* Detection Border */}
                                        <div className="absolute inset-0 border-2 border-primary rounded-lg animate-pulse">
                                            {/* Corner Markers */}
                                            <div className="absolute -top-1 -left-1 w-4 h-4 border-l-4 border-t-4 border-primary rounded-tl" />
                                            <div className="absolute -top-1 -right-1 w-4 h-4 border-r-4 border-t-4 border-primary rounded-tr" />
                                            <div className="absolute -bottom-1 -left-1 w-4 h-4 border-l-4 border-b-4 border-primary rounded-bl" />
                                            <div className="absolute -bottom-1 -right-1 w-4 h-4 border-r-4 border-b-4 border-primary rounded-br" />
                                        </div>

                                        {/* Card Image */}
                                        <img
                                            src="/images/sample-card.jpg"
                                            alt="Charizard Pokemon Card"
                                            className="absolute inset-2 rounded object-cover"
                                        />
                                    </motion.div>

                                    {/* Price Overlay */}
                                    <motion.div
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: 1.2 }}
                                        className="absolute bottom-4 left-4 right-4 glass rounded-xl p-3"
                                    >
                                        <div className="flex justify-between items-center">
                                            <div>
                                                <p className="text-xs text-muted-foreground">Detected</p>
                                                <p className="font-semibold">Charizard</p>
                                            </div>
                                            <div className="text-right">
                                                <p className="text-xs text-muted-foreground">Market Price</p>
                                                <p className="text-xl font-bold text-green-400">$199.99</p>
                                            </div>
                                        </div>
                                    </motion.div>

                                    {/* FPS Counter */}
                                    <motion.div
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        transition={{ delay: 1 }}
                                        className="absolute top-4 right-4 bg-black/50 rounded-lg px-2 py-1"
                                    >
                                        <p className="text-xs font-mono text-green-400">32 FPS</p>
                                    </motion.div>
                                </div>
                            </div>

                            {/* Floating Elements */}
                            <motion.div
                                animate={{ y: [0, -10, 0] }}
                                transition={{ duration: 3, repeat: Infinity }}
                                className="absolute -top-4 -right-4 w-20 h-20 rounded-xl gradient-primary flex items-center justify-center shadow-lg"
                            >
                                <TrendingUp className="w-10 h-10 text-white" />
                            </motion.div>

                            <motion.div
                                animate={{ y: [0, 10, 0] }}
                                transition={{ duration: 3, repeat: Infinity, delay: 0.5 }}
                                className="absolute -bottom-4 -left-4 w-16 h-16 rounded-xl bg-secondary flex items-center justify-center shadow-lg"
                            >
                                <Target className="w-8 h-8 text-black" />
                            </motion.div>
                        </div>
                    </motion.div>
                </div>
            </div>
        </section>
    );
}
