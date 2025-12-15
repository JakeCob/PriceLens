"use client";

import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import {
    Zap,
    Target,
    TrendingUp,
    Database,
    Shield,
    Smartphone
} from "lucide-react";

const features = [
    {
        icon: Zap,
        title: "Lightning Fast Detection",
        description: "YOLO11-powered detection runs at 30+ FPS with less than 100ms latency. See prices before you even finish looking at the card.",
        color: "text-primary",
        bgColor: "bg-primary/10",
    },
    {
        icon: Target,
        title: "95% Accurate Recognition",
        description: "Our AI model is trained on thousands of Pokemon cards and can recognize cards even at angles, in poor lighting, or partially visible.",
        color: "text-accent",
        bgColor: "bg-accent/10",
    },
    {
        icon: TrendingUp,
        title: "Real-Time Market Prices",
        description: "Pull prices from TCGPlayer, CardMarket, and other sources. Get low, mid, high, and market prices instantly.",
        color: "text-green-500",
        bgColor: "bg-green-500/10",
    },
    {
        icon: Database,
        title: "10,000+ Card Database",
        description: "Every Pokemon TCG set from Base Set to the latest releases. We're constantly adding new cards as they're released.",
        color: "text-secondary",
        bgColor: "bg-secondary/10",
    },
    {
        icon: Shield,
        title: "Privacy First",
        description: "All detection happens in real-time on our secure servers. We never store your camera feed or card images.",
        color: "text-blue-400",
        bgColor: "bg-blue-400/10",
    },
    {
        icon: Smartphone,
        title: "Works Everywhere",
        description: "Use on desktop with webcam or on mobile with your phone camera. No app download required - just open your browser.",
        color: "text-pink-500",
        bgColor: "bg-pink-500/10",
    },
];

const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
        opacity: 1,
        transition: {
            staggerChildren: 0.1,
        },
    },
};

const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
        opacity: 1,
        y: 0,
        transition: { duration: 0.5 },
    },
};

export function FeaturesSection() {
    return (
        <section className="py-24 relative overflow-hidden">
            {/* Background */}
            <div className="absolute inset-0 -z-10">
                <div className="absolute top-0 left-1/4 w-72 h-72 bg-primary/5 rounded-full blur-3xl" />
                <div className="absolute bottom-0 right-1/4 w-72 h-72 bg-accent/5 rounded-full blur-3xl" />
            </div>

            <div className="container mx-auto px-4">
                {/* Section Header */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="text-center mb-16"
                >
                    <h2 className="text-3xl md:text-4xl font-bold mb-4">
                        Everything You Need to{" "}
                        <span className="text-gradient">Price Your Cards</span>
                    </h2>
                    <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                        Powerful features designed to make card pricing fast, accurate, and effortless.
                    </p>
                </motion.div>

                {/* Features Grid */}
                <motion.div
                    variants={containerVariants}
                    initial="hidden"
                    whileInView="visible"
                    viewport={{ once: true }}
                    className="grid md:grid-cols-2 lg:grid-cols-3 gap-6"
                >
                    {features.map((feature) => (
                        <motion.div key={feature.title} variants={itemVariants}>
                            <Card className="h-full bg-card/50 border-border/50 hover:border-primary/30 transition-colors group">
                                <CardContent className="p-6">
                                    <div className={`w-12 h-12 rounded-xl ${feature.bgColor} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                                        <feature.icon className={`w-6 h-6 ${feature.color}`} />
                                    </div>
                                    <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                                    <p className="text-muted-foreground">{feature.description}</p>
                                </CardContent>
                            </Card>
                        </motion.div>
                    ))}
                </motion.div>
            </div>
        </section>
    );
}
