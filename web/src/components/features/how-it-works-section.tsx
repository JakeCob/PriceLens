"use client";

import { motion } from "framer-motion";
import { Scan, Sparkles, DollarSign } from "lucide-react";

const steps = [
    {
        number: "01",
        icon: Scan,
        title: "Point Your Camera",
        description: "Open PriceLens and allow camera access. Hold any Pokemon card in view of your webcam or phone camera.",
        color: "from-primary to-primary/50",
    },
    {
        number: "02",
        icon: Sparkles,
        title: "AI Detects Your Card",
        description: "Our YOLO11 model instantly detects and identifies the card, even at angles or in varying lighting conditions.",
        color: "from-accent to-accent/50",
    },
    {
        number: "03",
        icon: DollarSign,
        title: "Get Instant Prices",
        description: "See real-time market prices from multiple sources overlaid directly on your camera view. Add cards to your collection with one tap.",
        color: "from-green-500 to-green-500/50",
    },
];

export function HowItWorksSection() {
    return (
        <section className="py-24 bg-card/30">
            <div className="container mx-auto px-4">
                {/* Section Header */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="text-center mb-16"
                >
                    <h2 className="text-3xl md:text-4xl font-bold mb-4">
                        How It Works
                    </h2>
                    <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                        Getting prices for your cards takes just seconds. Here&apos;s how simple it is.
                    </p>
                </motion.div>

                {/* Steps */}
                <div className="grid md:grid-cols-3 gap-8 relative">
                    {/* Connection Line (Desktop) */}
                    <div className="hidden md:block absolute top-24 left-1/6 right-1/6 h-0.5 bg-gradient-to-r from-primary via-accent to-green-500" />

                    {steps.map((step, index) => (
                        <motion.div
                            key={step.number}
                            initial={{ opacity: 0, y: 30 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.2 }}
                            className="relative text-center"
                        >
                            {/* Step Number Circle */}
                            <div className="relative z-10 mx-auto mb-6">
                                <div className={`w-20 h-20 rounded-2xl bg-gradient-to-br ${step.color} flex items-center justify-center mx-auto shadow-lg`}>
                                    <step.icon className="w-10 h-10 text-white" />
                                </div>
                                <div className="absolute -top-2 -right-2 w-8 h-8 rounded-full bg-background border-2 border-primary flex items-center justify-center">
                                    <span className="text-xs font-bold">{step.number}</span>
                                </div>
                            </div>

                            {/* Content */}
                            <h3 className="text-xl font-semibold mb-3">{step.title}</h3>
                            <p className="text-muted-foreground max-w-xs mx-auto">
                                {step.description}
                            </p>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
}
