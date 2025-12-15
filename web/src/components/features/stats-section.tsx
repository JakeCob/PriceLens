"use client";

import { motion } from "framer-motion";

const stats = [
    { value: "95%", label: "Detection Accuracy", sublabel: "Across all card sets" },
    { value: "10K+", label: "Cards in Database", sublabel: "From Base Set to present" },
    { value: "<100ms", label: "Response Time", sublabel: "Near-instant results" },
    { value: "30+", label: "Price Sources", sublabel: "Real-time aggregation" },
];

export function StatsSection() {
    return (
        <section className="py-20 relative overflow-hidden">
            {/* Background Gradient */}
            <div className="absolute inset-0 gradient-primary opacity-10" />

            <div className="container mx-auto px-4 relative z-10">
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
                    {stats.map((stat, index) => (
                        <motion.div
                            key={stat.label}
                            initial={{ opacity: 0, scale: 0.8 }}
                            whileInView={{ opacity: 1, scale: 1 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.1 }}
                            className="text-center"
                        >
                            <p className="text-4xl md:text-5xl font-bold text-gradient mb-2">
                                {stat.value}
                            </p>
                            <p className="text-lg font-medium mb-1">{stat.label}</p>
                            <p className="text-sm text-muted-foreground">{stat.sublabel}</p>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
}
