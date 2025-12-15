"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { ArrowRight, Scan } from "lucide-react";

export function CTASection() {
    return (
        <section className="py-24 relative overflow-hidden">
            {/* Background */}
            <div className="absolute inset-0 -z-10">
                <div className="absolute inset-0 gradient-primary opacity-5" />
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-primary/10 rounded-full blur-3xl" />
            </div>

            <div className="container mx-auto px-4">
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="max-w-3xl mx-auto text-center"
                >
                    {/* Icon */}
                    <motion.div
                        initial={{ scale: 0 }}
                        whileInView={{ scale: 1 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.2, type: "spring" }}
                        className="w-20 h-20 rounded-2xl gradient-primary flex items-center justify-center mx-auto mb-8"
                    >
                        <Scan className="w-10 h-10 text-white" />
                    </motion.div>

                    {/* Heading */}
                    <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold mb-6">
                        Ready to Start Scanning?
                    </h2>

                    <p className="text-lg text-muted-foreground mb-8 max-w-xl mx-auto">
                        Join thousands of collectors using PriceLens to get instant, accurate
                        prices for their Pokemon cards. It&apos;s free to get started.
                    </p>

                    {/* CTA Buttons */}
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <Button
                            size="lg"
                            className="gradient-primary text-lg px-8 py-6 group"
                            asChild
                        >
                            <Link href="/dashboard">
                                Start Scanning Free
                                <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
                            </Link>
                        </Button>
                        <Button
                            size="lg"
                            variant="outline"
                            className="text-lg px-8 py-6"
                            asChild
                        >
                            <Link href="/pricing">
                                View Pricing
                            </Link>
                        </Button>
                    </div>

                    {/* Trust Badge */}
                    <p className="text-sm text-muted-foreground mt-8">
                        ✨ No credit card required • Free tier available • Cancel anytime
                    </p>
                </motion.div>
            </div>
        </section>
    );
}
