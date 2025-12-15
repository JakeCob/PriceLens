import { Header, Footer } from "@/components/layout";
import { Metadata } from "next";
import {
    Scan,
    Zap,
    Target,
    Users,
    Github,
    Linkedin,
    Twitter,
    Lightbulb,
    Rocket,
    Heart
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";

export const metadata: Metadata = {
    title: "About",
    description: "Learn about PriceLens - the story behind our AI-powered Pokemon card price scanner.",
};

const timeline = [
    {
        icon: Lightbulb,
        title: "The Idea",
        description: "Frustrated with manually looking up card prices at tournaments, we envisioned a faster way.",
    },
    {
        icon: Zap,
        title: "Building the Tech",
        description: "Trained YOLO11 on thousands of Pokemon cards to achieve 95%+ detection accuracy.",
    },
    {
        icon: Rocket,
        title: "Launch",
        description: "Released PriceLens to help collectors worldwide get instant, accurate prices.",
    },
];

const values = [
    {
        icon: Target,
        title: "Accuracy First",
        description: "We obsess over detection accuracy because wrong prices are worse than no prices.",
    },
    {
        icon: Zap,
        title: "Speed Matters",
        description: "Real-time means real-time. We optimize for milliseconds, not seconds.",
    },
    {
        icon: Heart,
        title: "For Collectors",
        description: "Built by Pokemon collectors who understand what the community needs.",
    },
    {
        icon: Users,
        title: "Community Driven",
        description: "Your feedback shapes our roadmap. We build what you ask for.",
    },
];

export default function AboutPage() {
    return (
        <div className="min-h-screen flex flex-col">
            <Header />
            <main className="flex-1 pt-24">
                {/* Hero Section */}
                <section className="py-16 relative overflow-hidden">
                    <div className="absolute inset-0 -z-10">
                        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl" />
                        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/10 rounded-full blur-3xl" />
                    </div>

                    <div className="container mx-auto px-4 text-center">
                        <div className="w-20 h-20 rounded-2xl gradient-primary flex items-center justify-center mx-auto mb-8">
                            <Scan className="w-10 h-10 text-white" />
                        </div>
                        <h1 className="text-4xl md:text-5xl font-bold mb-6">
                            About <span className="text-gradient">PriceLens</span>
                        </h1>
                        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                            We&apos;re on a mission to make Pokemon card pricing fast, accurate, and accessible to every collector.
                        </p>
                    </div>
                </section>

                {/* Story Section */}
                <section className="py-16 bg-card/30">
                    <div className="container mx-auto px-4">
                        <div className="max-w-3xl mx-auto">
                            <h2 className="text-3xl font-bold mb-8 text-center">Our Story</h2>
                            <div className="prose prose-invert max-w-none">
                                <p className="text-lg text-muted-foreground mb-6">
                                    It started at a local Pokemon tournament. Between rounds, I watched collectors
                                    frantically typing card names into their phones, waiting for slow websites to load,
                                    trying to negotiate trades with outdated price info.
                                </p>
                                <p className="text-lg text-muted-foreground mb-6">
                                    There had to be a better way. What if you could just point your camera at a card
                                    and instantly see its market value? No typing, no waiting, no guessing.
                                </p>
                                <p className="text-lg text-muted-foreground">
                                    That question led to PriceLens — an AI-powered scanner that identifies Pokemon cards
                                    in real-time and overlays live market prices directly on your camera feed. Built with
                                    YOLO11 computer vision and trained on thousands of cards, it achieves 95%+ accuracy
                                    at 30+ frames per second.
                                </p>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Timeline */}
                <section className="py-16">
                    <div className="container mx-auto px-4">
                        <h2 className="text-3xl font-bold mb-12 text-center">How We Got Here</h2>
                        <div className="grid md:grid-cols-3 gap-8">
                            {timeline.map((item, index) => (
                                <div key={item.title} className="text-center">
                                    <div className="w-16 h-16 rounded-2xl gradient-primary flex items-center justify-center mx-auto mb-4">
                                        <item.icon className="w-8 h-8 text-white" />
                                    </div>
                                    <div className="text-sm text-primary font-medium mb-2">Step {index + 1}</div>
                                    <h3 className="text-xl font-semibold mb-2">{item.title}</h3>
                                    <p className="text-muted-foreground">{item.description}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* Values */}
                <section className="py-16 bg-card/30">
                    <div className="container mx-auto px-4">
                        <h2 className="text-3xl font-bold mb-12 text-center">What We Believe</h2>
                        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                            {values.map((value) => (
                                <Card key={value.title} className="bg-card/50 border-border/50">
                                    <CardContent className="p-6 text-center">
                                        <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-4">
                                            <value.icon className="w-6 h-6 text-primary" />
                                        </div>
                                        <h3 className="font-semibold mb-2">{value.title}</h3>
                                        <p className="text-sm text-muted-foreground">{value.description}</p>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    </div>
                </section>

                {/* Team / Creator */}
                <section className="py-16">
                    <div className="container mx-auto px-4">
                        <h2 className="text-3xl font-bold mb-12 text-center">The Team</h2>
                        <div className="max-w-md mx-auto text-center">
                            <div className="w-24 h-24 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center mx-auto mb-4 text-3xl font-bold text-white">
                                JC
                            </div>
                            <h3 className="text-xl font-semibold mb-1">Jake Cob</h3>
                            <p className="text-muted-foreground mb-4">Creator & Developer</p>
                            <p className="text-sm text-muted-foreground mb-6">
                                Full-stack developer and Pokemon collector. Building tools to make the TCG community better.
                            </p>
                            <div className="flex justify-center gap-4">
                                <Button variant="outline" size="icon" asChild>
                                    <a href="https://github.com/JakeCob" target="_blank" rel="noopener noreferrer">
                                        <Github className="w-4 h-4" />
                                    </a>
                                </Button>
                                <Button variant="outline" size="icon" asChild>
                                    <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer">
                                        <Linkedin className="w-4 h-4" />
                                    </a>
                                </Button>
                                <Button variant="outline" size="icon" asChild>
                                    <a href="https://twitter.com" target="_blank" rel="noopener noreferrer">
                                        <Twitter className="w-4 h-4" />
                                    </a>
                                </Button>
                            </div>
                        </div>
                    </div>
                </section>

                {/* CTA */}
                <section className="py-16 bg-card/30">
                    <div className="container mx-auto px-4 text-center">
                        <h2 className="text-3xl font-bold mb-4">Ready to Try PriceLens?</h2>
                        <p className="text-muted-foreground mb-8">
                            Join thousands of collectors getting instant card prices.
                        </p>
                        <Button size="lg" className="gradient-primary" asChild>
                            <Link href="/dashboard">Start Scanning Free</Link>
                        </Button>
                    </div>
                </section>
            </main>
            <Footer />
        </div>
    );
}
