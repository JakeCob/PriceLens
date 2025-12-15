import { Header, Footer } from "@/components/layout";
import { Metadata } from "next";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import {
    Book,
    Rocket,
    Camera,
    Settings,
    HelpCircle,
    Code,
    Zap,
    ChevronRight
} from "lucide-react";

export const metadata: Metadata = {
    title: "Documentation",
    description: "Get started with PriceLens. Learn how to scan cards, manage collections, and use all features.",
};

const docCategories = [
    {
        icon: Rocket,
        title: "Getting Started",
        description: "Quick start guide and setup instructions",
        links: [
            { title: "Quick Start Guide", href: "/docs/quick-start" },
            { title: "System Requirements", href: "/docs/requirements" },
            { title: "Camera Setup", href: "/docs/camera-setup" },
            { title: "First Scan Tutorial", href: "/docs/first-scan" },
        ],
    },
    {
        icon: Book,
        title: "User Guides",
        description: "Learn how to use all features",
        links: [
            { title: "How to Scan Cards", href: "/docs/scanning" },
            { title: "Understanding Prices", href: "/docs/price-data" },
            { title: "Collection Management", href: "/docs/collections" },
            { title: "Setting Up Alerts", href: "/docs/alerts" },
        ],
    },
    {
        icon: Camera,
        title: "Features",
        description: "Detailed feature documentation",
        links: [
            { title: "Real-Time Detection", href: "/docs/detection" },
            { title: "Multi-Card Scanning", href: "/docs/multi-card" },
            { title: "Price Sources", href: "/docs/price-sources" },
            { title: "Export Options", href: "/docs/export" },
        ],
    },
    {
        icon: Code,
        title: "API Reference",
        description: "For developers and integrations",
        links: [
            { title: "Authentication", href: "/docs/api/auth" },
            { title: "Card Detection API", href: "/docs/api/detection" },
            { title: "Price Lookup API", href: "/docs/api/prices" },
            { title: "Rate Limits", href: "/docs/api/rate-limits" },
        ],
    },
    {
        icon: Settings,
        title: "Troubleshooting",
        description: "Common issues and solutions",
        links: [
            { title: "Camera Not Working", href: "/docs/troubleshoot/camera" },
            { title: "Cards Not Detected", href: "/docs/troubleshoot/detection" },
            { title: "Price Data Issues", href: "/docs/troubleshoot/prices" },
            { title: "Performance Tips", href: "/docs/troubleshoot/performance" },
        ],
    },
    {
        icon: HelpCircle,
        title: "FAQ",
        description: "Frequently asked questions",
        links: [
            { title: "General Questions", href: "/docs/faq/general" },
            { title: "Accuracy & Detection", href: "/docs/faq/accuracy" },
            { title: "Pricing & Billing", href: "/docs/faq/billing" },
            { title: "Privacy & Security", href: "/docs/faq/privacy" },
        ],
    },
];

export default function DocsPage() {
    return (
        <div className="min-h-screen flex flex-col">
            <Header />
            <main className="flex-1 pt-24">
                {/* Hero */}
                <section className="py-16 relative overflow-hidden">
                    <div className="absolute inset-0 -z-10">
                        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl" />
                    </div>

                    <div className="container mx-auto px-4 text-center">
                        <Badge variant="outline" className="mb-6 px-4 py-2 border-primary/30 bg-primary/10">
                            <Book className="w-4 h-4 mr-2" />
                            Documentation
                        </Badge>
                        <h1 className="text-4xl md:text-5xl font-bold mb-6">
                            How Can We{" "}
                            <span className="text-gradient">Help?</span>
                        </h1>
                        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                            Everything you need to get started with PriceLens and make the most of all features.
                        </p>
                    </div>
                </section>

                {/* Quick Links */}
                <section className="py-8">
                    <div className="container mx-auto px-4">
                        <div className="flex flex-wrap justify-center gap-4 mb-12">
                            <Badge variant="secondary" className="px-4 py-2 cursor-pointer hover:bg-primary/20 transition-colors">
                                <Rocket className="w-4 h-4 mr-2" /> Quick Start
                            </Badge>
                            <Badge variant="secondary" className="px-4 py-2 cursor-pointer hover:bg-primary/20 transition-colors">
                                <Camera className="w-4 h-4 mr-2" /> Camera Setup
                            </Badge>
                            <Badge variant="secondary" className="px-4 py-2 cursor-pointer hover:bg-primary/20 transition-colors">
                                <Zap className="w-4 h-4 mr-2" /> Scanning Guide
                            </Badge>
                            <Badge variant="secondary" className="px-4 py-2 cursor-pointer hover:bg-primary/20 transition-colors">
                                <Code className="w-4 h-4 mr-2" /> API Docs
                            </Badge>
                        </div>
                    </div>
                </section>

                {/* Doc Categories */}
                <section className="py-8">
                    <div className="container mx-auto px-4">
                        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {docCategories.map((category) => (
                                <Card key={category.title} className="bg-card/50 border-border/50 hover:border-primary/30 transition-colors">
                                    <CardContent className="p-6">
                                        <div className="flex items-center gap-3 mb-4">
                                            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                                                <category.icon className="w-5 h-5 text-primary" />
                                            </div>
                                            <div>
                                                <h3 className="font-semibold">{category.title}</h3>
                                                <p className="text-sm text-muted-foreground">{category.description}</p>
                                            </div>
                                        </div>
                                        <ul className="space-y-2">
                                            {category.links.map((link) => (
                                                <li key={link.href}>
                                                    <Link
                                                        href={link.href}
                                                        className="flex items-center justify-between text-sm text-muted-foreground hover:text-foreground transition-colors group"
                                                    >
                                                        {link.title}
                                                        <ChevronRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                                                    </Link>
                                                </li>
                                            ))}
                                        </ul>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    </div>
                </section>

                {/* Quick Start Card */}
                <section className="py-16 bg-card/30">
                    <div className="container mx-auto px-4">
                        <Card className="max-w-3xl mx-auto bg-gradient-to-br from-primary/10 to-accent/10 border-primary/20">
                            <CardContent className="p-8">
                                <div className="flex items-start gap-4">
                                    <div className="w-12 h-12 rounded-xl gradient-primary flex items-center justify-center flex-shrink-0">
                                        <Rocket className="w-6 h-6 text-white" />
                                    </div>
                                    <div>
                                        <h3 className="text-xl font-semibold mb-2">New to PriceLens?</h3>
                                        <p className="text-muted-foreground mb-4">
                                            Get up and running in 5 minutes with our quick start guide. Learn how to set up your camera,
                                            scan your first card, and start building your collection.
                                        </p>
                                        <Link
                                            href="/docs/quick-start"
                                            className="inline-flex items-center text-primary hover:underline font-medium"
                                        >
                                            Start the Quick Start Guide
                                            <ChevronRight className="w-4 h-4 ml-1" />
                                        </Link>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </section>

                {/* Contact */}
                <section className="py-16">
                    <div className="container mx-auto px-4 text-center">
                        <h2 className="text-2xl font-bold mb-4">Can&apos;t find what you&apos;re looking for?</h2>
                        <p className="text-muted-foreground mb-6">
                            Our support team is here to help.
                        </p>
                        <Link
                            href="mailto:support@pricelens.app"
                            className="text-primary hover:underline font-medium"
                        >
                            Contact Support →
                        </Link>
                    </div>
                </section>
            </main>
            <Footer />
        </div>
    );
}
