import { Header, Footer } from "@/components/layout";
import { Metadata } from "next";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
    Zap,
    Target,
    TrendingUp,
    Database,
    Shield,
    Smartphone,
    Camera,
    Bell,
    BarChart3,
    Folder,
    Globe,
    Clock
} from "lucide-react";

export const metadata: Metadata = {
    title: "Features",
    description: "Explore all the features of PriceLens - real-time detection, accurate pricing, collection management, and more.",
};

const featureCategories = [
    {
        title: "Real-Time Detection",
        description: "Our YOLO11-powered detection engine identifies cards in milliseconds",
        features: [
            {
                icon: Camera,
                title: "Webcam Integration",
                description: "Use your laptop webcam or external USB camera for instant detection.",
            },
            {
                icon: Smartphone,
                title: "Mobile Support",
                description: "Works on iOS and Android browsers - no app download required.",
            },
            {
                icon: Zap,
                title: "30+ FPS Processing",
                description: "Smooth real-time detection that keeps up with your camera feed.",
            },
            {
                icon: Target,
                title: "Multi-Card Detection",
                description: "Scan multiple cards at once - perfect for binder pages or collections.",
            },
        ],
    },
    {
        title: "Price Intelligence",
        description: "Accurate, up-to-date pricing from multiple trusted sources",
        features: [
            {
                icon: TrendingUp,
                title: "Live Market Prices",
                description: "Real-time prices from TCGPlayer, CardMarket, and other major marketplaces.",
            },
            {
                icon: BarChart3,
                title: "Price History",
                description: "See how card values have changed over time with interactive charts.",
            },
            {
                icon: Bell,
                title: "Price Alerts",
                description: "Get notified when cards hit your target price - never miss a deal.",
            },
            {
                icon: Globe,
                title: "Multi-Currency",
                description: "View prices in USD, EUR, GBP, and other currencies.",
            },
        ],
    },
    {
        title: "Collection Management",
        description: "Organize and track your Pokemon card collection",
        features: [
            {
                icon: Folder,
                title: "Digital Binder",
                description: "Keep a digital record of every card you own with photos and notes.",
            },
            {
                icon: TrendingUp,
                title: "Portfolio Tracking",
                description: "Watch your collection value grow over time with detailed analytics.",
            },
            {
                icon: Database,
                title: "Export Options",
                description: "Export your collection to CSV or sync with other platforms.",
            },
            {
                icon: Clock,
                title: "Scan History",
                description: "Review all the cards you've scanned with timestamps and prices.",
            },
        ],
    },
    {
        title: "Platform & Security",
        description: "Built for reliability, speed, and privacy",
        features: [
            {
                icon: Shield,
                title: "Privacy First",
                description: "Camera data is processed in real-time and never stored on our servers.",
            },
            {
                icon: Zap,
                title: "<100ms Latency",
                description: "Optimized pipeline delivers results before you can blink.",
            },
            {
                icon: Database,
                title: "10,000+ Cards",
                description: "Comprehensive database from Base Set to the latest releases.",
            },
            {
                icon: Globe,
                title: "Works Offline",
                description: "Core detection works offline - pricing syncs when connected.",
            },
        ],
    },
];

export default function FeaturesPage() {
    return (
        <div className="min-h-screen flex flex-col">
            <Header />
            <main className="flex-1 pt-24">
                {/* Hero */}
                <section className="py-16 relative overflow-hidden">
                    <div className="absolute inset-0 -z-10">
                        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl" />
                        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/10 rounded-full blur-3xl" />
                    </div>

                    <div className="container mx-auto px-4 text-center">
                        <Badge variant="outline" className="mb-6 px-4 py-2 border-primary/30 bg-primary/10">
                            All Features
                        </Badge>
                        <h1 className="text-4xl md:text-5xl font-bold mb-6">
                            Everything You Need to{" "}
                            <span className="text-gradient">Price Cards</span>
                        </h1>
                        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                            Powerful AI detection, real-time pricing, and collection management — all in one platform.
                        </p>
                    </div>
                </section>

                {/* Feature Categories */}
                {featureCategories.map((category, categoryIndex) => (
                    <section
                        key={category.title}
                        className={`py-16 ${categoryIndex % 2 === 1 ? "bg-card/30" : ""}`}
                    >
                        <div className="container mx-auto px-4">
                            <div className="text-center mb-12">
                                <h2 className="text-3xl font-bold mb-3">{category.title}</h2>
                                <p className="text-muted-foreground max-w-xl mx-auto">
                                    {category.description}
                                </p>
                            </div>

                            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                                {category.features.map((feature) => (
                                    <Card key={feature.title} className="bg-card/50 border-border/50 hover:border-primary/30 transition-colors">
                                        <CardContent className="p-6">
                                            <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4">
                                                <feature.icon className="w-6 h-6 text-primary" />
                                            </div>
                                            <h3 className="font-semibold mb-2">{feature.title}</h3>
                                            <p className="text-sm text-muted-foreground">{feature.description}</p>
                                        </CardContent>
                                    </Card>
                                ))}
                            </div>
                        </div>
                    </section>
                ))}

                {/* Comparison */}
                <section className="py-16">
                    <div className="container mx-auto px-4">
                        <div className="text-center mb-12">
                            <h2 className="text-3xl font-bold mb-3">Why PriceLens?</h2>
                            <p className="text-muted-foreground">
                                See how we compare to traditional methods
                            </p>
                        </div>

                        <div className="max-w-4xl mx-auto">
                            <Card className="overflow-hidden">
                                <CardContent className="p-0">
                                    <table className="w-full">
                                        <thead className="bg-muted/50">
                                            <tr>
                                                <th className="text-left p-4 font-semibold">Feature</th>
                                                <th className="text-center p-4 font-semibold">PriceLens</th>
                                                <th className="text-center p-4 font-semibold">Manual Lookup</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-border">
                                            <tr>
                                                <td className="p-4">Time to Price a Card</td>
                                                <td className="p-4 text-center text-green-400 font-semibold">&lt;1 second</td>
                                                <td className="p-4 text-center text-muted-foreground">30-60 seconds</td>
                                            </tr>
                                            <tr>
                                                <td className="p-4">Multi-Card Scanning</td>
                                                <td className="p-4 text-center text-green-400">✓</td>
                                                <td className="p-4 text-center text-muted-foreground">✗</td>
                                            </tr>
                                            <tr>
                                                <td className="p-4">Hands-Free</td>
                                                <td className="p-4 text-center text-green-400">✓</td>
                                                <td className="p-4 text-center text-muted-foreground">✗</td>
                                            </tr>
                                            <tr>
                                                <td className="p-4">Collection Tracking</td>
                                                <td className="p-4 text-center text-green-400">✓</td>
                                                <td className="p-4 text-center text-muted-foreground">Spreadsheet</td>
                                            </tr>
                                            <tr>
                                                <td className="p-4">Price Alerts</td>
                                                <td className="p-4 text-center text-green-400">✓</td>
                                                <td className="p-4 text-center text-muted-foreground">✗</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </CardContent>
                            </Card>
                        </div>
                    </div>
                </section>
            </main>
            <Footer />
        </div>
    );
}
