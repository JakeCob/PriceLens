"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
    Scan,
    Camera,
    TrendingUp,
    Bell,
    Folder,
    Settings,
    LogOut,
    Menu,
    X,
    ChevronRight,
    DollarSign,
    Package,
    Zap,
    BarChart3
} from "lucide-react";
import Link from "next/link";
import Image from "next/image";

const sidebarLinks = [
    { icon: Scan, label: "Scanner", href: "/dashboard/scan" },
    { icon: Folder, label: "Collection", href: "/dashboard/collection" },
    { icon: Bell, label: "Alerts", href: "/dashboard/alerts" },
    { icon: BarChart3, label: "Analytics", href: "/dashboard/analytics" },
    { icon: Settings, label: "Settings", href: "/dashboard/settings" },
];

// Sample collection cards with real images
const collectionCards = [
    {
        id: "base1-4",
        name: "Charizard",
        set: "Base Set",
        price: 199.99,
        change: 5.2,
        image: "/images/charizard.jpg",
        rarity: "Holo Rare"
    },
    {
        id: "base1-2",
        name: "Blastoise",
        set: "Base Set",
        price: 89.99,
        change: 2.1,
        image: "/images/blastoise.jpg",
        rarity: "Holo Rare"
    },
    {
        id: "base1-1",
        name: "Alakazam",
        set: "Base Set",
        price: 45.00,
        change: -1.5,
        image: "/images/alakazam.jpg",
        rarity: "Holo Rare"
    },
    {
        id: "me1-60",
        name: "Mega Gardevoir ex",
        set: "Mythical Island",
        price: 12.50,
        change: 8.3,
        image: "/images/Mega Gardevoir ex_me1-60.jpg",
        rarity: "Double Rare"
    },
    {
        id: "me1-3",
        name: "Mega Venusaur ex",
        set: "Mythical Island",
        price: 8.99,
        change: 3.2,
        image: "/images/Mega Venusaur ex_me1-3.jpg",
        rarity: "Double Rare"
    },
    {
        id: "me2-58",
        name: "Honchkrow",
        set: "Mythical Island",
        price: 0.04,
        change: 0.0,
        image: "/images/honchkrow.jpg",
        rarity: "Common"
    },
];

const recentScans = [
    { name: "Charizard", set: "Base Set", price: 199.99, time: "2 min ago", image: "/images/charizard.jpg" },
    { name: "Mega Gardevoir ex", set: "ME1", price: 12.50, time: "5 min ago", image: "/images/Mega Gardevoir ex_me1-60.jpg" },
    { name: "Honchkrow", set: "ME2", price: 0.04, time: "10 min ago", image: "/images/honchkrow.jpg" },
];

export default function DashboardPage() {
    const [sidebarOpen, setSidebarOpen] = useState(false);

    const totalValue = collectionCards.reduce((sum, card) => sum + card.price, 0);

    return (
        <div className="min-h-screen bg-background flex">
            {/* Sidebar */}
            <aside className={`
        fixed inset-y-0 left-0 z-50 w-64 bg-card border-r border-border transform transition-transform lg:translate-x-0 lg:static
        ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}
      `}>
                <div className="flex flex-col h-full">
                    {/* Logo */}
                    <div className="flex items-center justify-between p-4 border-b border-border">
                        <Link href="/" className="flex items-center gap-2">
                            <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center">
                                <Scan className="w-5 h-5 text-white" />
                            </div>
                            <span className="text-xl font-bold text-gradient">PriceLens</span>
                        </Link>
                        <Button
                            variant="ghost"
                            size="icon"
                            className="lg:hidden"
                            onClick={() => setSidebarOpen(false)}
                        >
                            <X className="w-5 h-5" />
                        </Button>
                    </div>

                    {/* Nav Links */}
                    <nav className="flex-1 p-4 space-y-2">
                        {sidebarLinks.map((link) => (
                            <Link
                                key={link.href}
                                href={link.href}
                                className="flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-muted-foreground hover:bg-muted hover:text-foreground"
                            >
                                <link.icon className="w-5 h-5" />
                                {link.label}
                            </Link>
                        ))}
                    </nav>

                    {/* User */}
                    <div className="p-4 border-t border-border">
                        <div className="flex items-center gap-3 mb-4">
                            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center text-white font-semibold">
                                JC
                            </div>
                            <div className="flex-1 min-w-0">
                                <p className="font-medium truncate">Jake Cob</p>
                                <p className="text-xs text-muted-foreground">Pro Plan</p>
                            </div>
                        </div>
                        <Button variant="ghost" size="sm" className="w-full justify-start text-muted-foreground" asChild>
                            <Link href="/">
                                <LogOut className="w-4 h-4 mr-2" />
                                Sign out
                            </Link>
                        </Button>
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <div className="flex-1 flex flex-col min-w-0">
                {/* Top Bar */}
                <header className="h-16 border-b border-border flex items-center justify-between px-4">
                    <Button
                        variant="ghost"
                        size="icon"
                        className="lg:hidden"
                        onClick={() => setSidebarOpen(true)}
                    >
                        <Menu className="w-5 h-5" />
                    </Button>

                    <h1 className="text-xl font-semibold hidden lg:block">Dashboard</h1>

                    <div className="flex items-center gap-4">
                        <Badge variant="outline" className="hidden sm:flex">
                            <Zap className="w-3 h-3 mr-1 text-green-400" />
                            Pro Plan
                        </Badge>
                    </div>
                </header>

                {/* Dashboard Content */}
                <main className="flex-1 p-6 overflow-auto">
                    {/* Stats Cards */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                        <Card className="bg-card/50 border-border/50">
                            <CardContent className="p-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-muted-foreground">Collection Value</p>
                                        <p className="text-2xl font-bold">${totalValue.toFixed(2)}</p>
                                    </div>
                                    <div className="w-10 h-10 rounded-lg bg-green-500/10 flex items-center justify-center">
                                        <DollarSign className="w-5 h-5 text-green-500" />
                                    </div>
                                </div>
                                <p className="text-xs text-green-400 mt-2">+12.5% this month</p>
                            </CardContent>
                        </Card>

                        <Card className="bg-card/50 border-border/50">
                            <CardContent className="p-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-muted-foreground">Total Cards</p>
                                        <p className="text-2xl font-bold">{collectionCards.length}</p>
                                    </div>
                                    <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                                        <Package className="w-5 h-5 text-primary" />
                                    </div>
                                </div>
                                <p className="text-xs text-muted-foreground mt-2">+3 this week</p>
                            </CardContent>
                        </Card>

                        <Card className="bg-card/50 border-border/50">
                            <CardContent className="p-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-muted-foreground">Scans Today</p>
                                        <p className="text-2xl font-bold">12</p>
                                    </div>
                                    <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center">
                                        <Scan className="w-5 h-5 text-accent" />
                                    </div>
                                </div>
                                <p className="text-xs text-muted-foreground mt-2">Unlimited (Pro)</p>
                            </CardContent>
                        </Card>

                        <Card className="bg-card/50 border-border/50">
                            <CardContent className="p-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-muted-foreground">Active Alerts</p>
                                        <p className="text-2xl font-bold">3</p>
                                    </div>
                                    <div className="w-10 h-10 rounded-lg bg-secondary/10 flex items-center justify-center">
                                        <Bell className="w-5 h-5 text-secondary" />
                                    </div>
                                </div>
                                <p className="text-xs text-muted-foreground mt-2">1 triggered today</p>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Quick Scan Button */}
                    <Card className="mb-8 bg-gradient-to-br from-primary/10 to-accent/10 border-primary/20">
                        <CardContent className="p-6">
                            <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                                <div className="flex items-center gap-4">
                                    <div className="w-14 h-14 rounded-2xl gradient-primary flex items-center justify-center">
                                        <Camera className="w-7 h-7 text-white" />
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-semibold">Ready to Scan</h3>
                                        <p className="text-muted-foreground">Point your camera at a Pokemon card to get instant prices</p>
                                    </div>
                                </div>
                                <Button size="lg" className="gradient-primary" asChild>
                                    <Link href="/dashboard/scan">
                                        Start Scanning
                                        <ChevronRight className="w-5 h-5 ml-1" />
                                    </Link>
                                </Button>
                            </div>
                        </CardContent>
                    </Card>

                    {/* My Collection */}
                    <div className="mb-8">
                        <div className="flex items-center justify-between mb-4">
                            <h2 className="text-xl font-semibold">My Collection</h2>
                            <Link href="/dashboard/collection" className="text-sm text-primary hover:underline">
                                View all →
                            </Link>
                        </div>
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
                            {collectionCards.map((card) => (
                                <Card key={card.id} className="bg-card/50 border-border/50 overflow-hidden group hover:border-primary/50 transition-colors">
                                    <div className="aspect-[2.5/3.5] relative bg-slate-800">
                                        <img
                                            src={card.image}
                                            alt={card.name}
                                            className="absolute inset-0 w-full h-full object-cover group-hover:scale-105 transition-transform"
                                        />
                                        <Badge className="absolute top-2 right-2 text-xs bg-black/70">
                                            {card.rarity}
                                        </Badge>
                                    </div>
                                    <CardContent className="p-3">
                                        <p className="font-medium text-sm truncate">{card.name}</p>
                                        <p className="text-xs text-muted-foreground">{card.set}</p>
                                        <div className="flex items-center justify-between mt-2">
                                            <p className="font-bold text-green-400">${card.price.toFixed(2)}</p>
                                            <span className={`text-xs ${card.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                {card.change >= 0 ? '+' : ''}{card.change}%
                                            </span>
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    </div>

                    {/* Recent Scans & Top Cards */}
                    <div className="grid lg:grid-cols-2 gap-6">
                        <Card className="bg-card/50 border-border/50">
                            <CardContent className="p-6">
                                <div className="flex items-center justify-between mb-4">
                                    <h3 className="font-semibold">Recent Scans</h3>
                                    <Link href="/dashboard/history" className="text-sm text-primary hover:underline">
                                        View all
                                    </Link>
                                </div>
                                <div className="space-y-4">
                                    {recentScans.map((scan, i) => (
                                        <div key={i} className="flex items-center justify-between">
                                            <div className="flex items-center gap-3">
                                                <div className="w-12 h-16 rounded-lg overflow-hidden bg-slate-800">
                                                    <img
                                                        src={scan.image}
                                                        alt={scan.name}
                                                        className="w-full h-full object-cover"
                                                    />
                                                </div>
                                                <div>
                                                    <p className="font-medium">{scan.name}</p>
                                                    <p className="text-xs text-muted-foreground">{scan.set} • {scan.time}</p>
                                                </div>
                                            </div>
                                            <p className="font-semibold text-green-400">${scan.price.toFixed(2)}</p>
                                        </div>
                                    ))}
                                </div>
                            </CardContent>
                        </Card>

                        <Card className="bg-card/50 border-border/50">
                            <CardContent className="p-6">
                                <div className="flex items-center justify-between mb-4">
                                    <h3 className="font-semibold">Top Value Cards</h3>
                                    <Link href="/dashboard/collection" className="text-sm text-primary hover:underline">
                                        View collection
                                    </Link>
                                </div>
                                <div className="space-y-4">
                                    {collectionCards
                                        .sort((a, b) => b.price - a.price)
                                        .slice(0, 3)
                                        .map((card, i) => (
                                            <div key={card.id} className="flex items-center justify-between">
                                                <div className="flex items-center gap-3">
                                                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-yellow-500/20 to-orange-500/20 flex items-center justify-center font-bold text-yellow-400">
                                                        {i + 1}
                                                    </div>
                                                    <div className="w-10 h-14 rounded overflow-hidden bg-slate-800">
                                                        <img
                                                            src={card.image}
                                                            alt={card.name}
                                                            className="w-full h-full object-cover"
                                                        />
                                                    </div>
                                                    <div>
                                                        <p className="font-medium">{card.name}</p>
                                                        <p className="text-xs text-muted-foreground">{card.set}</p>
                                                    </div>
                                                </div>
                                                <div className="text-right">
                                                    <p className="font-semibold">${card.price.toFixed(2)}</p>
                                                    <p className={`text-xs ${card.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                        {card.change >= 0 ? '+' : ''}{card.change}%
                                                    </p>
                                                </div>
                                            </div>
                                        ))}
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </main>
            </div>
        </div>
    );
}
