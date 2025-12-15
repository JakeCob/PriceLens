"use client";

import Link from "next/link";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Menu, X, Scan, Sparkles } from "lucide-react";
import { motion } from "framer-motion";

const navLinks = [
    { href: "/features", label: "Features" },
    { href: "/pricing", label: "Pricing" },
    { href: "/docs", label: "Docs" },
    { href: "/about", label: "About" },
];

export function Header() {
    const [isOpen, setIsOpen] = useState(false);

    return (
        <motion.header
            initial={{ y: -100 }}
            animate={{ y: 0 }}
            className="fixed top-0 left-0 right-0 z-50 glass"
        >
            <div className="container mx-auto px-4">
                <div className="flex h-16 items-center justify-between">
                    {/* Logo */}
                    <Link href="/" className="flex items-center gap-2">
                        <div className="relative">
                            <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center">
                                <Scan className="w-5 h-5 text-white" />
                            </div>
                            <Sparkles className="w-3 h-3 text-secondary absolute -top-1 -right-1" />
                        </div>
                        <span className="text-xl font-bold text-gradient hidden sm:inline">
                            PriceLens
                        </span>
                    </Link>

                    {/* Desktop Navigation */}
                    <nav className="hidden md:flex items-center gap-6">
                        {navLinks.map((link) => (
                            <Link
                                key={link.href}
                                href={link.href}
                                className="text-muted-foreground hover:text-foreground transition-colors text-sm font-medium"
                            >
                                {link.label}
                            </Link>
                        ))}
                    </nav>

                    {/* Desktop CTA */}
                    <div className="hidden md:flex items-center gap-3">
                        <Button variant="ghost" size="sm" asChild>
                            <Link href="/dashboard">Log in</Link>
                        </Button>
                        <Button size="sm" className="gradient-primary" asChild>
                            <Link href="/dashboard">
                                Try Free
                            </Link>
                        </Button>
                    </div>

                    {/* Mobile Menu */}
                    <Sheet open={isOpen} onOpenChange={setIsOpen}>
                        <SheetTrigger asChild className="md:hidden">
                            <Button variant="ghost" size="icon">
                                <Menu className="h-5 w-5" />
                                <span className="sr-only">Toggle menu</span>
                            </Button>
                        </SheetTrigger>
                        <SheetContent side="right" className="w-[300px] glass">
                            <div className="flex flex-col gap-6 mt-6">
                                <nav className="flex flex-col gap-4">
                                    {navLinks.map((link) => (
                                        <Link
                                            key={link.href}
                                            href={link.href}
                                            onClick={() => setIsOpen(false)}
                                            className="text-foreground hover:text-primary transition-colors text-lg font-medium"
                                        >
                                            {link.label}
                                        </Link>
                                    ))}
                                </nav>
                                <div className="flex flex-col gap-3 pt-4 border-t border-border">
                                    <Button variant="outline" asChild>
                                        <Link href="/dashboard">Log in</Link>
                                    </Button>
                                    <Button className="gradient-primary" asChild>
                                        <Link href="/dashboard">
                                            Try Free
                                        </Link>
                                    </Button>
                                </div>
                            </div>
                        </SheetContent>
                    </Sheet>
                </div>
            </div>
        </motion.header>
    );
}
