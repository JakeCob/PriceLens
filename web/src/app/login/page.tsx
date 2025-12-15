"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Header, Footer } from "@/components/layout";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Scan, Github, Mail } from "lucide-react";
import Link from "next/link";

export default function LoginPage() {
    const [isLoading, setIsLoading] = useState(false);
    const router = useRouter();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        // Simulate login then redirect to scanner
        await new Promise((resolve) => setTimeout(resolve, 500));
        router.push("/dashboard/scan");
    };

    return (
        <div className="min-h-screen flex flex-col">
            <Header />
            <main className="flex-1 flex items-center justify-center pt-16 pb-8 px-4">
                <div className="w-full max-w-md">
                    <div className="text-center mb-8">
                        <div className="w-16 h-16 rounded-2xl gradient-primary flex items-center justify-center mx-auto mb-4">
                            <Scan className="w-8 h-8 text-white" />
                        </div>
                        <h1 className="text-2xl font-bold">Welcome back</h1>
                        <p className="text-muted-foreground">Sign in to your PriceLens account</p>
                    </div>

                    <Card className="bg-card/50 border-border/50">
                        <CardContent className="p-6">
                            <form onSubmit={handleSubmit} className="space-y-4">
                                <div className="space-y-2">
                                    <Label htmlFor="email">Email</Label>
                                    <Input
                                        id="email"
                                        type="email"
                                        placeholder="you@example.com"
                                        required
                                    />
                                </div>
                                <div className="space-y-2">
                                    <div className="flex justify-between">
                                        <Label htmlFor="password">Password</Label>
                                        <Link href="/forgot-password" className="text-sm text-primary hover:underline">
                                            Forgot password?
                                        </Link>
                                    </div>
                                    <Input
                                        id="password"
                                        type="password"
                                        placeholder="••••••••"
                                        required
                                    />
                                </div>
                                <Button
                                    type="submit"
                                    className="w-full gradient-primary"
                                    disabled={isLoading}
                                >
                                    {isLoading ? "Signing in..." : "Sign in"}
                                </Button>
                            </form>

                            <div className="relative my-6">
                                <Separator />
                                <span className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-card px-2 text-xs text-muted-foreground">
                                    or continue with
                                </span>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <Button variant="outline" className="w-full">
                                    <Github className="w-4 h-4 mr-2" />
                                    GitHub
                                </Button>
                                <Button variant="outline" className="w-full">
                                    <Mail className="w-4 h-4 mr-2" />
                                    Google
                                </Button>
                            </div>
                        </CardContent>
                    </Card>

                    <p className="text-center text-sm text-muted-foreground mt-6">
                        Don&apos;t have an account?{" "}
                        <Link href="/signup" className="text-primary hover:underline font-medium">
                            Sign up free
                        </Link>
                    </p>
                </div>
            </main>
            <Footer />
        </div>
    );
}
