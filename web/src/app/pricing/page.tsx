import { Header, Footer } from "@/components/layout";
import { Metadata } from "next";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Check, X, Zap, Star, Building } from "lucide-react";
import Link from "next/link";

export const metadata: Metadata = {
    title: "Pricing",
    description: "Simple, transparent pricing for PriceLens. Start free, upgrade when you need more.",
};

const plans = [
    {
        name: "Free",
        description: "Perfect for casual collectors",
        price: "$0",
        period: "forever",
        icon: Zap,
        features: [
            { text: "10 scans per day", included: true },
            { text: "Basic price data", included: true },
            { text: "Single card detection", included: true },
            { text: "Community support", included: true },
            { text: "Multi-card detection", included: false },
            { text: "Price alerts", included: false },
            { text: "Collection tracking", included: false },
            { text: "API access", included: false },
        ],
        cta: "Get Started",
        href: "/dashboard",
        popular: false,
    },
    {
        name: "Pro",
        description: "For serious collectors",
        price: "$9.99",
        period: "per month",
        icon: Star,
        features: [
            { text: "Unlimited scans", included: true },
            { text: "Real-time price updates", included: true },
            { text: "Multi-card detection", included: true },
            { text: "Collection tracking", included: true },
            { text: "Price alerts", included: true },
            { text: "Priority support", included: true },
            { text: "Export to CSV", included: true },
            { text: "API access", included: false },
        ],
        cta: "Start Free Trial",
        href: "/dashboard",
        popular: true,
    },
    {
        name: "Enterprise",
        description: "For businesses & power users",
        price: "Custom",
        period: "contact us",
        icon: Building,
        features: [
            { text: "Everything in Pro", included: true },
            { text: "Full API access", included: true },
            { text: "Custom integration", included: true },
            { text: "Dedicated support", included: true },
            { text: "SLA guarantee", included: true },
            { text: "White-label option", included: true },
            { text: "Bulk scanning", included: true },
            { text: "Custom training", included: true },
        ],
        cta: "Contact Sales",
        href: "mailto:sales@pricelens.app",
        popular: false,
    },
];

const faqs = [
    {
        question: "Can I cancel anytime?",
        answer: "Yes! You can cancel your subscription at any time. Your access will continue until the end of your billing period.",
    },
    {
        question: "What payment methods do you accept?",
        answer: "We accept all major credit cards (Visa, Mastercard, American Express) and PayPal. Enterprise customers can pay via invoice.",
    },
    {
        question: "Is there a free trial for Pro?",
        answer: "Yes! Pro comes with a 14-day free trial. No credit card required to start.",
    },
    {
        question: "Do you offer educational discounts?",
        answer: "Yes! Students and educators get 50% off Pro. Contact us with your .edu email to claim your discount.",
    },
    {
        question: "What happens if I exceed my scan limit on Free?",
        answer: "You'll see a friendly message asking you to wait until tomorrow or upgrade to Pro for unlimited scans.",
    },
];

export default function PricingPage() {
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
                        <h1 className="text-4xl md:text-5xl font-bold mb-6">
                            Simple, Transparent{" "}
                            <span className="text-gradient">Pricing</span>
                        </h1>
                        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                            Start free, upgrade when you need more. No hidden fees, no surprises.
                        </p>
                    </div>
                </section>

                {/* Pricing Cards */}
                <section className="py-8">
                    <div className="container mx-auto px-4">
                        <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
                            {plans.map((plan) => (
                                <Card
                                    key={plan.name}
                                    className={`relative bg-card/50 border-border/50 ${plan.popular ? "border-primary ring-2 ring-primary/20" : ""
                                        }`}
                                >
                                    {plan.popular && (
                                        <Badge className="absolute -top-3 left-1/2 -translate-x-1/2 gradient-primary">
                                            Most Popular
                                        </Badge>
                                    )}
                                    <CardHeader className="text-center pb-4">
                                        <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-4">
                                            <plan.icon className="w-6 h-6 text-primary" />
                                        </div>
                                        <CardTitle className="text-2xl">{plan.name}</CardTitle>
                                        <CardDescription>{plan.description}</CardDescription>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="text-center mb-6">
                                            <span className="text-4xl font-bold">{plan.price}</span>
                                            <span className="text-muted-foreground ml-2">/{plan.period}</span>
                                        </div>

                                        <ul className="space-y-3 mb-6">
                                            {plan.features.map((feature) => (
                                                <li key={feature.text} className="flex items-center gap-3">
                                                    {feature.included ? (
                                                        <Check className="w-5 h-5 text-green-400 flex-shrink-0" />
                                                    ) : (
                                                        <X className="w-5 h-5 text-muted-foreground/50 flex-shrink-0" />
                                                    )}
                                                    <span className={feature.included ? "" : "text-muted-foreground/50"}>
                                                        {feature.text}
                                                    </span>
                                                </li>
                                            ))}
                                        </ul>

                                        <Button
                                            className={`w-full ${plan.popular ? "gradient-primary" : ""}`}
                                            variant={plan.popular ? "default" : "outline"}
                                            asChild
                                        >
                                            <Link href={plan.href}>{plan.cta}</Link>
                                        </Button>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    </div>
                </section>

                {/* FAQ */}
                <section className="py-16 bg-card/30">
                    <div className="container mx-auto px-4">
                        <h2 className="text-3xl font-bold mb-12 text-center">
                            Frequently Asked Questions
                        </h2>
                        <div className="max-w-3xl mx-auto space-y-6">
                            {faqs.map((faq) => (
                                <Card key={faq.question} className="bg-card/50 border-border/50">
                                    <CardContent className="p-6">
                                        <h3 className="font-semibold mb-2">{faq.question}</h3>
                                        <p className="text-muted-foreground">{faq.answer}</p>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    </div>
                </section>

                {/* CTA */}
                <section className="py-16">
                    <div className="container mx-auto px-4 text-center">
                        <h2 className="text-3xl font-bold mb-4">Still have questions?</h2>
                        <p className="text-muted-foreground mb-8">
                            We&apos;re here to help. Reach out anytime.
                        </p>
                        <div className="flex gap-4 justify-center">
                            <Button variant="outline" asChild>
                                <Link href="/docs">View Documentation</Link>
                            </Button>
                            <Button className="gradient-primary" asChild>
                                <Link href="mailto:support@pricelens.app">Contact Support</Link>
                            </Button>
                        </div>
                    </div>
                </section>
            </main>
            <Footer />
        </div>
    );
}
