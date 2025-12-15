import Link from "next/link";
import { Scan, Github, Twitter, Youtube, Mail } from "lucide-react";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

const footerLinks = {
    product: [
        { href: "/features", label: "Features" },
        { href: "/pricing", label: "Pricing" },
        { href: "/docs", label: "Documentation" },
        { href: "/changelog", label: "Changelog" },
    ],
    company: [
        { href: "/about", label: "About" },
        { href: "/blog", label: "Blog" },
        { href: "/contact", label: "Contact" },
    ],
    legal: [
        { href: "/privacy", label: "Privacy Policy" },
        { href: "/terms", label: "Terms of Service" },
        { href: "/cookies", label: "Cookie Policy" },
    ],
};

const socialLinks = [
    { href: "https://github.com/JakeCob/PriceLens", icon: Github, label: "GitHub" },
    { href: "https://twitter.com/pricelens", icon: Twitter, label: "Twitter" },
    { href: "https://youtube.com/@pricelens", icon: Youtube, label: "YouTube" },
];

export function Footer() {
    return (
        <footer className="border-t border-border bg-card/50">
            <div className="container mx-auto px-4 py-12">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-8">
                    {/* Brand Section */}
                    <div className="lg:col-span-2">
                        <Link href="/" className="flex items-center gap-2 mb-4">
                            <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center">
                                <Scan className="w-5 h-5 text-white" />
                            </div>
                            <span className="text-xl font-bold text-gradient">PriceLens</span>
                        </Link>
                        <p className="text-muted-foreground text-sm mb-6 max-w-xs">
                            Real-time Pokemon card price detection powered by AI.
                            Point your camera at any card and get instant market prices.
                        </p>

                        {/* Newsletter */}
                        <div className="space-y-2">
                            <p className="text-sm font-medium">Subscribe to updates</p>
                            <div className="flex gap-2">
                                <Input
                                    type="email"
                                    placeholder="Enter your email"
                                    className="max-w-[200px]"
                                />
                                <Button size="sm" className="gradient-primary">
                                    <Mail className="w-4 h-4" />
                                </Button>
                            </div>
                        </div>
                    </div>

                    {/* Product Links */}
                    <div>
                        <h4 className="font-semibold mb-4">Product</h4>
                        <ul className="space-y-2">
                            {footerLinks.product.map((link) => (
                                <li key={link.href}>
                                    <Link
                                        href={link.href}
                                        className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                                    >
                                        {link.label}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* Company Links */}
                    <div>
                        <h4 className="font-semibold mb-4">Company</h4>
                        <ul className="space-y-2">
                            {footerLinks.company.map((link) => (
                                <li key={link.href}>
                                    <Link
                                        href={link.href}
                                        className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                                    >
                                        {link.label}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* Legal Links */}
                    <div>
                        <h4 className="font-semibold mb-4">Legal</h4>
                        <ul className="space-y-2">
                            {footerLinks.legal.map((link) => (
                                <li key={link.href}>
                                    <Link
                                        href={link.href}
                                        className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                                    >
                                        {link.label}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>

                <Separator className="my-8" />

                {/* Bottom Bar */}
                <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
                    <p className="text-sm text-muted-foreground">
                        © {new Date().getFullYear()} PriceLens. Made with ❤️ by collectors, for collectors.
                    </p>

                    {/* Social Links */}
                    <div className="flex items-center gap-4">
                        {socialLinks.map((social) => (
                            <a
                                key={social.href}
                                href={social.href}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-muted-foreground hover:text-foreground transition-colors"
                                aria-label={social.label}
                            >
                                <social.icon className="w-5 h-5" />
                            </a>
                        ))}
                    </div>
                </div>
            </div>
        </footer>
    );
}
