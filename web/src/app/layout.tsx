import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { ThemeProvider } from "next-themes";
import "./globals.css";

const inter = Inter({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "PriceLens - Real-Time Pokemon Card Price Scanner",
    template: "%s | PriceLens",
  },
  description: "Point your camera at any Pokemon card and get instant market prices with AI-powered detection. 95% accuracy, 30+ FPS, 10,000+ cards in database.",
  keywords: ["Pokemon", "TCG", "card scanner", "price checker", "AI", "computer vision"],
  authors: [{ name: "PriceLens Team" }],
  creator: "PriceLens",
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://pricelens.app",
    siteName: "PriceLens",
    title: "PriceLens - Real-Time Pokemon Card Price Scanner",
    description: "Point your camera at any Pokemon card and get instant market prices with AI-powered detection.",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "PriceLens - Pokemon Card Price Scanner",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "PriceLens - Real-Time Pokemon Card Price Scanner",
    description: "Point your camera at any Pokemon card and get instant market prices with AI-powered detection.",
    images: ["/og-image.png"],
    creator: "@pricelens",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
  icons: {
    icon: "/favicon.ico",
    shortcut: "/favicon-16x16.png",
    apple: "/apple-touch-icon.png",
  },
  manifest: "/site.webmanifest",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} antialiased`}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
