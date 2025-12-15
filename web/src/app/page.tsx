import { Header, Footer } from "@/components/layout";
import {
  HeroSection,
  FeaturesSection,
  HowItWorksSection,
  StatsSection,
  CTASection
} from "@/components/features";

export default function HomePage() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1">
        <HeroSection />
        <FeaturesSection />
        <HowItWorksSection />
        <StatsSection />
        <CTASection />
      </main>
      <Footer />
    </div>
  );
}
