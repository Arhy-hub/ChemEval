import type { Metadata, Viewport } from "next";
import { Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";
import { VaultProvider } from "@/lib/vault";
import { DemoBanner } from "@/components/DemoBanner";
import { RegisterSW } from "@/components/RegisterSW";

const jakarta = Plus_Jakarta_Sans({ variable: "--font-jakarta", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Juno",
  description: "Your friendly assistant for building a private medical profile for drug trials.",
  appleWebApp: { capable: true, title: "Juno", statusBarStyle: "black-translucent" },
  icons: { icon: "/icon.svg", apple: "/icon-192.png" },
};

export const viewport: Viewport = {
  themeColor: "#070b12",
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en-GB">
      <body className={`${jakarta.variable} font-sans antialiased`}>
        <VaultProvider>
          <div className="app-bg mx-auto flex h-dvh max-w-md flex-col overflow-hidden">
            <DemoBanner />
            <div className="flex min-h-0 flex-1 flex-col">{children}</div>
          </div>
        </VaultProvider>
        <RegisterSW />
      </body>
    </html>
  );
}
