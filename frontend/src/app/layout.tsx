import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Nexus AI Console",
  description: "DeepSeek-style Chat and API Key Dashboard for Nexus Local LLM",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full dark">
      <body className={`${inter.className} h-full text-slate-200 antialiased`}>
        {children}
      </body>
    </html>
  );
}
