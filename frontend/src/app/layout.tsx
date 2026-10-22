import type { Metadata } from "next";
import "./globals.css";


export const metadata: Metadata = {
  title: "Nexus AI Console",
  description: "Private AI chat and API key management for Nexus",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full dark">
      <body className={`h-full text-slate-200 antialiased`}>
        {children}
      </body>
    </html>
  );
}
