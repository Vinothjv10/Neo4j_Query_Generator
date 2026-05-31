import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Text2SQL BI — Natural Language to SQL",
  description: "Convert natural language business questions into SQL queries using AI",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.className}>
      <body className="min-h-screen flex flex-col">
        {children}
        <footer className="mt-auto border-t border-gray-800 py-4 text-center text-xs text-gray-600">
          Text2SQL BI &mdash; Neo4j Schema Registry &middot; NVIDIA LLM &middot; PostgreSQL
        </footer>
      </body>
    </html>
  );
}
