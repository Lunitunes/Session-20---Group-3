import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import { AnalysisSummary } from "@/types/analysis";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatTimestamp(ts: string) {
  return new Date(ts).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export async function fetchAnalysis(id: string): Promise<AnalysisSummary | null> {
  try {
    const res = await fetch("/return_index");

    if (!res.ok) {
      console.error(`Failed to fetch analysis ${id}. HTTP ${res.status}`);
      return null
    }
    const analysisMetadata: AnalysisSummary = await res.json();
    return analysisMetadata
  } catch (err) {
    console.error("Network/Parsing error:", err);
    return null
  }
}