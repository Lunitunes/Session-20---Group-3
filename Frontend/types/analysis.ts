

export interface AnalysisSummary {
  id:string,
  name: string,
  row_count: number,
  timestamp: string,
  category_counts: Record<string, number>,
  model: string
}