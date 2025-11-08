

export interface AnalysisSummary {
  analysis_id:string,
  analysis_name: string,
  row_count: number,
  timestamp: string,
  category_count: Record<string, number>
}