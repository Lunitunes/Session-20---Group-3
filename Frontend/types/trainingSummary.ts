export interface TrainingSummary {
  analysis_id:string,
  analysis_name: string,
  row_count: number,
  timestamp: string,
  category_count: Record<string, number>
}

export interface CategoryChartRow {
  category: string;
  value: number;
}