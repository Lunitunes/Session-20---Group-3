import { Table,TableBody,TableCaption,TableCell,TableFooter,TableHead,TableHeader,TableRow, } from "@/components/ui/table"
import ListItem from "@/components/ListItem"
import { AnalysisSummary } from "@/types/analysisSummary";

export default async function DataVisualisations(){
  const res = await fetch("http://localhost:8000/return_index", { cache: "no-store" });
  const analyses: AnalysisSummary[] = await res.json();
  console.log(analyses)
  
  return(
    <div className="">
      <h2 className="text-4xl my-5 text-center">Previous Analysis Requests</h2>
      {analyses.length === 0 && (
        <p className="text-muted-foreground text-sm">No analyses yet. Upload one to get started.</p>
      )}

      {analyses.map((analysis) => (
        <ListItem key={analysis.analysis_id} data={analysis} />
      ))}
    </div>
  )
}