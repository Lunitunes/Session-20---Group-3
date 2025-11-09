import TrainingDatasetVisuals from "../TrainingDataVisuals/page";
import RadarChartCompare from "@/components/charts/RadarChart";

import { ChartConfig } from "@/components/ui/chart";

export default async function PredictionChartPage({ params }: { params: { analysis_id: string } }) {
  const { analysis_id } = await params;
  
  
  // Load training.json to get training dataset id
  const trainingMeta = await fetch("http://localhost:8000/get_training_data", { cache: "no-store" })
  .then(r => r.json());
  const trainingId = trainingMeta[0].analysis_id;
  
  // Get training radar
  const trainingRadar = await fetch(`http://localhost:8000/get_training_radar/${trainingId}`, { cache: "no-store" })
  .then(r => r.json());
  
  // Get prediction radar
  const predictionRadar = await fetch(`http://localhost:8000/get_prediction_radar/${analysis_id}`, { cache: "no-store" })
  .then(r => r.json());
  
  const chartConfig: ChartConfig = {
    Normal:        { label: "Normal",        color: "hsl(var(--chart-1))" },
    Fuzzers:       { label: "Fuzzers",       color: "hsl(var(--chart-2))" },
    Analysis:      { label: "Analysis",      color: "hsl(var(--chart-3))" },
    Backdoor:      { label: "Backdoor",      color: "hsl(var(--chart-4))" },
    DoS:           { label: "DoS",           color: "hsl(var(--chart-5))" },
    Exploits:      { label: "Exploits",      color: "hsl(var(--chart-6))" },
    Generic:       { label: "Generic",       color: "hsl(var(--chart-7))" },
    Reconnaissance:{ label: "Recon",         color: "hsl(var(--chart-8))" },
    Shellcode:     { label: "Shellcode",     color: "hsl(var(--chart-9))" },
    Worms:         { label: "Worms",         color: "hsl(var(--chart-10))" },
  
    Prediction:    { label: "Prediction Input", color: "hsl(var(--primary))" },
  };
  
  return (
    <div className="w-5xl m-auto space-y-12 my-5">
      <h1 className="text-center text-3xl font-semibold">Prediction Analysis</h1>

      {/* Training Dataset Overview */}
      <TrainingDatasetVisuals />

      {/* Radar Comparison */}
      <RadarChartCompare
        config={chartConfig}
        trainingRadar={trainingRadar.trainingRadar}
        predictionRadar={predictionRadar.radarDataPrediction}
        categories={trainingRadar.metric}
      />
    </div>
  );
}
