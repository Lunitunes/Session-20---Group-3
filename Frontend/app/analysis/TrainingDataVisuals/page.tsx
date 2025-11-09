'use client'
import { Card } from "@/components/ui/card";
import { ChartConfig, ChartContainer } from "@/components/ui/chart";
import { TrainingSummary, CategoryChartRow } from "@/types/trainingSummary";
import ChartPieLegend from "./PieChart";

import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell, 
  Pie, PieChart
} from "recharts";
import TrainingBarActive from "./BarChart";


export default function TrainingDatasetVisuals() {
  const data :TrainingSummary = {
    analysis_id: "f170b0ff",
    analysis_name: "Test",
    row_count: 340,
    timestamp: "2025-11-08T17:26:41.913147",
    category_count: {
      Backdoor: 40,
      Normal: 34,
      Analysis: 34,
      DoS: 34,
      Generic: 34,
      Reconnaissance: 34,
      Shellcode: 34,
      Worms: 34,
      Fuzzers: 32,
      Exploits: 30,
    }
  };

  type CategoryKey = keyof typeof chartConfig;
  const chartData: CategoryChartRow[] = Object.entries(data.category_count).map(
    ([category, value]) => ({
      category: category as CategoryKey,
      value,
    })
  );

  const chartConfig = {
    Backdoor: {
      label: "Backdoor",
      color: "hsl(var(--chart-1))"
    },
    Normal: {
      label: "Normal",
      color: "hsl(var(--chart-2))"
    },
    Analysis: {
      label: "Analysis",
      color: "hsl(var(--chart-3))"
    },
    DoS: {
      label: "DoS",
      color: "hsl(var(--chart-4))"
    },
    Generic: {
      label: "Generic",
      color: "hsl(var(--chart-5))"
    },
    Reconnaissance: {
      label: "Reconnaissance",
      color: "hsl(var(--chart-6))"
    },
    Shellcode: {
      label: "Shellcode",
      color: "hsl(var(--chart-7))"
    },
    Worms: {
      label: "Worms",
      color: "hsl(var(--chart-8))"
    },
    Fuzzers: {
      label: "Fuzzers",
      color: "hsl(var(--chart-9))"
    },
    Exploits: {
      label: "Exploits",
      color: "hsl(var(--chart-10))"
    },
  } satisfies ChartConfig

  return(
    <Card className="w-full grid gap-3 shadow-lg rounded-sm p-4 bg-muted">
      {/* General Stats Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">

      </div>

      {/* Training Graphs Row */}
      <h2 className="text-center text-3xl mt-3">Training Data General Stats Category</h2>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-2 p-5">
        <TrainingBarActive chartConfig={chartConfig} chartData={chartData}/>
        <ChartPieLegend chartConfig={chartConfig} chartData={chartData}/>
      </div>

      <div className="grid grid-cols-1">
        
      </div>
    </Card>
  );
}