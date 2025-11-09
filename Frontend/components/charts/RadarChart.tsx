"use client";

import { useState } from "react";
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  Tooltip,
  Legend,
} from "recharts";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { ChartContainer, ChartConfig } from "@/components/ui/chart";

interface RadarChartCompareProps {
  config: ChartConfig;
  trainingRadar: Record<string, number[]>;
  predictionRadar: { metric: string; Prediction: number }[];
  categories: string[];
}

export default function RadarChartCompare({
  config,
  trainingRadar,
  predictionRadar,
  categories,
}: RadarChartCompareProps) {
  const data = categories.map((metric, index) => {
    const row: Record<string, any> = { metric };

    Object.entries(trainingRadar).forEach(([className, values]) => {
      row[className] = values[index];
    });

    const p = predictionRadar.find((x) => x.metric === metric);
    if (p) row["Prediction"] = p.Prediction;

    return row;
  });

  const [visible, setVisible] = useState(
    Object.fromEntries(Object.keys(config).map((key) => [key, true]))
  );
  console.log("RADAR DATA:", data);
  return (
    <Card className="bg-muted">
      <CardHeader className="items-center pb-4">
        <CardTitle>Behavior Comparison Radar</CardTitle>
        <CardDescription>
          Compare the uploaded data to each learned category baseline.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4 pb-0">
        <ChartContainer config={config} className="mx-auto aspect-square max-h-[350px]">
          <RadarChart data={data}>
            <PolarAngleAxis dataKey="metric" />
            <PolarGrid />
            <Tooltip />
            <Legend />

            {/* Baseline Profiles */}
            {Object.keys(trainingRadar).map((className) => (
              <Radar
                key={className}
                dataKey={className}
                name={String(config[className].label)}
                stroke={config[className].color}
                fill={config[className].color}
                fillOpacity={0.15}
                hide={!visible[className]}
              />
            ))}

            {/* Prediction Overlay */}
            <Radar
              dataKey="Prediction"
              name={String(config["Prediction"].label)}
              stroke={config["Prediction"].color}
              fill={config["Prediction"].color}
              strokeWidth={3}
              hide={!visible["Prediction"]}
            />
          </RadarChart>
        </ChartContainer>

        {/* Toggle Controls */}
        <div className="flex flex-wrap justify-center gap-2">
          {Object.keys(config).map((key) => (
            <button
              key={key}
              onClick={() => setVisible((v) => ({ ...v, [key]: !v[key] }))}
              className={`px-2 py-1 text-xs rounded border transition ${
                visible[key]
                  ? "bg-primary text-white border-primary"
                  : "border-muted text-muted-foreground"
              }`}
            >
              {config[key].label}
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
