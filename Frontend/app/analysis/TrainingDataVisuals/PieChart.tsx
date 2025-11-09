"use client"

import { Pie, PieChart, Cell, Tooltip } from "recharts"

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent
} from "@/components/ui/chart"

type Props = {
  chartConfig: ChartConfig
  chartData: { category: string; value: number }[]
}

export default function ChartPieLegend({ chartConfig, chartData }: Props) {
  return (
    <Card className="flex flex-col">
      <CardHeader className="items-center pb-0">
        <CardTitle>Category Distribution</CardTitle>
        <CardDescription>Training dataset class frequency</CardDescription>
      </CardHeader>

      <CardContent className="flex-1 pb-0">
        <ChartContainer
          config={chartConfig}
          className="mx-auto aspect-square max-h-[300px]"
        >
          <PieChart>
            <ChartTooltip content={<ChartTooltipContent nameKey="Category"/>}/>
            <Pie
              data={chartData}
              dataKey="value"
              nameKey="category"
              innerRadius={0}
              outerRadius={100}
              paddingAngle={0}
            >
              {chartData.map((entry) => (
                <Cell
                  key={entry.category}
                  fill={chartConfig[entry.category as keyof typeof chartConfig].color}
                />
              ))}
            </Pie>

            <ChartLegend
              content={<ChartLegendContent nameKey="category" />}
              className="mt-4 flex flex-wrap gap-3 justify-center"
            />
          </PieChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
