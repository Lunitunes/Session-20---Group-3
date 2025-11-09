"use client"

import { TrendingUp } from "lucide-react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Rectangle,
  XAxis,
  Cell,
} from "recharts"

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"

type Props = {
  chartData: { category: string; value: number }[]
  chartConfig: ChartConfig
}

export default function TrainingBarActive({ chartData, chartConfig }: Props) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Category Frequency</CardTitle>
        <CardDescription>Training Dataset Class Distribution</CardDescription>
      </CardHeader>

      <CardContent>
        <ChartContainer config={chartConfig}>
          <BarChart data={chartData} margin={{ bottom: 80 }} >
            <CartesianGrid vertical={false} />

            <XAxis
              dataKey="category"
              tickLine={false}
              tickMargin={30}
              angle={-90}
              axisLine={false}
              tickFormatter={(value: string) =>
                String(chartConfig[value as keyof typeof chartConfig]?.label ?? value)
              }
              
            />

            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent nameKey="value" />}
            />

            <Bar
              dataKey="value"
              radius={8}
              strokeWidth={2}
            >
              {chartData.map((entry) => (
                <Cell
                  key={entry.category}
                  fill={
                    chartConfig[entry.category as keyof typeof chartConfig].color
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
