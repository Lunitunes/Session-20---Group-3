import { Card } from "@/components/ui/card"
import { formatTimestamp } from "@/lib/utils";
import { AnalysisSummary } from "@/types/analysis"
import { Eye, Trash2 } from 'lucide-react';
import Link from "next/link";

export default function ListItem({data}: { data: AnalysisSummary }){


  const date = formatTimestamp(data.timestamp)
  return(

    <div className="m-2 mx-auto p-3">
      <Card className="shadow-xs items-center">
        <div className="flex flex-row justify-between items-center w-full mx-auto">
          <div className="flex flex-row flex-2 sm:block mx-3">
            <p className="font-mono">{`${data.analysis_id}`}</p>
          </div>
          
          <div className="flex-3">
            <p className="">{`${data.analysis_name}`}</p>
          </div>
          <div className="flex-1.5 mx-3">
            <p className="font-mono">{`${date}`}</p>
          </div>
          <div className="flex-0.5 flex flex-row items-end justify-end">
            <div className="mx-3">
              <Link href={`/analysis/${data.analysis_id}`}>
                <Eye />
              </Link>
            </div>
            <div className="mr-3">
              <Trash2/>
            </div>
          </div>
        </div>
      </Card>
    </div>
  )
}