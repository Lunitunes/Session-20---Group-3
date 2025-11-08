import { Table,TableBody,TableCaption,TableCell,TableFooter,TableHead,TableHeader,TableRow, } from "@/components/ui/table"
import ListItem from "@/components/ListItem"

export default function DataVisualisations(){
  const dumby_data = {
    "analysis_id": "f170b0ff",
    "analysis_name": "Test",
    "row_count": 340,
    "timestamp": "2025-11-08T17:26:41.913147",
    "category_count": {
      "Backdoor": 40,
      "Normal": 34,
      "Analysis": 34,
      "DoS": 34,
      "Generic": 34,
      "Reconnaissance": 34,
      "Shellcode": 34,
      "Worms": 34,
      "Fuzzers": 32,
      "Exploits": 30
    }
  }


  return(
    <div className="">
      <h2 className="text-4xl my-5 text-center">Previous Analysis Requests</h2>
      <ListItem data={dumby_data}/>
      <Table>

      </Table>
    </div>
  )
}