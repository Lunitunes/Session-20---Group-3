import TrainingDatasetVisuals from "../TrainingDataVisuals/page"


export default function ChartsPage(){
  

  // const chartData = 
  


  return(
    <div className="w-5xl m-auto">
      <h1 className="text-center ">Data Visualised</h1>

      <TrainingDatasetVisuals/>
      <div className="grid">
        <div id="BarChart" className="">

        </div>
        <div id="PieChart" className="">

        </div>
        <div className="">

        </div>
      </div>
    </div>
  )
}