import DataVisualisations from "./data-visualisations/page";
import InputComponent from "./input/page";


export default function Home() {
  return (
    <div className="w-2xl mx-auto">
      <main className="grid items-center mx-auto">
        <InputComponent/>
        <DataVisualisations/>
      </main>
    </div>
  );
}
