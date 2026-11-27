import { createBrowserRouter } from "react-router";
import { Root } from "./components/Root";
import { DemoMode } from "./components/DemoMode";
import { LearnMode } from "./components/LearnMode";
import { PipelineOverview } from "./components/learn/PipelineOverview";
import { ModuleIR } from "./components/learn/ModuleIR";
import { ModuleSumm } from "./components/learn/ModuleSumm";
import { ModuleQA } from "./components/learn/ModuleQA";
import { ModuleQADistilbert } from "./components/learn/ModuleQADistilbert";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Root,
    children: [
      { index: true, Component: DemoMode },
      { 
        path: "learn", 
        Component: LearnMode,
        children: [
          { index: true, Component: PipelineOverview },
          { path: "ir", Component: ModuleIR },
          { path: "summ", Component: ModuleSumm },
          { path: "qa", Component: ModuleQA },
          { path: "qa-distilbert", Component: ModuleQADistilbert },
        ]
      },
    ],
  },
]);
