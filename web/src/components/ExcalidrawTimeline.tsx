import { useEffect, useRef } from "react";
import { Excalidraw } from "@excalidraw/excalidraw";
import "@excalidraw/excalidraw/index.css";
import timelineElementsRaw from "../data/research_papers_timeline_elements.json";

const timelineElements = timelineElementsRaw as any[];

interface ExcalidrawTimelineProps {
  height?: number;
  width?: number | string;
}

export function ExcalidrawTimeline({ height = 700, width = "100%" }: ExcalidrawTimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const apiRef = useRef<any>(null);

  useEffect(() => {
    if (apiRef.current) {
      // Scroll to show all content on mount
      apiRef.current.scrollToContent();
    }
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        height,
        width,
        border: "1px solid #e2e8f0",
        borderRadius: "12px",
        overflow: "hidden",
        backgroundColor: "#ffffff",
      }}
    >
      <Excalidraw
        initialData={{
          elements: timelineElements,
          appState: {
            viewBackgroundColor: "#ffffff",
            gridSize: 20,
            gridModeEnabled: true,
            viewModeEnabled: false,
            zenModeEnabled: false,
            zoom: { value: 1 } as any,
            scrollX: 0,
            scrollY: 0,
          },
        }}
        excalidrawAPI={(api) => {
          apiRef.current = api;
        }}
        viewModeEnabled={true}
        UIOptions={{
          canvasActions: {
            export: { saveFileToDisk: true, exportOpts: {} },
            exportToSvg: true,
            exportAsImage: { png: true, svg: true },
            loadScene: true,
            saveAsImage: true,
            saveToActiveFile: true,
          } as any,
        }}
      />
    </div>
  );
}