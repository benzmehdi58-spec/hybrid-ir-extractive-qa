import { useState, useEffect, useRef } from "react";
import { motion } from "motion/react";
import { Search, Check, Loader2, Maximize2, X, ChevronRight, Share2, ShieldAlert, Zap, Cpu, History, ArrowLeft, GitMerge, Settings } from "lucide-react";
import { runPipeline, checkHealth, type PipelineResponse, type RetrievedPassage, runDistilBertQA, type DistilBertResult, runBiLSTMQA, type BiLSTMResult } from "../api/pipeline";

const exampleQueries = [
  "What is the theory of relativity?",
  "When did World War II end?",
  "Who wrote Romeo and Juliet?"
];

// Stable attention weights — pre-computed once to avoid flicker on re-render
const ATTENTION_WEIGHTS = Array.from({ length: 32 }, () => Math.random());

export function DemoMode() {
  const [query, setQuery] = useState("");
  const [isFocused, setIsFocused] = useState(false);
  const [pipelineState, setPipelineState] = useState<"IDLE" | "IR" | "SUMM" | "QA" | "DONE">("IDLE");
  const [activeTab, setActiveTab] = useState<"A" | "B">("B");

  // Real model state
  const [pipelineData, setPipelineData] = useState<PipelineResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [distilbertResult, setDistilbertResult] = useState<DistilBertResult | null>(null);
  const [distilbertLoading, setDistilbertLoading] = useState(false);
  const [bilstmResult, setBilstmResult] = useState<BiLSTMResult | null>(null);
  const [bilstmLoading, setBilstmLoading] = useState(false);

  // Which retrieved passage is expanded in IR
  const [expandedPassage, setExpandedPassage] = useState<number>(0);

  // Which retrieved passage is shown in Summarization
  const [activeSummPassage, setActiveSummPassage] = useState<number>(0);

  // Which retrieved passage produced the winning answer
  const [bestPassageIdx, setBestPassageIdx] = useState<number>(0);

  // Track pending pipeline animation timers so rapid searches don't collide
  const timersRef = useRef<number[]>([]);

  const clearTimers = () => {
    timersRef.current.forEach((id) => window.clearTimeout(id));
    timersRef.current = [];
  };

  // Check backend health on mount and clear timers on unmount
  useEffect(() => {
    checkHealth().then(setBackendOnline);
    return clearTimers;
  }, []);

  const handleSearch = async (q: string) => {
    if (!q.trim()) return;
    clearTimers();
    setQuery(q);
    setApiError(null);
    setPipelineData(null);
    setDistilbertResult(null);
    setBilstmResult(null);
    setExpandedPassage(0);
    setActiveSummPassage(0);
    setBestPassageIdx(0);

    // Start pipeline animation
    setPipelineState("IDLE");
    timersRef.current.push(window.setTimeout(() => setPipelineState("IR"), 100));

    setIsLoading(true);
    try {
      const data = await runPipeline(q, 3);
      setPipelineData(data);

      // Determine which retrieved passage won best_qa (match by object fields or highest confidence)
      let winningIdx = 0;
      if (data.retrieved.length > 0) {
        if (data.best_qa) {
          const matchedIdx = data.retrieved.findIndex(
            (r) =>
              r.qa_distilbert &&
              r.qa_distilbert.answer === data.best_qa!.answer &&
              r.qa_distilbert.start_logit === data.best_qa!.start_logit &&
              r.qa_distilbert.end_logit === data.best_qa!.end_logit
          );
          if (matchedIdx >= 0) winningIdx = matchedIdx;
        }
        setBestPassageIdx(winningIdx);
        setDistilbertResult(data.best_qa ?? data.retrieved[winningIdx].qa_distilbert ?? null);

        const bilstmQA = data.retrieved[winningIdx].qa_bilstm;
        if (bilstmQA) {
          setBilstmResult(bilstmQA);
        } else {
          setBilstmLoading(true);
          try {
            const summary = data.retrieved[winningIdx].summary.summary;
            const qa = await runBiLSTMQA(q, summary);
            setBilstmResult(qa);
          } catch {
            // silently fail
          } finally {
            setBilstmLoading(false);
          }
        }
      }

      // Continue pipeline animation with real data
      timersRef.current.push(window.setTimeout(() => setPipelineState("SUMM"), 400));
      timersRef.current.push(window.setTimeout(() => setPipelineState("QA"), 1800));
      timersRef.current.push(window.setTimeout(() => setPipelineState("DONE"), 3200));
    } catch (err) {
      clearTimers();
      setApiError("Backend offline or error. Check that the FastAPI server is running on port 8000.");
      setPipelineState("IDLE");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex-1 flex overflow-hidden" style={{ position: "relative", zIndex: 1 }}>
      {/* Backend status indicator */}
      {backendOnline === false && (
        <div
          className="fixed top-16 left-1/2 -translate-x-1/2 z-50 px-4 py-2 rounded-full text-xs font-mono flex items-center gap-2"
          style={{
            background: "rgba(239,68,68,0.15)",
            border: "1px solid rgba(239,68,68,0.40)",
            color: "#F87171",
            backdropFilter: "blur(12px)",
          }}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
          Backend offline — run: uvicorn backend.main:app --port 8000
        </div>
      )}
      {apiError && (
        <div
          className="fixed top-16 left-1/2 -translate-x-1/2 z-50 px-4 py-2 rounded-full text-xs font-mono"
          style={{
            background: "rgba(239,68,68,0.15)",
            border: "1px solid rgba(239,68,68,0.40)",
            color: "#F87171",
            backdropFilter: "blur(12px)",
          }}
        >
          {apiError}
        </div>
      )}
      {/* Left Pipeline Area */}
      <div className="flex-1 overflow-y-auto p-8 flex flex-col items-center">
        <div className="w-full max-w-4xl flex flex-col gap-6">

          {/* Question Input Bar */}
          <div
            className="p-4 transition-all duration-300"
            style={{
              background: "rgba(255, 255, 255, 0.04)",
              backdropFilter: "blur(16px)",
              borderRadius: "12px",
              border: isFocused
                ? "1px solid rgba(0, 212, 255, 0.50)"
                : "1px solid rgba(255, 255, 255, 0.08)",
              boxShadow: isFocused
                ? "0 0 0 3px rgba(0, 212, 255, 0.10), 0 0 20px rgba(0, 212, 255, 0.08), 0 8px 32px rgba(0,0,0,0.5)"
                : "0 8px 32px rgba(0,0,0,0.4)",
            }}
          >
            <div className="relative flex items-center">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onFocus={() => setIsFocused(true)}
                onBlur={() => setIsFocused(false)}
                placeholder='Ask anything... e.g. "Who invented the telephone?"'
                className="w-full bg-transparent text-lg font-sans placeholder:text-muted-foreground outline-none border-none py-2 pr-28"
                style={{ color: "#F1F5F9" }}
                onKeyDown={(e) => e.key === "Enter" && handleSearch(query)}
              />
              <button
                onClick={() => handleSearch(query)}
                className="absolute right-0 px-4 py-2 rounded-lg font-medium text-sm flex items-center gap-2 transition-all duration-200"
                style={{
                  background: "linear-gradient(135deg, rgba(0,212,255,0.90), rgba(0,150,220,0.90))",
                  boxShadow: "0 0 20px rgba(0,212,255,0.35), 0 4px 12px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.20)",
                  color: "#020510",
                  fontWeight: 600,
                }}
              >
                <Search className="w-4 h-4" />
                Search
              </button>
            </div>
            <div className="mt-4 flex gap-2 flex-wrap">
              {exampleQueries.map((ex, i) => (
                <button
                  key={i}
                  onClick={() => handleSearch(ex)}
                  className="px-3 py-1.5 rounded-full text-primary text-xs transition-all duration-200"
                  style={{
                    background: "rgba(0, 212, 255, 0.06)",
                    border: "1px solid rgba(0, 212, 255, 0.20)",
                    backdropFilter: "blur(8px)",
                    transition: "all 200ms cubic-bezier(0.34, 1.56, 0.64, 1)",
                  }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLElement).style.background = "rgba(0, 212, 255, 0.14)";
                    (e.currentTarget as HTMLElement).style.borderColor = "rgba(0, 212, 255, 0.50)";
                    (e.currentTarget as HTMLElement).style.boxShadow = "0 0 12px rgba(0, 212, 255, 0.15)";
                    (e.currentTarget as HTMLElement).style.transform = "translateY(-1px)";
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLElement).style.background = "rgba(0, 212, 255, 0.06)";
                    (e.currentTarget as HTMLElement).style.borderColor = "rgba(0, 212, 255, 0.20)";
                    (e.currentTarget as HTMLElement).style.boxShadow = "none";
                    (e.currentTarget as HTMLElement).style.transform = "translateY(0)";
                  }}
                >
                  "{ex}"
                </button>
              ))}
            </div>
          </div>

          {/* Module 1: IR */}
          <ModuleCard
            number="01"
            title="INFORMATION RETRIEVAL"
            accentColor="cyan"
            subLabel="TF-IDF + BM25 Hybrid Retriever · SQuAD 2.0"
            status={pipelineState === "IDLE" ? "IDLE" : pipelineState === "IR" ? "PROCESSING" : "DONE"}
            metrics={[
              { label: "Recall@10", value: "0.8340" },
              { label: "MRR", value: "0.7210" },
              ...(pipelineData ? [{ label: "Time", value: `${pipelineData.retrieval_time_ms}ms` }] : []),
            ]}
          >
            {pipelineState !== "IDLE" ? (
              <div className="grid grid-cols-5 gap-6">
                <div className="col-span-2 flex flex-col gap-3">
                  <div className="text-[10px] font-mono text-muted-foreground" style={{ letterSpacing: "0.15em" }}>METHOD</div>
                  <div className="space-y-3">
                    <ScoreBar label="TF-IDF" percentage={50} accentColor="rgba(0,212,255,0.80)" glowColor="rgba(0,212,255,0.40)" />
                    <ScoreBar label="BM25" percentage={50} accentColor="rgba(0,212,255,0.60)" glowColor="rgba(0,212,255,0.30)" />
                    <div className="text-xs text-muted-foreground font-mono" style={{ fontVariantNumeric: "tabular-nums" }}>α = 0.5 weighted fusion</div>
                  </div>
                </div>
                <div className="col-span-3 flex flex-col gap-2">
                  <div className="text-[10px] font-mono text-muted-foreground mb-1" style={{ letterSpacing: "0.15em" }}>
                    {pipelineData ? `TOP-${pipelineData.retrieved.length} RETRIEVED PASSAGES` : "RETRIEVING..."}
                  </div>
                  {(pipelineData?.retrieved ?? [{}, {}, {}] as any[]).map((result: any, i: number) => (
                    <div
                      key={i}
                      className="flex flex-col gap-1 p-2 transition-all duration-200 cursor-pointer"
                      style={{
                        background: expandedPassage === i ? "rgba(0,212,255,0.06)" : "rgba(255,255,255,0.03)",
                        border: "1px solid rgba(255,255,255,0.05)",
                        borderLeft: `2px solid rgba(0,212,255,${1 - i * 0.2})`,
                        borderRadius: "10px",
                        transition: "all 200ms cubic-bezier(0.34, 1.56, 0.64, 1)",
                      }}
                      onClick={() => setExpandedPassage(expandedPassage === i ? -1 : i)}
                      onMouseEnter={(e) => {
                        (e.currentTarget as HTMLElement).style.boxShadow = "inset 3px 0 0 #00D4FF, 0 0 20px rgba(0,212,255,0.10)";
                        (e.currentTarget as HTMLElement).style.transform = "translateY(-1px)";
                      }}
                      onMouseLeave={(e) => {
                        (e.currentTarget as HTMLElement).style.boxShadow = "none";
                        (e.currentTarget as HTMLElement).style.transform = "translateY(0)";
                      }}
                    >
                      <div className="flex justify-between items-center">
                        <span
                          className="text-xs font-mono text-muted-foreground px-1.5 py-0.5 rounded"
                          style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)" }}
                        >
                          #{result.rank ?? i + 1}
                        </span>
                        {result.score != null && (
                          <div className="flex gap-2 text-xs font-mono" style={{ fontVariantNumeric: "tabular-nums" }}>
                            <span className="text-muted-foreground">TF-IDF: <span style={{ color: "#00D4FF" }}>{result.tfidf_score?.toFixed(3)}</span></span>
                            <span className="text-muted-foreground">BM25: <span style={{ color: "#00D4FF" }}>{result.bm25_score?.toFixed(3)}</span></span>
                            <span className="text-muted-foreground">Hybrid: <span style={{ color: "#00D4FF" }}>{result.score?.toFixed(3)}</span></span>
                          </div>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground mt-1" style={{
                        display: "-webkit-box",
                        WebkitLineClamp: expandedPassage === i ? 999 : 2,
                        WebkitBoxOrient: "vertical" as any,
                        overflow: "hidden",
                      }}>
                        {result.passage ?? "Loading..."}
                      </p>
                      {expandedPassage === i && result.passage && (
                        <div className="text-[10px] font-mono text-muted-foreground mt-1 text-right" style={{ color: "rgba(0,212,255,0.6)" }}>
                          click to collapse ↑
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div
                className="h-32 flex items-center justify-center text-muted-foreground text-sm rounded"
                style={{ border: "1px dashed rgba(255,255,255,0.08)" }}
              >
                Awaiting query...
              </div>
            )}
          </ModuleCard>

          <FlowConnector
            label="Top-k passages →"
            active={pipelineState !== "IDLE"}
            accentRgb="0, 212, 255"
          />

          {/* Module 2: SUMM */}
          <ModuleCard
            number="02"
            title="TEXT SUMMARIZATION"
            accentColor="amber"
            subLabel="Hybrid: TF-IDF + TextRank + Position Weighting · CNN/DailyMail"
            status={pipelineState === "IDLE" || pipelineState === "IR" ? "IDLE" : pipelineState === "SUMM" ? "PROCESSING" : "DONE"}
            metrics={[
              { label: "ROUGE-1", value: "0.4210" },
              { label: "ROUGE-2", value: "0.1890" },
              { label: "ROUGE-L", value: "0.3870" },
              ...(pipelineData ? [{ label: "Time", value: `${pipelineData.summarization_time_ms}ms` }] : []),
            ]}
          >
            {(pipelineState === "SUMM" || pipelineState === "QA" || pipelineState === "DONE") ? (
              <div className="flex flex-col gap-3">
                {/* Passage selector tabs if multiple passages */}
                {pipelineData && pipelineData.retrieved.length > 1 && (
                  <div className="flex gap-2">
                    {pipelineData.retrieved.map((r, i) => (
                      <button
                        key={i}
                        onClick={() => setActiveSummPassage(i)}
                        className="text-[10px] font-mono px-2 py-1 rounded transition-all"
                        style={{
                          background: activeSummPassage === i ? "rgba(245,158,11,0.15)" : "rgba(255,255,255,0.03)",
                          border: activeSummPassage === i ? "1px solid rgba(245,158,11,0.40)" : "1px solid rgba(255,255,255,0.06)",
                          color: activeSummPassage === i ? "#F59E0B" : "#475569",
                        }}
                      >
                        Passage #{r.rank}
                      </button>
                    ))}
                  </div>
                )}

                {/* Main split view */}
                {(() => {
                  const activeResult = pipelineData?.retrieved[activeSummPassage];
                  const summ = activeResult?.summary;

                  return (
                    <div className="grid grid-cols-2 gap-4">
                      {/* Left: Original with strikethrough removed sentences */}
                      <div className="p-3 rounded flex flex-col" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)", borderRadius: "10px" }}>
                        <div className="text-[10px] font-mono text-muted-foreground mb-2 flex justify-between" style={{ letterSpacing: "0.15em" }}>
                          <span>ORIGINAL PASSAGE</span>
                          <span className="px-1.5 py-0.5 rounded" style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)" }}>
                            {summ ? `${summ.total} sentences` : "..."}
                          </span>
                        </div>
                        <div className="text-xs space-y-1 max-h-48 overflow-y-auto">
                          {summ ? (
                            summ.sentences.map((sent, i) => (
                              <span
                                key={i}
                                className="block leading-relaxed"
                                style={
                                  summ.selected_indices.includes(i)
                                    ? { color: "#F1F5F9" }
                                    : { color: "#475569", textDecoration: "line-through", opacity: 0.5 }
                                }
                              >
                                {sent}
                              </span>
                            ))
                          ) : (
                            <span className="text-muted-foreground">Loading...</span>
                          )}
                        </div>
                      </div>

                      {/* Right: Compressed summary with amber highlights */}
                      <div className="p-3 rounded flex flex-col" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)", borderRadius: "10px" }}>
                        <div className="text-[10px] font-mono text-muted-foreground mb-2 flex justify-between" style={{ letterSpacing: "0.15em" }}>
                          <span>COMPRESSED SUMMARY</span>
                          {summ && (
                            <span className="px-1.5 py-0.5 rounded text-secondary" style={{ background: "rgba(245,158,11,0.10)", border: "1px solid rgba(245,158,11,0.20)" }}>
                              {summ.kept} / {summ.total} kept · {summ.compression}% reduction
                            </span>
                          )}
                        </div>
                        <div className="text-xs text-foreground space-y-1 max-h-48 overflow-y-auto">
                          {summ ? (
                            summ.selected_indices.map((idx) => (
                              <span
                                key={idx}
                                className="block p-1 rounded mt-1 leading-relaxed"
                                style={{ background: "rgba(245,158,11,0.10)", border: "1px solid rgba(245,158,11,0.15)" }}
                              >
                                {summ.sentences[idx]}
                              </span>
                            ))
                          ) : (
                            <span className="text-muted-foreground">Awaiting retrieval...</span>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </div>
            ) : (
              <div className="h-32 flex items-center justify-center text-muted-foreground text-sm rounded" style={{ border: "1px dashed rgba(255,255,255,0.08)" }}>
                Awaiting passages...
              </div>
            )}
          </ModuleCard>

          <FlowConnector
            label="Compressed passages →"
            active={pipelineState === "SUMM" || pipelineState === "QA" || pipelineState === "DONE"}
            accentRgb="245, 158, 11"
          />

          {/* Module 3: QA (Dual Model) */}
          <div
            className="p-5 transition-all duration-500 w-full"
            style={{
              background: "rgba(10, 20, 40, 0.45)",
              backdropFilter: "blur(24px) saturate(180%)",
              borderRadius: "16px",
              borderTop: "1px solid rgba(255,255,255,0.13)",
              borderLeft: "1px solid rgba(255,255,255,0.10)",
              border: "1px solid rgba(255,255,255,0.07)",
              boxShadow: (pipelineState === "QA" || pipelineState === "DONE")
                ? (pipelineState === "QA"
                  ? "0 0 0 1px rgba(167,139,250,0.06), 0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08), 0 0 60px rgba(167,139,250,0.20), 0 0 120px rgba(167,139,250,0.08)"
                  : "0 0 0 1px rgba(167,139,250,0.06), 0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08), 0 0 40px rgba(167,139,250,0.08), 0 0 80px rgba(167,139,250,0.04)")
                : "0 0 0 1px rgba(0,212,255,0.04), 0 8px 32px rgba(0,0,0,0.6), 0 2px 8px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08)",
              transform: "translateZ(0)",
            }}
          >
            <div className="flex justify-between items-center mb-4">
              <div className="flex items-center gap-3">
                <div
                  className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-mono font-bold"
                  style={(pipelineState === "QA" || pipelineState === "DONE")
                    ? { background: "rgba(167,139,250,0.20)", color: "#A78BFA", border: "1px solid rgba(167,139,250,0.40)" }
                    : { background: "rgba(255,255,255,0.06)", color: "#475569" }
                  }
                >
                  03
                </div>
                <h3
                  className="font-mono text-sm"
                  style={{
                    color: (pipelineState === "QA" || pipelineState === "DONE") ? "#A78BFA" : "#F1F5F9",
                    letterSpacing: "0.15em",
                  }}
                >
                  QUESTION ANSWERING
                </h3>
              </div>
              <StatusBadge status={pipelineState === "IDLE" || pipelineState === "IR" || pipelineState === "SUMM" ? "IDLE" : pipelineState === "QA" ? "PROCESSING" : "DONE"} />
            </div>
            <div className="text-xs text-muted-foreground mb-6 font-mono" style={{ letterSpacing: "0.05em" }}>
              Dual Model Comparison · SQuAD 2.0 + HotpotQA
            </div>

            {(pipelineState === "QA" || pipelineState === "DONE") ? (
              <div className="flex flex-col gap-6 relative">
                <div
                  className="grid grid-cols-2 gap-0 relative overflow-hidden"
                  style={{
                    background: "rgba(255,255,255,0.02)",
                    border: "1px solid rgba(255,255,255,0.06)",
                    borderRadius: "12px",
                  }}
                >
                  {/* Left Column: BiLSTM */}
                  <div className="p-5 flex flex-col h-full" style={{ borderRight: "1px solid rgba(255,255,255,0.06)" }}>
                    <div className="mb-4">
                      <div className="text-[10px] font-mono text-muted-foreground mb-1" style={{ letterSpacing: "0.15em" }}>MODEL A</div>
                      <div className="flex items-center justify-between mb-2">
                        <div className="font-display font-medium text-lg text-foreground" style={{ letterSpacing: "-0.02em" }}>BiLSTM + Attention</div>
                        <div
                          className="text-[10px] font-mono px-2 py-0.5 rounded-full"
                          style={{ border: "1px solid rgba(245,158,11,0.50)", color: "#F59E0B", background: "rgba(245,158,11,0.10)" }}
                        >
                          From Scratch
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-2 text-[10px] font-mono text-muted-foreground">
                        <span>GloVe 100d</span><span>·</span><span>2-layer BiLSTM</span><span>·</span><span>BiDAF Attention</span>
                      </div>
                    </div>

                    <div className="flex-1 space-y-4">
                      {bilstmLoading ? (
                        <div className="flex items-center gap-2 text-muted-foreground text-xs font-mono py-4">
                          <div className="w-1.5 h-1.5 rounded-full bg-[#F59E0B] animate-pulse" />
                          Running BiLSTM inference...
                        </div>
                      ) : bilstmResult ? (
                        <div className="space-y-3">
                          {/* Real context token chips */}
                          <div className="flex flex-wrap gap-1 max-h-24 overflow-y-auto">
                            {bilstmResult.ctx_tokens.map((token, i) => {
                              const isAnswerSpan = i >= bilstmResult.start_token && i <= bilstmResult.end_token && bilstmResult.answerable;
                              const isStart = i === bilstmResult.start_token && bilstmResult.answerable;
                              const isEnd   = i === bilstmResult.end_token   && bilstmResult.answerable;
                              return (
                                <span
                                  key={i}
                                  className="text-xs px-1.5 py-0.5 font-mono flex items-center gap-0.5"
                                  style={isAnswerSpan ? {
                                    background: "rgba(245,158,11,0.18)",
                                    border: "1px solid rgba(245,158,11,0.45)",
                                    boxShadow: "0 0 8px rgba(245,158,11,0.20)",
                                    color: "#FCD34D",
                                    borderRadius: "4px",
                                  } : {
                                    background: "rgba(255,255,255,0.04)",
                                    border: "1px solid rgba(255,255,255,0.07)",
                                    color: "#94A3B8",
                                    borderRadius: "4px",
                                  }}
                                >
                                  {isStart && <span style={{ color: "#F59E0B", fontSize: "8px" }}>▶</span>}
                                  {token}
                                  {isEnd && <span style={{ color: "#F59E0B", fontSize: "8px" }}>◀</span>}
                                </span>
                              );
                            })}
                          </div>

                          {/* Start / End positions */}
                          <div className="flex justify-between items-center text-[10px] font-mono text-muted-foreground"
                            style={{ fontVariantNumeric: "tabular-nums" }}>
                            <span>Start: <span style={{ color: "#F59E0B" }}>token {bilstmResult.start_token}</span></span>
                            <span>End: <span style={{ color: "#F59E0B" }}>token {bilstmResult.end_token}</span></span>
                          </div>
                        </div>
                      ) : (
                        /* Idle placeholder — same chips as before, dimmed */
                        <div className="flex flex-wrap gap-1 opacity-30">
                          {["context","tokens","will","appear","here"].map((t, i) => (
                            <span key={i} className="text-xs px-1.5 py-0.5 font-mono rounded"
                              style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", color: "#475569" }}>
                              {t}
                            </span>
                          ))}
                        </div>
                      )}

                      {/* Attention heatmap */}
                      <div className="mt-2 pt-4" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                        <div className="text-[10px] font-mono text-muted-foreground mb-2" style={{ letterSpacing: "0.15em" }}>
                          ATTENTION WEIGHTS (BiDAF)
                        </div>
                        {bilstmResult?.attention_weights?.length ? (
                          <div>
                            {/* Q-token column labels */}
                            <div className="flex gap-px mb-px pl-10">
                              {bilstmResult.q_tokens.slice(0, bilstmResult.attention_weights[0]?.length ?? 0).map((qt, j) => (
                                <div key={j} className="w-4 text-[7px] font-mono text-center truncate"
                                  style={{ color: "#475569", width: "16px", flexShrink: 0 }}>
                                  {qt.slice(0, 3)}
                                </div>
                              ))}
                            </div>
                            {bilstmResult.attention_weights.map((row, i) => (
                              <div key={i} className="flex items-center gap-px mb-px">
                                {/* Ctx token row label */}
                                <div className="text-[7px] font-mono text-right truncate mr-1"
                                  style={{ color: "#475569", width: "32px", flexShrink: 0 }}>
                                  {(bilstmResult.ctx_tokens[i] ?? "").slice(0, 5)}
                                </div>
                                {row.map((w, j) => (
                                  <div
                                    key={j}
                                    className="w-4 h-4 rounded-sm"
                                    title={`ctx[${i}]↔q[${j}]: ${w.toFixed(3)}`}
                                    style={{
                                      width: "16px",
                                      height: "16px",
                                      flexShrink: 0,
                                      borderRadius: "2px",
                                      backgroundColor: `rgba(245,158,11,${Math.min(1, w * 0.80 + 0.04)})`,
                                      boxShadow: w > 0.6 ? "inset 0 0 4px rgba(245,158,11,0.5)" : "none",
                                    }}
                                  />
                                ))}
                              </div>
                            ))}
                          </div>
                        ) : (
                          /* Placeholder grid while idle */
                          <div
                            className="grid grid-cols-8 p-1"
                            style={{
                              border: "1px solid rgba(245,158,11,0.15)",
                              borderRadius: "8px",
                              boxShadow: "0 0 20px rgba(245,158,11,0.06)",
                              gap: "1px",
                              background: "rgba(255,255,255,0.02)",
                            }}
                          >
                            {ATTENTION_WEIGHTS.map((w, i) => (
                              <div key={i} className="w-4 h-4"
                                style={{ borderRadius: "2px", backgroundColor: `rgba(245,158,11,${w * 0.30 + 0.02})` }} />
                            ))}
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="mt-4 pt-4" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                      <div
                        className="flex gap-2 items-center px-2 py-1 rounded text-xs font-mono"
                        style={{
                          background: "rgba(245,158,11,0.08)",
                          border: "1px solid rgba(245,158,11,0.25)",
                          color: "#F59E0B",
                          fontVariantNumeric: "tabular-nums",
                        }}
                      >
                        <span>EM: ~10%</span><span>·</span><span>F1: ~17%</span><span>·</span><span>HotpotQA: low</span>
                      </div>
                    </div>
                  </div>

                  {/* VS center pill */}
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-10">
                    <div
                      className="text-[10px] font-mono font-bold px-2 py-1 rounded-full"
                      style={{
                        background: "rgba(10,20,40,0.80)",
                        border: "1px solid rgba(255,255,255,0.12)",
                        color: "#F1F5F9",
                        backdropFilter: "blur(8px)",
                      }}
                    >
                      VS
                    </div>
                  </div>

                  {/* Right Column: DistilBERT */}
                  <div className="p-5 flex flex-col h-full">
                    <div className="mb-4">
                      <div className="text-[10px] font-mono text-muted-foreground mb-1" style={{ letterSpacing: "0.15em" }}>MODEL B</div>
                      <div className="flex items-center justify-between mb-2">
                        <div className="font-display font-medium text-lg text-foreground" style={{ letterSpacing: "-0.02em" }}>DistilBERT Fine-tuned</div>
                        <div
                          className="text-[10px] font-mono px-2 py-0.5 rounded-full"
                          style={{ border: "1px solid rgba(0,212,255,0.50)", color: "#00D4FF", background: "rgba(0,212,255,0.10)" }}
                        >
                          Pretrained + Fine-tuned
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-2 text-[10px] font-mono text-muted-foreground">
                        <span>distilbert-base-uncased</span><span>·</span><span>384 tokens</span><span>·</span><span>Sliding Window</span>
                      </div>
                    </div>

                    <div className="flex-1 space-y-4">
                      {/* Token visualization — real DistilBERT tokens */}
                      {distilbertLoading ? (
                        <div className="flex items-center gap-2 text-muted-foreground text-xs font-mono py-4">
                          <div className="w-1.5 h-1.5 rounded-full bg-[#00D4FF] animate-pulse" />
                          Running DistilBERT inference...
                        </div>
                      ) : distilbertResult ? (
                        <div className="space-y-3">
                          {/* Token chips */}
                          <div className="flex flex-wrap gap-1 max-h-36 overflow-y-auto">
                            {distilbertResult.tokens.map((token, i) => {
                              const ttype = distilbertResult.token_type[i];
                              const isAnswerSpan = i >= distilbertResult.start_token && i <= distilbertResult.end_token && distilbertResult.answerable;
                              const isStart = i === distilbertResult.start_token && distilbertResult.answerable;
                              const isEnd   = i === distilbertResult.end_token   && distilbertResult.answerable;

                              let chipStyle: React.CSSProperties;
                              if (isAnswerSpan) {
                                chipStyle = {
                                  background: "rgba(0,212,255,0.18)",
                                  border: "1px solid rgba(0,212,255,0.45)",
                                  boxShadow: "0 0 8px rgba(0,212,255,0.20)",
                                  color: "#67E8F9",
                                  borderRadius: "4px",
                                };
                              } else if (ttype === "cls" || ttype === "sep") {
                                chipStyle = {
                                  background: "rgba(255,255,255,0.04)",
                                  border: "1px solid rgba(255,255,255,0.10)",
                                  color: "#64748B",
                                  borderRadius: "4px",
                                  fontStyle: "italic",
                                };
                              } else if (ttype === "question") {
                                chipStyle = {
                                  background: "rgba(59,130,246,0.12)",
                                  border: "1px solid rgba(59,130,246,0.25)",
                                  color: "#93C5FD",
                                  borderRadius: "4px",
                                };
                              } else if (ttype === "pad") {
                                chipStyle = { display: "none" };
                              } else {
                                chipStyle = {
                                  background: "rgba(255,255,255,0.03)",
                                  border: "1px solid rgba(255,255,255,0.06)",
                                  color: "#475569",
                                  borderRadius: "4px",
                                };
                              }

                              return (
                                <span
                                  key={i}
                                  className="text-[10px] px-1.5 py-0.5 font-mono flex items-center gap-0.5"
                                  style={chipStyle}
                                >
                                  {isStart && <span style={{ color: "#00D4FF", fontSize: "7px" }}>▶</span>}
                                  {token}
                                  {isEnd && <span style={{ color: "#00D4FF", fontSize: "7px" }}>◀</span>}
                                </span>
                              );
                            })}
                          </div>

                          <div className="text-[10px] font-mono text-muted-foreground flex gap-3">
                            <span>
                              <span style={{ color: "#475569" }}>Start: </span>
                              <span style={{ color: "#00D4FF", fontVariantNumeric: "tabular-nums" }}>
                                token {distilbertResult.start_token}
                              </span>
                            </span>
                            <span>·</span>
                            <span>
                              <span style={{ color: "#475569" }}>End: </span>
                              <span style={{ color: "#00D4FF", fontVariantNumeric: "tabular-nums" }}>
                                token {distilbertResult.end_token}
                              </span>
                            </span>
                            <span>·</span>
                            <span className="text-[10px]" style={{ color: "#475569" }}>STRIDE=128</span>
                          </div>

                          {/* Answer */}
                          <div className="mt-2 pt-3" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                            <div
                              className="text-xl font-bold text-foreground mb-2"
                              style={{ letterSpacing: "-0.02em", textShadow: "0 0 20px rgba(255,255,255,0.15)" }}
                            >
                              {distilbertResult.answerable
                                ? `"${distilbertResult.answer}"`
                                : <span style={{ color: "#475569", fontStyle: "italic" }}>Unanswerable</span>
                              }
                            </div>
                            <div className="text-[10px] font-mono flex gap-2 flex-wrap" style={{ fontVariantNumeric: "tabular-nums" }}>
                              <span style={{ color: distilbertResult.answerable ? "#10B981" : "#EF4444" }}>
                                {distilbertResult.answerable ? "Answerable ✓" : "Unanswerable ✗"}
                              </span>
                              <span className="text-muted-foreground">·</span>
                              <span className="text-muted-foreground">
                                Start logit: <span style={{ color: "#00D4FF" }}>{distilbertResult.start_logit.toFixed(2)}</span>
                              </span>
                              <span className="text-muted-foreground">·</span>
                              <span className="text-muted-foreground">
                                End logit: <span style={{ color: "#00D4FF" }}>{distilbertResult.end_logit.toFixed(2)}</span>
                              </span>
                              <span className="text-muted-foreground">·</span>
                              <span className="text-muted-foreground">
                                CLS: <span style={{ color: "#94A3B8" }}>{distilbertResult.cls_logit.toFixed(2)}</span>
                              </span>
                            </div>
                          </div>

                          {/* Top-5 start logits mini bar chart */}
                          <div className="mt-2 pt-3" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                            <div className="text-[10px] font-mono text-muted-foreground mb-2" style={{ letterSpacing: "0.12em" }}>
                              START LOGITS — TOP 5 CANDIDATES
                            </div>
                            <div className="space-y-1">
                              {distilbertResult.start_logits_top5.map(([tokenIdx, logit], i) => {
                                const maxLogit = distilbertResult.start_logits_top5[0][1];
                                const pct = maxLogit > 0 ? Math.max(5, (logit / maxLogit) * 100) : 5;
                                const isSelected = tokenIdx === distilbertResult.start_token;
                                return (
                                  <div key={i} className="flex items-center gap-2">
                                    <span
                                      className="text-[9px] font-mono w-20 truncate text-right"
                                      style={{ color: isSelected ? "#00D4FF" : "#475569" }}
                                    >
                                      {distilbertResult.tokens[tokenIdx] ?? `t${tokenIdx}`}
                                    </span>
                                    <div
                                      className="flex-1 h-1 rounded-full overflow-hidden"
                                      style={{ background: "rgba(255,255,255,0.04)" }}
                                    >
                                      <div
                                        className="h-full rounded-full"
                                        style={{
                                          width: `${pct}%`,
                                          background: isSelected
                                            ? "linear-gradient(90deg, #0E4A6E, #00D4FF)"
                                            : "rgba(255,255,255,0.12)",
                                          boxShadow: isSelected ? "0 0 6px rgba(0,212,255,0.40)" : "none",
                                        }}
                                      />
                                    </div>
                                    <span
                                      className="text-[9px] font-mono w-10 text-right"
                                      style={{ color: isSelected ? "#00D4FF" : "#475569", fontVariantNumeric: "tabular-nums" }}
                                    >
                                      {logit.toFixed(1)}
                                    </span>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        </div>
                      ) : (
                        /* Fallback — same mock layout as before while IDLE */
                        <div className="flex flex-wrap gap-1 font-mono text-[10px] mb-2 opacity-40">
                          {["[CLS]", "question", "tokens", "[SEP]", "context", "tokens", "[SEP]"].map((t, i) => (
                            <span
                              key={i}
                              className="px-1.5 py-0.5 rounded"
                              style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", color: "#475569" }}
                            >
                              {t}
                            </span>
                          ))}
                          <div className="w-full mt-2 text-muted-foreground text-[10px]">Awaiting summary...</div>
                        </div>
                      )}
                    </div>

                    <div className="mt-4 pt-4" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                      <div
                        className="flex gap-2 items-center px-2 py-1 rounded text-xs font-mono flex-wrap"
                        style={{
                          background: "rgba(0,212,255,0.08)",
                          border: "1px solid rgba(0,212,255,0.25)",
                          color: "#00D4FF",
                          fontVariantNumeric: "tabular-nums",
                        }}
                      >
                        <span>EM: ~79%</span><span>·</span><span>F1: ~83%</span><span>·</span><span>HotpotQA: moderate</span>
                        {distilbertResult?.inference_time_ms != null && (
                          <><span>·</span><span style={{ color: "#94A3B8" }}>Infer: {distilbertResult.inference_time_ms}ms</span></>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Neon comparison bars */}
                <div
                  className="p-4 text-xs font-mono overflow-hidden"
                  style={{
                    background: "rgba(255,255,255,0.03)",
                    border: "1px solid rgba(255,255,255,0.06)",
                    borderRadius: "12px",
                    backdropFilter: "blur(12px)",
                  }}
                >
                  <div className="text-[10px] text-muted-foreground mb-3" style={{ letterSpacing: "0.15em" }}>MODEL COMPARISON</div>
                  <div className="grid grid-cols-[100px_1fr_40px] gap-x-4 gap-y-3 items-center">
                    <div className="text-muted-foreground">Exact Match</div>
                    <div className="flex flex-col gap-1.5">
                      <NeonBar percentage={10} accentGradient="linear-gradient(90deg, #92400E, #F59E0B)" glowColor="rgba(245,158,11,0.40)" />
                      <NeonBar percentage={79} accentGradient="linear-gradient(90deg, #0E4A6E, #00D4FF)" glowColor="rgba(0,212,255,0.40)" />
                    </div>
                    <div className="flex flex-col gap-1.5 text-[10px] text-right" style={{ fontVariantNumeric: "tabular-nums" }}>
                      <span style={{ color: "#F59E0B" }}>~10%</span>
                      <span style={{ color: "#00D4FF" }}>~79%</span>
                    </div>

                    <div className="text-muted-foreground">F1 Score</div>
                    <div className="flex flex-col gap-1.5">
                      <NeonBar percentage={17} accentGradient="linear-gradient(90deg, #92400E, #F59E0B)" glowColor="rgba(245,158,11,0.40)" />
                      <NeonBar percentage={83} accentGradient="linear-gradient(90deg, #0E4A6E, #00D4FF)" glowColor="rgba(0,212,255,0.40)" />
                    </div>
                    <div className="flex flex-col gap-1.5 text-[10px] text-right" style={{ fontVariantNumeric: "tabular-nums" }}>
                      <span style={{ color: "#F59E0B" }}>~17%</span>
                      <span style={{ color: "#00D4FF" }}>~83%</span>
                    </div>

                    <div className="text-muted-foreground">HotpotQA EM</div>
                    <div className="flex flex-col gap-1.5">
                      <NeonBar percentage={10} accentGradient="linear-gradient(90deg, #92400E, #F59E0B)" glowColor="rgba(245,158,11,0.40)" />
                      <NeonBar percentage={50} accentGradient="linear-gradient(90deg, #0E4A6E, #00D4FF)" glowColor="rgba(0,212,255,0.40)" />
                    </div>
                    <div className="flex flex-col gap-1.5 text-[10px] text-right">
                      <span style={{ color: "#F59E0B" }}>low</span>
                      <span style={{ color: "#00D4FF" }}>mod</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-32 flex items-center justify-center text-muted-foreground text-sm rounded" style={{ border: "1px dashed rgba(255,255,255,0.08)" }}>
                Awaiting summary...
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Right Answer Panel — elevated glass */}
      <div
        className="w-[400px] flex flex-col overflow-hidden shrink-0"
        style={{
          background: "linear-gradient(180deg, rgba(0,212,255,0.04) 0%, rgba(12,24,50,0.55) 40%)",
          backdropFilter: "blur(32px) saturate(200%)",
          borderLeft: "1px solid rgba(255,255,255,0.08)",
          borderTop: "1px solid rgba(0, 212, 255, 0.20)",
          boxShadow: "0 0 0 1px rgba(0,212,255,0.06), -8px 0 48px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.12)",
        }}
      >
        <div className="p-6 flex-shrink-0" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
          <h2 className="font-display text-xl tracking-tight text-foreground mb-4 uppercase" style={{ letterSpacing: "-0.02em" }}>Answer</h2>
          <div className="min-h-[120px]">
            {pipelineState === "DONE" ? (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-4"
              >
                {/* Model A Answer */}
                <div
                  className="rounded p-3"
                  style={{
                    background: "rgba(255,255,255,0.03)",
                    border: "1px solid rgba(255,255,255,0.05)",
                    borderLeft: "2px solid #F59E0B",
                    borderRadius: "10px",
                  }}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-[10px] font-mono text-muted-foreground" style={{ letterSpacing: "0.10em" }}>MODEL A</span>
                    <span
                      className="text-[10px] font-mono px-1.5 rounded"
                      style={{ color: "#F59E0B", background: "rgba(245,158,11,0.12)", border: "1px solid rgba(245,158,11,0.25)" }}
                    >
                      BiLSTM
                    </span>
                  </div>
                  <div className="font-display text-lg text-foreground leading-tight" style={{ letterSpacing: "-0.02em" }}>
                    {bilstmResult?.answerable
                      ? `"${bilstmResult.answer}"`
                      : bilstmResult && !bilstmResult.answerable
                        ? <span style={{ color: "#475569", fontStyle: "italic", fontSize: "14px" }}>Unanswerable</span>
                        : <span style={{ color: "#475569", fontStyle: "italic", fontSize: "14px" }}>Awaiting result...</span>
                    }
                  </div>
                </div>

                {/* Model B Answer */}
                <div
                  className="rounded p-3"
                  style={{
                    background: "rgba(255,255,255,0.03)",
                    border: "1px solid rgba(255,255,255,0.05)",
                    borderLeft: "2px solid #00D4FF",
                    borderRadius: "10px",
                  }}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-[10px] font-mono text-muted-foreground" style={{ letterSpacing: "0.10em" }}>MODEL B</span>
                    <span
                      className="text-[10px] font-mono px-1.5 rounded"
                      style={{ color: "#00D4FF", background: "rgba(0,212,255,0.12)", border: "1px solid rgba(0,212,255,0.25)" }}
                    >
                      DistilBERT
                    </span>
                  </div>
                  <div className="font-display text-lg text-foreground leading-tight" style={{ letterSpacing: "-0.02em", textShadow: "0 0 20px rgba(255,255,255,0.15)" }}>
                    {distilbertResult?.answerable
                      ? `"${distilbertResult.answer}"`
                      : distilbertResult && !distilbertResult.answerable
                        ? <span style={{ color: "#475569", fontStyle: "italic", fontSize: "14px" }}>Unanswerable</span>
                        : <span style={{ color: "#475569", fontStyle: "italic", fontSize: "14px" }}>Awaiting result...</span>
                    }
                  </div>
                </div>

                {bilstmResult && distilbertResult && (
                  <div className="flex justify-center mt-2">
                    {bilstmResult.answerable && distilbertResult.answerable ? (
                      bilstmResult.answer.toLowerCase().trim() === distilbertResult.answer.toLowerCase().trim() ? (
                        <span className="px-3 py-1 rounded-full text-xs font-mono font-bold flex items-center gap-1"
                          style={{ background: "rgba(16,185,129,0.12)", border: "1px solid rgba(16,185,129,0.35)", color: "#10B981", boxShadow: "0 0 12px rgba(16,185,129,0.15)" }}>
                          <Check className="w-3 h-3" /> MATCH
                        </span>
                      ) : (
                        <span className="px-3 py-1 rounded-full text-xs font-mono font-bold flex items-center gap-1"
                          style={{ background: "rgba(245,158,11,0.12)", border: "1px solid rgba(245,158,11,0.35)", color: "#F59E0B" }}>
                          ✗ DIFFER
                        </span>
                      )
                    ) : !bilstmResult.answerable && !distilbertResult.answerable ? (
                      <span className="px-3 py-1 rounded-full text-xs font-mono font-bold"
                        style={{ background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.35)", color: "#F87171" }}>
                        BOTH UNANSWERABLE
                      </span>
                    ) : (
                      <span className="px-3 py-1 rounded-full text-xs font-mono font-bold"
                        style={{ background: "rgba(245,158,11,0.12)", border: "1px solid rgba(245,158,11,0.35)", color: "#F59E0B" }}>
                        ✗ DIFFER
                      </span>
                    )}
                  </div>
                )}
              </motion.div>
            ) : pipelineState !== "IDLE" ? (
              <div className="flex items-center gap-2 text-muted-foreground h-full">
                <div className="w-2 h-2 rounded-full bg-primary animate-ping" />
                <span className="font-mono text-sm">Processing pipeline...</span>
              </div>
            ) : (
              <div className="text-muted-foreground text-sm italic h-full flex items-center">Waiting for query...</div>
            )}
          </div>
        </div>

        {pipelineState === "DONE" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="p-6 flex-1 overflow-y-auto space-y-6"
          >
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <div className="text-[10px] font-mono text-muted-foreground" style={{ letterSpacing: "0.15em" }}>
                  SOURCE PASSAGE{pipelineData?.retrieved[bestPassageIdx] ? ` · RANK #${pipelineData.retrieved[bestPassageIdx].rank}` : ""}
                </div>
                <div
                  className="flex rounded p-0.5"
                  style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)" }}
                >
                  <button
                    className="px-2 py-0.5 text-[10px] font-mono rounded transition-all"
                    style={{
                      background: activeTab === "A" ? "rgba(245,158,11,0.15)" : "transparent",
                      color: activeTab === "A" ? "#F59E0B" : "#475569",
                      border: activeTab === "A" ? "1px solid rgba(245,158,11,0.30)" : "1px solid transparent",
                    }}
                    onClick={() => setActiveTab("A")}
                  >
                    Model A
                  </button>
                  <button
                    className="px-2 py-0.5 text-[10px] font-mono rounded transition-all"
                    style={{
                      background: activeTab === "B" ? "rgba(0,212,255,0.15)" : "transparent",
                      color: activeTab === "B" ? "#00D4FF" : "#475569",
                      border: activeTab === "B" ? "1px solid rgba(0,212,255,0.30)" : "1px solid transparent",
                    }}
                    onClick={() => setActiveTab("B")}
                  >
                    Model B
                  </button>
                </div>
              </div>
              <div
                className="rounded-lg p-4 text-sm leading-relaxed text-muted-foreground"
                style={{
                  background: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.05)",
                  borderRadius: "10px",
                }}
              >
                {pipelineData && activeTab === "A" && bilstmResult?.answerable ? (
                  (() => {
                    const passage  = pipelineData.retrieved[bestPassageIdx]?.summary?.summary ?? "";
                    const answer   = bilstmResult.answer;
                    const answerIdx = passage.toLowerCase().indexOf(answer.toLowerCase());
                    if (answerIdx === -1) return <span>{passage}</span>;
                    return (
                      <>
                        {passage.slice(0, answerIdx)}
                        <span style={{
                          background: "rgba(245,158,11,0.20)",
                          borderBottom: "2px solid #F59E0B",
                          color: "#FCD34D",
                          padding: "0 2px",
                        }}>
                          {passage.slice(answerIdx, answerIdx + answer.length)}
                        </span>
                        {passage.slice(answerIdx + answer.length)}
                      </>
                    );
                  })()
                ) : pipelineData && activeTab === "B" && distilbertResult?.answerable ? (
                  (() => {
                    const passage = pipelineData.retrieved[bestPassageIdx]?.passage ?? "";
                    const answer = distilbertResult.answer;
                    const answerIdx = passage.toLowerCase().indexOf(answer.toLowerCase());
                    if (answerIdx === -1) return <span>{passage}</span>;
                    
                    return (
                      <>
                        {passage.slice(0, answerIdx)}
                        <span style={{ 
                          background: "rgba(147,51,234,0.20)", 
                          borderBottom: "2px solid #A855F7",
                          color: "#D8B4FE",
                          padding: "0 2px",
                        }}>
                          {passage.slice(answerIdx, answerIdx + answer.length)}
                        </span>
                        {passage.slice(answerIdx + answer.length)}
                      </>
                    );
                  })()
                ) : (
                  (activeTab === "B"
                    ? pipelineData?.retrieved[bestPassageIdx]?.passage
                    : pipelineData?.retrieved[bestPassageIdx]?.summary?.summary) ||
                  "Submit a query to see the retrieved and summarized passage."
                )}
              </div>
            </div>

            <div
              className="rounded-lg overflow-hidden"
              style={{
                background: "rgba(255,255,255,0.02)",
                border: "1px solid rgba(255,255,255,0.05)",
                borderRadius: "10px",
              }}
            >
              <div
                className="p-3 text-xs font-mono text-muted-foreground uppercase"
                style={{
                  borderBottom: "1px solid rgba(255,255,255,0.05)",
                  background: "rgba(255,255,255,0.02)",
                  letterSpacing: "0.15em",
                }}
              >
                Pipeline Metrics
              </div>
              <div className="p-4 space-y-3 font-mono text-sm">
                {[
                  { label: "IR Recall@10", value: "0.8340" },
                  { label: "IR MRR", value: "0.7210" },
                  { label: "ROUGE-1", value: "0.4210" },
                  { label: "ROUGE-2", value: "0.1890" },
                ].map((m) => (
                  <div key={m.label} className="flex justify-between items-center text-sm">
                    <span className="text-muted-foreground">{m.label}</span>
                    <span
                      className="px-2 py-0.5"
                      style={{
                        background: "rgba(255,255,255,0.04)",
                        border: "1px solid rgba(255,255,255,0.08)",
                        borderRadius: "6px",
                        fontVariantNumeric: "tabular-nums",
                        color: "#F1F5F9",
                        backdropFilter: "blur(8px)",
                      }}
                    >
                      {m.value}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-auto pt-4" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
              <div className="flex justify-between text-xs font-mono text-muted-foreground mb-2">
                <span>Processing Time</span>
                <span className="text-foreground" style={{ fontVariantNumeric: "tabular-nums" }}>
                  {pipelineData ? `${pipelineData.total_time_ms}ms Total` : "91ms Total"}
                </span>
              </div>
              <div className="h-1.5 flex rounded-full overflow-hidden w-full" style={{ background: "rgba(255,255,255,0.05)" }}>
                <div style={{ background: "#00D4FF", width: "46%", boxShadow: "0 0 8px rgba(0,212,255,0.5)" }} title="IR: 42ms" />
                <div style={{ background: "#F59E0B", width: "20%", boxShadow: "0 0 8px rgba(245,158,11,0.5)" }} title="Summ: 18ms" />
                <div style={{ background: "#A78BFA", width: "34%", boxShadow: "0 0 8px rgba(167,139,250,0.5)" }} title="QA: 31ms" />
              </div>
              <div className="flex justify-between text-[10px] font-mono text-muted-foreground mt-1" style={{ fontVariantNumeric: "tabular-nums" }}>
                <span>IR: {pipelineData ? `${pipelineData.retrieval_time_ms}ms` : "42ms"}</span>
                <span>Summ: {pipelineData ? `${pipelineData.summarization_time_ms}ms` : "18ms"}</span>
                <span>QA: {distilbertResult?.inference_time_ms != null ? `${distilbertResult.inference_time_ms}ms` : pipelineData?.qa_time_ms != null ? `${pipelineData.qa_time_ms}ms` : "31ms"}</span>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}

// --- Subcomponents ---

interface ModuleCardProps {
  number: string;
  title: string;
  accentColor: "cyan" | "amber" | "purple";
  subLabel: string;
  status: "IDLE" | "PROCESSING" | "DONE";
  metrics: { label: string; value: string }[];
  children: React.ReactNode;
}

const ACCENT = {
  cyan:   { rgb: "0,212,255",   hex: "#00D4FF", textColor: "#00D4FF", borderColor: "rgba(0,212,255,0.50)"   },
  amber:  { rgb: "245,158,11",  hex: "#F59E0B", textColor: "#F59E0B", borderColor: "rgba(245,158,11,0.50)"  },
  purple: { rgb: "167,139,250", hex: "#A78BFA", textColor: "#A78BFA", borderColor: "rgba(167,139,250,0.50)" },
};

function ModuleCard({ number, title, accentColor, subLabel, status, metrics, children }: ModuleCardProps) {
  const ac = ACCENT[accentColor];
  const isActive = status === "PROCESSING" || status === "DONE";
  const isProcessing = status === "PROCESSING";

  const activeGlow = isProcessing
    ? `0 0 60px rgba(${ac.rgb},0.20), 0 0 120px rgba(${ac.rgb},0.08)`
    : `0 0 40px rgba(${ac.rgb},0.08), 0 0 80px rgba(${ac.rgb},0.04)`;

  const boxShadow = isActive
    ? `0 0 0 1px rgba(${ac.rgb},0.06), 0 8px 32px rgba(0,0,0,0.6), 0 2px 8px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08), ${activeGlow}`
    : "0 0 0 1px rgba(0,212,255,0.04), 0 8px 32px rgba(0,0,0,0.6), 0 2px 8px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08)";

  return (
    <div
      className="p-5 transition-all duration-500 w-full"
      style={{
        background: "rgba(10, 20, 40, 0.45)",
        backdropFilter: "blur(24px) saturate(180%)",
        borderRadius: "16px",
        border: isActive ? `1px solid rgba(${ac.rgb},0.20)` : "1px solid rgba(255,255,255,0.07)",
        borderTop: isActive ? `1px solid rgba(${ac.rgb},0.30)` : "1px solid rgba(255,255,255,0.13)",
        borderLeft: "1px solid rgba(255,255,255,0.10)",
        boxShadow,
        transform: "translateZ(0)",
      }}
    >
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center gap-3">
          <div
            className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-mono font-bold"
            style={isActive
              ? { background: `rgba(${ac.rgb},0.20)`, color: ac.hex, border: `1px solid rgba(${ac.rgb},0.40)` }
              : { background: "rgba(255,255,255,0.06)", color: "#475569" }
            }
          >
            {number}
          </div>
          <h3
            className="font-mono text-sm"
            style={{
              color: isActive ? ac.hex : "#F1F5F9",
              letterSpacing: "0.15em",
            }}
          >
            {title}
          </h3>
        </div>
        <StatusBadge status={status} />
      </div>
      <div className="text-xs text-muted-foreground mb-6 font-mono">{subLabel}</div>
      <div className="mb-6">{children}</div>
      <div className="flex flex-wrap gap-3 pt-4" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
        {metrics.map((m, i) => (
          <div
            key={i}
            className="flex gap-2 items-center px-2 py-1 text-xs font-mono"
            style={{
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: "6px",
              backdropFilter: "blur(8px)",
              fontVariantNumeric: "tabular-nums",
            }}
          >
            <span className="text-muted-foreground">{m.label}:</span>
            <span className="text-foreground">{m.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: "IDLE" | "PROCESSING" | "DONE" }) {
  if (status === "IDLE") return (
    <span
      className="text-[10px] font-mono px-2 py-0.5 rounded-full"
      style={{
        background: "rgba(255,255,255,0.04)",
        border: "1px solid rgba(255,255,255,0.10)",
        color: "#475569",
      }}
    >
      IDLE
    </span>
  );

  if (status === "PROCESSING") return (
    <div className="relative flex items-center">
      <span
        className="text-[10px] font-mono px-2 py-0.5 rounded-full flex items-center gap-1.5 relative z-10"
        style={{
          background: "rgba(0,212,255,0.10)",
          border: "1px solid rgba(0,212,255,0.35)",
          color: "#00D4FF",
          boxShadow: "0 0 12px rgba(0,212,255,0.20)",
        }}
      >
        <span className="w-1.5 h-1.5 rounded-full bg-[#00D4FF] animate-pulse" />
        PROCESSING
      </span>
      <div
        className="absolute inset-0 rounded-full"
        style={{
          border: "1px solid rgba(0,212,255,0.30)",
          animation: "pulse-ring 1.5s ease-out infinite",
        }}
      />
    </div>
  );

  return (
    <span
      className="text-[10px] font-mono px-2 py-0.5 rounded-full flex items-center gap-1"
      style={{
        background: "rgba(16,185,129,0.10)",
        border: "1px solid rgba(16,185,129,0.35)",
        color: "#10B981",
        boxShadow: "0 0 12px rgba(16,185,129,0.15)",
      }}
    >
      <Check className="w-3 h-3" />
      DONE
    </span>
  );
}

function FlowConnector({ label, active, accentRgb }: { label: string; active: boolean; accentRgb: string }) {
  const color = active ? `rgba(${accentRgb},1)` : "rgba(255,255,255,0.12)";
  const colorFaint = active ? `rgba(${accentRgb},0.30)` : "rgba(255,255,255,0.06)";

  return (
    <div className="h-12 w-full flex flex-col items-center justify-center relative">
      {/* Holographic tube — 3 lines */}
      <div className="relative h-full flex items-center justify-center" style={{ width: "2px" }}>
        {/* Side lines */}
        <div className="absolute h-full" style={{ width: "1px", left: "-3px", background: colorFaint }} />
        <div className="absolute h-full" style={{ width: "1px", right: "-3px", background: colorFaint }} />
        {/* Center line */}
        <div className="h-full" style={{ width: "2px", background: color, opacity: active ? 1 : 0.4 }} />

        {/* Glowing orb */}
        {active && (
          <motion.div
            className="absolute -translate-x-1/2"
            style={{
              width: "6px",
              height: "6px",
              borderRadius: "50%",
              background: `rgba(${accentRgb},1)`,
              boxShadow: `0 0 8px rgba(${accentRgb},1), 0 0 16px rgba(${accentRgb},1)`,
              left: "50%",
            }}
            animate={{ top: ["0%", "100%"] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
          />
        )}
      </div>

      {/* Label pill — glass */}
      <div
        className="absolute px-3 py-1 text-[10px] font-mono"
        style={{
          background: active ? `rgba(${accentRgb},0.08)` : "rgba(255,255,255,0.03)",
          border: `1px solid ${active ? `rgba(${accentRgb},0.40)` : "rgba(255,255,255,0.08)"}`,
          borderRadius: "20px",
          backdropFilter: "blur(8px)",
          color: active ? `rgba(${accentRgb},1)` : "#475569",
          boxShadow: active ? `0 2px 8px rgba(0,0,0,0.3)` : "none",
        }}
      >
        {label}
      </div>
    </div>
  );
}

function ScoreBar({ label, percentage, accentColor, glowColor }: { label: string; percentage: number; accentColor: string; glowColor: string }) {
  return (
    <div className="flex flex-col gap-1 w-full">
      <div className="flex justify-between text-xs font-mono text-muted-foreground">
        <span>{label}</span>
      </div>
      <div
        className="h-1.5 rounded-full overflow-hidden"
        style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)" }}
      >
        <div
          className="h-full rounded-full"
          style={{
            width: `${percentage}%`,
            background: accentColor,
            boxShadow: `0 0 8px ${glowColor}`,
          }}
        />
      </div>
    </div>
  );
}

function NeonBar({ percentage, accentGradient, glowColor }: { percentage: number; accentGradient: string; glowColor: string }) {
  return (
    <div
      className="h-1.5 rounded-full overflow-hidden"
      style={{
        background: "rgba(255,255,255,0.04)",
        border: "1px solid rgba(255,255,255,0.06)",
        height: "6px",
      }}
    >
      <div
        className="h-full rounded-full"
        style={{
          width: `${percentage}%`,
          background: accentGradient,
          boxShadow: `0 0 8px ${glowColor}`,
        }}
      />
    </div>
  );
}
