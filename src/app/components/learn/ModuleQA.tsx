import { useState } from "react";
import { Search } from "lucide-react";
import { motion } from "motion/react";

export function ModuleQA() {
  const [context, setContext] = useState("Alexander Graham Bell was a Scottish-born inventor, scientist, and engineer who is credited with patenting the first practical telephone.");
  const [question, setQuestion] = useState("Who invented the telephone?");
  const [simulated, setSimulated] = useState(false);

  const handleSimulate = () => {
    if (!context || !question) return;
    setSimulated(false);
    setTimeout(() => setSimulated(true), 100);
  };

  const tokens = context.split(" ");
  const answerStartIndex = 0; // "Alexander"
  const answerEndIndex = 2; // "Bell"

  return (
    <div className="grid md:grid-cols-2 gap-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
      
      {/* Left Column */}
      <div className="space-y-8">
        <div>
          <h2 className="text-2xl font-display font-bold mb-4">Architecture</h2>
          <div
            className="p-6 flex flex-col items-center gap-2"
            style={{
              background: "rgba(10,20,40,0.45)",
              backdropFilter: "blur(24px) saturate(180%)",
              border: "1px solid rgba(255,255,255,0.07)",
              borderTop: "1px solid rgba(255,255,255,0.13)",
              borderLeft: "1px solid rgba(255,255,255,0.10)",
              borderRadius: "16px",
              boxShadow: "0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08)",
            }}
          >
            <div className="text-xs font-mono mb-2 text-muted-foreground">Question + Context</div>
            <div className="w-px h-4" style={{ background: "rgba(255,255,255,0.08)" }} />
            <ArchitectureBlock label="GloVe 100d Embeddings" />
            <div className="w-px h-4" style={{ background: "rgba(255,255,255,0.08)" }} />
            <div className="flex items-center gap-4">
              <ArchitectureBlock label="2-layer BiLSTM Encoder" />
              <span className="text-[10px] text-muted-foreground font-mono w-24">← reads both directions</span>
            </div>
            <div className="w-px h-4" style={{ background: "rgba(255,255,255,0.08)" }} />
            <div className="flex items-center gap-4">
              <ArchitectureBlock label="BiDAF Attention" highlight />
              <span className="text-[10px] text-muted-foreground font-mono w-24">← aligns query & context</span>
            </div>
            <div className="w-px h-4" style={{ background: "rgba(255,255,255,0.08)" }} />
            <ArchitectureBlock label="2-layer Modeling LSTM" />
            <div className="w-px h-4" style={{ background: "rgba(255,255,255,0.08)" }} />
            <div className="flex gap-2">
              <ArchitectureBlock label="Span Start" small />
              <ArchitectureBlock label="Span End" small />
            </div>
            <div className="w-px h-4" style={{ background: "rgba(255,255,255,0.08)" }} />
            <div className="text-xs font-mono font-bold text-accent mt-2" style={{ letterSpacing: "0.10em" }}>Answer Span</div>
          </div>
        </div>

        <div>
          <h2 className="text-2xl font-display font-bold mb-4">What is BiDAF Attention?</h2>
          <p className="text-sm text-muted-foreground mb-4">
            Bidirectional Attention Flow (BiDAF) computes attention in two directions simultaneously: Context-to-Query (which question words are most relevant to each context word) and Query-to-Context (which context words have the closest similarity to one of the query words).
          </p>
          <div
            className="p-3"
            style={{
              background: "rgba(255,255,255,0.02)",
              border: "1px solid rgba(167,139,250,0.15)",
              borderRadius: "8px",
              boxShadow: "0 0 20px rgba(167,139,250,0.08)",
            }}
          >
            <div className="grid grid-cols-6 gap-0.5 w-fit mx-auto">
              {Array.from({ length: 24 }).map((_, i) => {
                const w = Math.random();
                return (
                  <div
                    key={i}
                    className="w-6 h-6"
                    style={{
                      borderRadius: "2px",
                      backgroundColor: `rgba(167,139,250,${w * 0.66 + 0.04})`,
                      boxShadow: w > 0.7 ? "inset 0 0 6px rgba(167,139,250,0.5)" : "none",
                    }}
                  />
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Right Column */}
      <div className="space-y-6 flex flex-col h-full">
        <div
          className="p-6 flex-1"
          style={{
            background: "rgba(10,20,40,0.45)",
            backdropFilter: "blur(24px) saturate(180%)",
            border: "1px solid rgba(255,255,255,0.07)",
            borderTop: "1px solid rgba(255,255,255,0.13)",
            borderLeft: "1px solid rgba(255,255,255,0.10)",
            borderRadius: "16px",
            boxShadow: "0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08)",
          }}
        >
          <div className="font-mono text-xs text-muted-foreground mb-4 uppercase" style={{ letterSpacing: "0.15em" }}>Interactive Demo</div>

          <div className="space-y-4 mb-6">
            <div>
              <label className="text-[10px] font-mono text-muted-foreground block mb-1" style={{ letterSpacing: "0.10em" }}>CONTEXT PASSAGE</label>
              <textarea
                className="w-full h-24 p-3 text-sm outline-none resize-none rounded-lg"
                style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", color: "#F1F5F9", borderRadius: "10px" }}
                value={context}
                onChange={(e) => setContext(e.target.value)}
                onFocus={(e) => { e.currentTarget.style.border = "1px solid rgba(167,139,250,0.40)"; }}
                onBlur={(e) => { e.currentTarget.style.border = "1px solid rgba(255,255,255,0.07)"; }}
              />
            </div>
            <div>
              <label className="text-[10px] font-mono text-muted-foreground block mb-1" style={{ letterSpacing: "0.10em" }}>QUESTION</label>
              <input
                type="text"
                className="w-full p-3 text-sm outline-none rounded-lg"
                style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", color: "#F1F5F9", borderRadius: "10px" }}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSimulate()}
                onFocus={(e) => { e.currentTarget.style.border = "1px solid rgba(167,139,250,0.40)"; }}
                onBlur={(e) => { e.currentTarget.style.border = "1px solid rgba(255,255,255,0.07)"; }}
              />
            </div>
            <button
              onClick={handleSimulate}
              className="w-full py-2 rounded-lg text-sm font-medium flex items-center justify-center gap-2"
              style={{
                background: "linear-gradient(135deg, rgba(167,139,250,0.90), rgba(130,100,220,0.90))",
                boxShadow: "0 0 20px rgba(167,139,250,0.35), 0 4px 12px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.20)",
                color: "#020510",
                fontWeight: 600,
              }}
            >
              <Search className="w-4 h-4" /> Find Answer
            </button>
          </div>

          {simulated && (
            <div className="pt-6 space-y-6" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
              <div className="text-sm leading-loose">
                {tokens.map((token, i) => {
                  const isAnswer = i >= answerStartIndex && i <= answerEndIndex;
                  return (
                    <motion.span
                      key={i}
                      className="inline-block mr-1.5 px-1 font-bold"
                      style={isAnswer ? {
                        background: "rgba(167,139,250,0.18)",
                        border: "1px solid rgba(167,139,250,0.45)",
                        boxShadow: "0 0 8px rgba(167,139,250,0.20)",
                        color: "#DDD6FE",
                        borderRadius: "4px",
                      } : {}}
                      initial={isAnswer ? { opacity: 0 } : {}}
                      animate={isAnswer ? { opacity: 1 } : {}}
                      transition={{ delay: i * 0.05 }}
                    >
                      {token}
                    </motion.span>
                  );
                })}
              </div>

              <div className="p-4 space-y-3" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)", borderRadius: "10px" }}>
                <div className="flex justify-between items-center">
                  <span className="text-xs font-mono text-muted-foreground">Predicted:</span>
                  <span className="text-sm font-bold text-foreground">"Alexander Graham Bell"</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs font-mono text-muted-foreground">Confidence:</span>
                  <span className="text-xs font-mono text-accent" style={{ fontVariantNumeric: "tabular-nums" }}>Start 87.3% · End 91.2%</span>
                </div>
                <div className="pt-2" style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}>
                  <div className="text-[10px] font-mono text-muted-foreground mb-2" style={{ letterSpacing: "0.10em" }}>TOP CANDIDATE START TOKENS</div>
                  <div className="space-y-1.5">
                    <CandidateBar token="Alexander" prob={0.87} />
                    <CandidateBar token="inventor," prob={0.12} />
                    <CandidateBar token="who" prob={0.01} />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div
          className="p-6"
          style={{
            background: "rgba(10,20,40,0.45)",
            backdropFilter: "blur(24px) saturate(180%)",
            border: "1px solid rgba(255,255,255,0.07)",
            borderTop: "1px solid rgba(255,255,255,0.13)",
            borderLeft: "1px solid rgba(255,255,255,0.10)",
            borderRadius: "16px",
            boxShadow: "0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08)",
          }}
        >
          <div className="font-mono text-xs text-muted-foreground mb-4 uppercase" style={{ letterSpacing: "0.15em" }}>Evaluation Card</div>
          <div className="grid grid-cols-2 gap-6">
            {[
              { name: "SQuAD 2.0", vals: [["Exact Match", "0.6120", false], ["F1 Score", "0.7340", true]] },
              { name: "HotpotQA",  vals: [["Exact Match", "0.4280", false], ["F1 Score", "0.5690", false]] },
            ].map((dataset) => (
              <div key={dataset.name}>
                <div className="text-sm font-bold mb-3 pb-2" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>{dataset.name}</div>
                <div className="space-y-2 font-mono text-xs" style={{ fontVariantNumeric: "tabular-nums" }}>
                  {dataset.vals.map(([label, val, highlight]) => (
                    <div key={label as string} className="flex justify-between">
                      <span className="text-muted-foreground">{label}</span>
                      <span style={{ color: highlight ? "#A78BFA" : "#F1F5F9", fontWeight: highlight ? 700 : 400 }}>{val}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 text-[10px] text-muted-foreground font-mono text-center">HotpotQA tests multi-hop reasoning</div>
        </div>
      </div>

    </div>
  );
}

function ArchitectureBlock({ label, highlight, small }: any) {
  return (
    <div
      className={`flex items-center justify-center text-center font-mono ${small ? "px-3 py-2 text-[10px]" : "px-6 py-3 text-xs w-64"}`}
      style={highlight ? {
        background: "rgba(167,139,250,0.15)",
        border: "1px solid rgba(167,139,250,0.50)",
        color: "#A78BFA",
        boxShadow: "0 0 15px rgba(167,139,250,0.20)",
        borderRadius: "8px",
      } : {
        background: "rgba(255,255,255,0.04)",
        border: "1px solid rgba(255,255,255,0.07)",
        color: "#F1F5F9",
        borderRadius: "8px",
      }}
    >
      {label}
    </div>
  );
}

function CandidateBar({ token, prob }: any) {
  return (
    <div className="flex items-center gap-2 text-xs font-mono">
      <div className="w-20 truncate text-muted-foreground text-[10px]">{token}</div>
      <div
        className="flex-1 h-1.5 rounded-full overflow-hidden"
        style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)", height: "6px" }}
      >
        <motion.div
          className="h-full rounded-full"
          style={{ background: "linear-gradient(90deg, rgba(100,60,200,0.8), #A78BFA)", boxShadow: "0 0 8px rgba(167,139,250,0.40)" }}
          initial={{ width: 0 }}
          animate={{ width: `${prob * 100}%` }}
          transition={{ duration: 0.5, delay: 0.2 }}
        />
      </div>
      <div className="w-8 text-right text-[10px] text-accent" style={{ fontVariantNumeric: "tabular-nums" }}>{(prob * 100).toFixed(0)}%</div>
    </div>
  );
}
