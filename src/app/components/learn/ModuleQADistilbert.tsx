import { useState } from "react";
import { Search, BookOpen, Fingerprint, Layers } from "lucide-react";
import { motion } from "motion/react";

export function ModuleQADistilbert() {
  const [context, setContext] = useState("Alexander Graham Bell was a Scottish-born inventor, scientist, and engineer who is credited with patenting the first practical telephone.");
  const [question, setQuestion] = useState("Who invented the telephone?");
  const [simulated, setSimulated] = useState(false);

  const handleSimulate = () => {
    if (!context || !question) return;
    setSimulated(false);
    setTimeout(() => setSimulated(true), 100);
  };

  return (
    <div className="grid md:grid-cols-2 gap-12 animate-in fade-in slide-in-from-bottom-4 duration-500 pb-12">
      
      {/* Left Column: Theory */}
      <div className="space-y-8">
        <div>
          <h2 className="text-2xl font-display font-bold mb-4">What is DistilBERT?</h2>
          <p className="text-sm text-muted-foreground mb-4">
            A distilled (compressed) version of BERT. It is 40% smaller and 60% faster, while retaining 97% of BERT's language understanding capabilities.
          </p>
          <div
            className="overflow-hidden font-mono text-xs"
            style={{
              background: "rgba(10,20,40,0.45)",
              backdropFilter: "blur(24px) saturate(180%)",
              border: "1px solid rgba(255,255,255,0.07)",
              borderTop: "1px solid rgba(255,255,255,0.13)",
              borderRadius: "16px",
              boxShadow: "0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08)",
            }}
          >
            <div className="grid grid-cols-3 p-3 font-bold" style={{ background: "rgba(255,255,255,0.04)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
              <div />
              <div className="text-primary text-center" style={{ letterSpacing: "0.10em" }}>BERT-base</div>
              <div className="text-primary text-center" style={{ letterSpacing: "0.10em" }}>DistilBERT</div>
            </div>
            {[
              ["Layers", "12", "6", false],
              ["Parameters", "110M", "66M", false],
              ["Speed", "1×", "1.6×", true],
              ["Performance", "100%", "97%", true],
            ].map(([label, bert, distil, green], i, arr) => (
              <div
                key={label as string}
                className="grid grid-cols-3 p-3 items-center"
                style={{ borderBottom: i < arr.length - 1 ? "1px solid rgba(255,255,255,0.04)" : "none", fontVariantNumeric: "tabular-nums" }}
              >
                <div className="text-muted-foreground">{label}</div>
                <div className="text-center text-foreground">{bert}</div>
                <div className="text-center font-bold" style={{ color: green ? "#10B981" : "#F1F5F9" }}>{distil}</div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h2 className="text-2xl font-display font-bold mb-4">Fine-tuning vs Training from Scratch</h2>
          <div
            className="p-6 relative flex gap-6"
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
            <div className="flex-1 space-y-3">
              <div className="text-secondary font-display font-bold text-sm">BiLSTM (Module 3)</div>
              <ul className="text-xs text-muted-foreground space-y-2 list-disc pl-4 marker:text-secondary">
                <li>Random weight initialization</li>
                <li>GloVe static embeddings</li>
                <li>Learns from 20k SQuAD samples</li>
                <li>No language knowledge before training</li>
                <li>Training: 30 epochs</li>
              </ul>
            </div>

            {/* VS divider */}
            <div className="absolute left-1/2 top-0 bottom-0 w-px -translate-x-1/2" style={{ background: "rgba(255,255,255,0.06)" }}>
              <div
                className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-[10px] font-mono font-bold px-2 py-1 rounded-full"
                style={{ background: "rgba(10,20,40,0.80)", color: "#F1F5F9", border: "1px solid rgba(255,255,255,0.12)", backdropFilter: "blur(8px)" }}
              >
                VS
              </div>
            </div>

            <div className="flex-1 space-y-3 pl-2">
              <div className="text-primary font-display font-bold text-sm">DistilBERT (Module 4)</div>
              <ul className="text-xs text-muted-foreground space-y-2 list-disc pl-4 marker:text-primary">
                <li>Pre-trained on BookCorpus + Wikipedia</li>
                <li>Contextual subword tokenization (WordPiece)</li>
                <li>Fine-tuned on full 87k SQuAD 2.0 samples</li>
                <li>Brings rich language understanding</li>
                <li>Fine-tuning: 3 epochs only</li>
              </ul>
            </div>
          </div>
        </div>

        <div>
          <h2 className="text-2xl font-display font-bold mb-4">Sliding Window Tokenization</h2>
          <div
            className="p-6 space-y-4"
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
            <div
              className="font-mono text-[10px] text-primary px-2 py-1 w-fit mb-2"
              style={{ background: "rgba(0,212,255,0.08)", border: "1px solid rgba(0,212,255,0.25)", borderRadius: "6px" }}
            >
              MAX_LENGTH=384 · STRIDE=128
            </div>
            <div className="space-y-2 relative">
              {[{ ml: "0" }, { ml: "30%" }].map((row, ri) => (
                <div
                  key={ri}
                  className="flex h-8 overflow-hidden"
                  style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "6px", marginLeft: row.ml, opacity: ri === 0 ? 0.85 : 1 }}
                >
                  {[
                    { label: "[CLS]", style: { background: "rgba(255,255,255,0.06)", color: "#94A3B8", borderRight: "1px solid rgba(255,255,255,0.06)" } },
                    { label: "[Q]",   style: { background: "rgba(59,130,246,0.15)", color: "#93C5FD", borderRight: "1px solid rgba(255,255,255,0.06)" } },
                    { label: "[SEP]", style: { background: "rgba(255,255,255,0.06)", color: "#94A3B8", borderRight: "1px solid rgba(255,255,255,0.06)" } },
                  ].map((chip) => (
                    <div key={chip.label} className="text-[10px] flex items-center px-2 font-mono" style={chip.style}>{chip.label}</div>
                  ))}
                  <div className="flex-1 text-xs flex items-center justify-center font-mono" style={{ background: "rgba(139,92,246,0.12)", color: "#C4B5FD" }}>
                    Context chunk {ri + 1} (256 tokens)
                  </div>
                  <div className="text-[10px] flex items-center px-2 font-mono" style={{ background: "rgba(255,255,255,0.06)", color: "#94A3B8", borderLeft: "1px solid rgba(255,255,255,0.06)" }}>[SEP]</div>
                </div>
              ))}
              <div className="absolute top-0 bottom-0 left-[30%] right-[30%] pointer-events-none flex items-end justify-center pb-1" style={{ borderLeft: "2px dashed rgba(0,212,255,0.40)", borderRight: "2px dashed rgba(0,212,255,0.40)", background: "rgba(0,212,255,0.04)" }}>
                <span className="text-[10px] font-mono text-primary px-1" style={{ background: "rgba(2,5,16,0.80)", borderRadius: "4px" }}>STRIDE = 128 tokens overlap</span>
              </div>
            </div>
          </div>
        </div>

        <div>
          <h2 className="text-2xl font-display font-bold mb-4">Unanswerable Questions (SQuAD 2.0)</h2>
          <p className="text-sm text-muted-foreground mb-4">
            When no answer exists, the model predicts the <span className="font-mono text-xs bg-muted px-1 rounded">[CLS]</span> token as both the start and end of the answer span.
          </p>
          <div
            className="p-4 font-mono text-xs space-y-3"
            style={{
              background: "rgba(0,0,0,0.40)",
              border: "1px solid rgba(255,255,255,0.06)",
              borderLeft: "3px solid rgba(0,212,255,0.50)",
              borderRadius: "0 8px 8px 0",
              backdropFilter: "blur(8px)",
            }}
          >
            <div className="flex items-center gap-3 text-muted-foreground">
              <span className="text-foreground">Unanswerable:</span>
              <span>Start → [CLS]</span>
              <span>·</span>
              <span>End → [CLS]</span>
              <span className="text-primary">→ "Unanswerable"</span>
            </div>
            <div className="flex items-center gap-3 text-muted-foreground">
              <span className="text-foreground">Answerable:</span>
              <span>Start → token 14</span>
              <span>·</span>
              <span>End → token 17</span>
              <span className="text-secondary">→ "Answer span"</span>
            </div>
          </div>
        </div>

        <div>
          <h2 className="text-2xl font-display font-bold mb-4">Training Configuration</h2>
          <div
            className="p-6 font-mono text-sm leading-relaxed overflow-x-auto"
            style={{
              background: "rgba(2,5,16,0.80)",
              border: "1px solid rgba(255,255,255,0.06)",
              borderRadius: "10px",
              backdropFilter: "blur(12px)",
            }}
          >
            <div className="grid grid-cols-[150px_1fr] gap-4" style={{ fontVariantNumeric: "tabular-nums" }}>
              {[
                ["Model",        "distilbert-base-uncased", "string"],
                ["Max length",   "384 tokens",              "number"],
                ["Stride",       "128 tokens",              "number"],
                ["Batch size",   "16",                      "number"],
                ["Epochs",       "3",                       "number"],
                ["LR",           "2e-5",                    "number"],
                ["Weight decay", "0.01",                    "number"],
                ["Warmup",       "10% of steps",            "string"],
                ["Optimizer",    "AdamW",                   "string"],
                ["Scheduler",    "Linear decay",            "string"],
              ].map(([key, val, type]) => [
                <div key={`k-${key}`} style={{ color: "#94A3B8" }}>{key}</div>,
                <div key={`v-${key}`} style={{ color: type === "number" ? "#F59E0B" : type === "boolean" ? "#10B981" : "#00D4FF" }}>{val}</div>,
              ])}
            </div>
          </div>
        </div>
      </div>

      {/* Right Column: Demo & Results */}
      <div className="space-y-6 flex flex-col">
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
          <div className="font-mono text-xs text-muted-foreground mb-4 uppercase" style={{ letterSpacing: "0.15em" }}>Interactive Demo</div>

          <div className="space-y-4 mb-6">
            <div>
              <label className="text-[10px] font-mono text-muted-foreground block mb-1" style={{ letterSpacing: "0.10em" }}>CONTEXT PASSAGE</label>
              <textarea
                className="w-full h-24 p-3 text-sm outline-none resize-none rounded-lg"
                style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", color: "#F1F5F9", borderRadius: "10px" }}
                value={context}
                onChange={(e) => setContext(e.target.value)}
                onFocus={(e) => { e.currentTarget.style.border = "1px solid rgba(0,212,255,0.40)"; }}
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
                onFocus={(e) => { e.currentTarget.style.border = "1px solid rgba(0,212,255,0.40)"; }}
                onBlur={(e) => { e.currentTarget.style.border = "1px solid rgba(255,255,255,0.07)"; }}
              />
            </div>
            <button
              onClick={handleSimulate}
              className="w-full py-2 rounded-lg text-sm font-medium flex items-center justify-center gap-2"
              style={{
                background: "linear-gradient(135deg, rgba(0,212,255,0.90), rgba(0,150,220,0.90))",
                boxShadow: "0 0 20px rgba(0,212,255,0.35), 0 4px 12px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.20)",
                color: "#020510",
                fontWeight: 600,
              }}
            >
              <Search className="w-4 h-4" /> Run DistilBERT
            </button>
          </div>

          {simulated && (
            <div className="pt-6 space-y-6" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
              <div className="flex flex-wrap gap-1 font-mono text-xs">
                {[
                  { label: "[CLS]",              style: { background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.07)", color: "#94A3B8", borderRadius: "4px" } },
                  { label: "question tokens",    style: { background: "rgba(59,130,246,0.15)",  border: "1px solid rgba(59,130,246,0.30)",  color: "#93C5FD", borderRadius: "4px" } },
                  { label: "[SEP]",              style: { background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.07)", color: "#94A3B8", borderRadius: "4px" } },
                  { label: "context: Alexander Graham", style: { background: "rgba(139,92,246,0.15)", border: "1px solid rgba(139,92,246,0.30)", color: "#C4B5FD", borderRadius: "4px" } },
                ].map((chip) => (
                  <span key={chip.label} className="px-1.5 py-0.5" style={chip.style}>{chip.label}</span>
                ))}
                <motion.span
                  className="px-1.5 py-0.5 font-bold"
                  style={{
                    background: "rgba(0,212,255,0.18)",
                    border: "1px solid rgba(0,212,255,0.45)",
                    boxShadow: "0 0 8px rgba(0,212,255,0.20)",
                    color: "#67E8F9",
                    borderRadius: "4px",
                  }}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  Bell
                </motion.span>
                {[
                  { label: "...",    style: { background: "rgba(139,92,246,0.15)", border: "1px solid rgba(139,92,246,0.30)", color: "#C4B5FD", borderRadius: "4px" } },
                  { label: "[SEP]", style: { background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.07)", color: "#94A3B8", borderRadius: "4px" } },
                ].map((chip) => (
                  <span key={chip.label} className="px-1.5 py-0.5" style={chip.style}>{chip.label}</span>
                ))}
              </div>

              <div className="p-4 space-y-3" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)", borderRadius: "10px" }}>
                <div className="flex flex-col gap-1">
                  <span className="text-xs font-mono text-muted-foreground">Predicted:</span>
                  <span className="text-2xl font-bold text-foreground" style={{ letterSpacing: "-0.02em", textShadow: "0 0 20px rgba(255,255,255,0.15)" }}>"Alexander Graham Bell"</span>
                </div>
                <div className="flex justify-between items-center pt-3" style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}>
                  <span className="text-xs font-mono text-muted-foreground">Confidence:</span>
                  <span className="text-xs font-mono text-primary" style={{ fontVariantNumeric: "tabular-nums" }}>Start logit 4.21 · End logit 3.87</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs font-mono text-muted-foreground">Answerability:</span>
                  <span className="text-xs font-mono text-foreground">CLS score: -2.34 → <span className="text-[#10B981]">Answerable ✓</span></span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Full Comparison Card */}
        <div
          className="p-6 md:col-span-2 col-span-1 mt-6"
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
          <div className="font-mono text-xs text-muted-foreground mb-6 uppercase text-center" style={{ letterSpacing: "0.15em" }}>BILSTM vs DISTILBERT — FULL RESULTS</div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm font-mono text-left whitespace-nowrap">
              <thead>
                <tr style={{ background: "rgba(255,255,255,0.04)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                  <th className="px-4 py-3 font-medium" style={{ color: "#94A3B8", letterSpacing: "0.08em" }}>Dataset</th>
                  <th className="px-4 py-3 font-medium" style={{ color: "#94A3B8", letterSpacing: "0.08em" }}>Metric</th>
                  <th className="px-4 py-3 font-medium text-secondary" style={{ letterSpacing: "0.08em" }}>BiLSTM + GloVe</th>
                  <th className="px-4 py-3 font-medium text-primary" style={{ letterSpacing: "0.08em" }}>DistilBERT</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ["SQuAD 2.0", "EM",  "~10%", 10,  "~79%", 79],
                  ["SQuAD 2.0", "F1",  "~17%", 17,  "~83%", 83],
                  ["HotpotQA",  "EM",  "low",  null, "moderate", null],
                  ["HotpotQA",  "F1",  "low",  null, "moderate", null],
                ].map(([dataset, metric, bilstmVal, bilstmPct, distilVal, distilPct], i) => (
                  <tr
                    key={i}
                    style={{ borderBottom: "1px solid rgba(255,255,255,0.04)", fontVariantNumeric: "tabular-nums" }}
                  >
                    <td className="px-4 py-3 text-foreground">{dataset}</td>
                    <td className="px-4 py-3 text-muted-foreground">{metric}</td>
                    <td className="px-4 py-3 text-secondary">
                      {bilstmPct != null ? (
                        <div className="flex items-center gap-2">
                          <span>{bilstmVal}</span>
                          <div className="w-12 overflow-hidden" style={{ height: "4px", background: "rgba(245,158,11,0.15)", borderRadius: "2px" }}>
                            <div style={{ height: "100%", background: "#F59E0B", width: `${bilstmPct}%`, boxShadow: "0 0 6px rgba(245,158,11,0.4)" }} />
                          </div>
                        </div>
                      ) : bilstmVal}
                    </td>
                    <td className="px-4 py-3 text-primary font-bold">
                      {distilPct != null ? (
                        <div className="flex items-center gap-2">
                          <span>{distilVal}</span>
                          <div className="w-12 overflow-hidden" style={{ height: "4px", background: "rgba(0,212,255,0.15)", borderRadius: "2px" }}>
                            <div style={{ height: "100%", background: "#00D4FF", width: `${distilPct}%`, boxShadow: "0 0 6px rgba(0,212,255,0.4)" }} />
                          </div>
                        </div>
                      ) : distilVal}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-[10px] font-mono text-muted-foreground mt-4 text-center">
            BiLSTM trained on 20k samples · DistilBERT fine-tuned on 87k samples · Same EM/F1 evaluation function
          </div>

          <div className="mt-8 pt-6" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
            <h3 className="font-display font-bold text-foreground mb-4" style={{ letterSpacing: "-0.02em" }}>Why the gap?</h3>
            <div className="grid grid-cols-3 gap-4">
              <ReasonPill
                icon={<BookOpen className="w-4 h-4 text-primary" />}
                title="Pre-training"
                desc="DistilBERT learned language from billions of words before seeing SQuAD."
              />
              <ReasonPill
                icon={<Fingerprint className="w-4 h-4 text-primary" />}
                title="Subword Tokenization"
                desc="WordPiece handles rare words better than GloVe's fixed vocabulary."
              />
              <ReasonPill
                icon={<Layers className="w-4 h-4 text-primary" />}
                title="Architecture"
                desc="Self-attention captures long-range dependencies BiLSTM misses."
              />
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

function ReasonPill({ icon, title, desc }: any) {
  return (
    <div
      className="p-4 flex flex-col gap-2"
      style={{
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.05)",
        borderRadius: "10px",
        transition: "all 200ms cubic-bezier(0.34, 1.56, 0.64, 1)",
      }}
    >
      <div className="flex items-center gap-2">
        <div className="p-1.5 rounded" style={{ background: "rgba(0,212,255,0.10)", border: "1px solid rgba(0,212,255,0.20)" }}>{icon}</div>
        <span className="font-bold text-sm text-foreground">{title}</span>
      </div>
      <p className="text-xs text-muted-foreground">{desc}</p>
    </div>
  );
}
