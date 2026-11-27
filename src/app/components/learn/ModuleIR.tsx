import { useState } from "react";
import { Search } from "lucide-react";
import { motion } from "motion/react";

export function ModuleIR() {
  const [alpha, setAlpha] = useState(0.5);
  const [query, setQuery] = useState("");
  const [simulated, setSimulated] = useState(false);

  const handleSimulate = () => {
    if (!query) return;
    setSimulated(false);
    setTimeout(() => setSimulated(true), 100);
  };

  return (
    <div className="grid md:grid-cols-2 gap-12 animate-in fade-in slide-in-from-bottom-4 duration-500">

      {/* Left Column: Theory */}
      <div className="space-y-8">
        <div>
          <h2 className="text-2xl font-display font-bold mb-4" style={{ letterSpacing: "-0.02em" }}>What is TF-IDF?</h2>
          <p className="text-sm text-muted-foreground mb-4 leading-relaxed">
            Term Frequency-Inverse Document Frequency measures how important a word is to a document in a collection. It increases proportionally to the number of times a word appears in the document, but is offset by the frequency of the word in the corpus.
          </p>
          {/* Formula block */}
          <div
            className="p-4 font-mono text-sm overflow-x-auto"
            style={{
              background: "rgba(0,0,0,0.40)",
              border: "1px solid rgba(255,255,255,0.06)",
              borderLeft: "3px solid rgba(0,212,255,0.50)",
              borderRadius: "0 8px 8px 0",
              backdropFilter: "blur(8px)",
              color: "#E0F7FF",
            }}
          >
            <span className="text-primary">TF-IDF(t,d)</span> = <span style={{ color: "#E0F7FF" }}>TF(t,d)</span> × <span className="text-secondary">log(N / DF(t))</span>
          </div>
          <div className="flex gap-4 mt-3 text-xs font-mono text-muted-foreground">
            <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-foreground" /> Term Freq</span>
            <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-secondary" /> Inverse Doc Freq</span>
          </div>
        </div>

        <div>
          <h2 className="text-2xl font-display font-bold mb-4" style={{ letterSpacing: "-0.02em" }}>What is BM25?</h2>
          <p className="text-sm text-muted-foreground mb-4 leading-relaxed">
            Okapi BM25 is an advanced bag-of-words retrieval function that improves upon TF-IDF by adding term frequency saturation and document length normalization.
          </p>
          <ul className="text-sm text-muted-foreground space-y-2 list-disc pl-4 marker:text-primary">
            <li>Prevents extremely high term frequencies from dominating the score</li>
            <li>Penalizes very long documents where terms appear frequently by chance</li>
          </ul>
        </div>

        <div>
          <h2 className="text-2xl font-display font-bold mb-4" style={{ letterSpacing: "-0.02em" }}>Score Fusion</h2>
          <div
            className="p-4 mb-4 font-mono text-sm"
            style={{
              background: "rgba(0,0,0,0.40)",
              border: "1px solid rgba(255,255,255,0.06)",
              borderLeft: "3px solid rgba(0,212,255,0.50)",
              borderRadius: "0 8px 8px 0",
              backdropFilter: "blur(8px)",
              color: "#E0F7FF",
            }}
          >
            score = <span className="text-primary">α</span> × norm(TF-IDF) + <span className="text-secondary">(1-α)</span> × norm(BM25)
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-xs font-mono text-muted-foreground" style={{ fontVariantNumeric: "tabular-nums" }}>
              <span>More TF-IDF (α={alpha.toFixed(2)})</span>
              <span>More BM25 (1-α={(1 - alpha).toFixed(2)})</span>
            </div>
            <input
              type="range"
              min="0" max="1" step="0.01"
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              className="w-full accent-primary"
            />
          </div>
        </div>
      </div>

      {/* Right Column: Demo */}
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
          <div className="font-mono text-xs text-muted-foreground mb-4" style={{ letterSpacing: "0.15em" }}>INTERACTIVE SIMULATION</div>

          <div className="flex gap-2 mb-8">
            <input
              type="text"
              placeholder="Type a query... e.g. 'history of physics'"
              className="flex-1 rounded-lg px-4 py-2 text-sm outline-none transition-all"
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
                color: "#F1F5F9",
              }}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSimulate()}
              onFocus={(e) => {
                e.currentTarget.style.border = "1px solid rgba(0,212,255,0.50)";
                e.currentTarget.style.boxShadow = "0 0 0 3px rgba(0,212,255,0.10)";
              }}
              onBlur={(e) => {
                e.currentTarget.style.border = "1px solid rgba(255,255,255,0.08)";
                e.currentTarget.style.boxShadow = "none";
              }}
            />
            <button
              onClick={handleSimulate}
              className="px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2"
              style={{
                background: "linear-gradient(135deg, rgba(0,212,255,0.90), rgba(0,150,220,0.90))",
                boxShadow: "0 0 20px rgba(0,212,255,0.35), 0 4px 12px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.20)",
                color: "#020510",
                fontWeight: 600,
              }}
            >
              <Search className="w-4 h-4" /> Run
            </button>
          </div>

          <div
            className="h-48 mb-4 flex items-end justify-between px-2 gap-1 pb-2"
            style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}
          >
            {Array.from({ length: 40 }).map((_, i) => {
              const isTop = i === 12 || i === 25 || i === 31;
              const height = simulated ? (isTop ? 80 + Math.random() * 20 : 10 + Math.random() * 30) : 10;
              return (
                <motion.div
                  key={i}
                  className="w-full rounded-t-sm"
                  style={simulated && isTop ? {
                    background: "#00D4FF",
                    boxShadow: "0 0 10px rgba(0,212,255,0.5)",
                  } : {
                    background: "rgba(255,255,255,0.06)",
                  }}
                  initial={{ height: "10%" }}
                  animate={{ height: `${height}%` }}
                  transition={{ duration: 0.5, delay: i * 0.01 }}
                />
              );
            })}
          </div>

          <div className="text-center font-mono text-xs text-muted-foreground">
            Corpus: 20,233 passages from SQuAD 2.0
          </div>
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
          <div className="font-mono text-xs text-muted-foreground mb-4 uppercase" style={{ letterSpacing: "0.15em" }}>Evaluation Results</div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                <th className="pb-2 font-medium text-muted-foreground">Metric</th>
                <th className="pb-2 font-medium text-muted-foreground">TF-IDF Only</th>
                <th className="pb-2 font-medium text-muted-foreground">BM25 Only</th>
                <th className="pb-2 font-medium text-primary">Hybrid</th>
              </tr>
            </thead>
            <tbody className="font-mono text-xs" style={{ fontVariantNumeric: "tabular-nums" }}>
              <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                <td className="py-3 text-muted-foreground">Recall@5</td>
                <td className="py-3 text-foreground">0.6210</td>
                <td className="py-3 text-foreground">0.7104</td>
                <td className="py-3 text-primary font-bold">0.7630</td>
              </tr>
              <tr>
                <td className="py-3 text-muted-foreground">Recall@10</td>
                <td className="py-3 text-foreground">0.6940</td>
                <td className="py-3 text-foreground">0.7820</td>
                <td className="py-3 text-primary font-bold">0.8340</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
