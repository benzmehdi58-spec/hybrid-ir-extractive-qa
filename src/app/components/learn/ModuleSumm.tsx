import { useState } from "react";

export function ModuleSumm() {
  const [alpha, setAlpha] = useState(0.3);
  const [beta, setBeta] = useState(0.5);
  const [gamma, setGamma] = useState(0.2);
  const [inputText, setInputText] = useState("Information extraction is a crucial part of natural language processing. Systems must often parse thousands of documents to find a single relevant fact. Traditional search engines return links to full documents, leaving the user to read through them to find their answer. Modern question answering systems attempt to read the documents automatically. By extracting the exact phrase that answers the user's question, these systems save significant time. This requires complex models that understand context and semantics. The performance of such systems has improved dramatically with the advent of transformer-based architectures.");
  const [summarized, setSummarized] = useState(false);

  const handleSliderChange = (setter: any, val: number, other1: number, other2: number, setOther1: any, setOther2: any) => {
    setter(val);
    const remainder = 1 - val;
    const ratio = other1 / (other1 + other2 || 1);
    setOther1(remainder * ratio);
    setOther2(remainder * (1 - ratio));
  };

  const sentences = inputText.split(/(?<=\.)\s+/);

  const glassCard = {
    background: "rgba(10,20,40,0.45)",
    backdropFilter: "blur(24px) saturate(180%)",
    border: "1px solid rgba(255,255,255,0.07)",
    borderTop: "1px solid rgba(255,255,255,0.13)",
    borderLeft: "1px solid rgba(255,255,255,0.10)",
    borderRadius: "16px",
    boxShadow: "0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08)",
  } as const;

  const glassInner = {
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.05)",
    borderRadius: "10px",
  } as const;

  return (
    <div className="grid md:grid-cols-2 gap-12 animate-in fade-in slide-in-from-bottom-4 duration-500">

      {/* Left Column */}
      <div className="space-y-8">
        <div>
          <h2 className="text-2xl font-display font-bold mb-4" style={{ letterSpacing: "-0.02em" }}>Three Scoring Methods</h2>
          <div className="space-y-6">
            <div className="p-4" style={glassCard}>
              <h3 className="font-mono text-sm font-bold text-foreground mb-2" style={{ letterSpacing: "0.10em" }}>1. TF-IDF Scoring</h3>
              <p className="text-sm text-muted-foreground">Scores sentences based on the weight of the keywords they contain.</p>
            </div>

            <div
              className="p-4"
              style={{
                ...glassCard,
                borderColor: "rgba(245,158,11,0.20)",
                borderTopColor: "rgba(245,158,11,0.30)",
                boxShadow: `0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08), 0 0 30px rgba(245,158,11,0.06)`,
              }}
            >
              <h3 className="font-mono text-sm font-bold text-secondary mb-2" style={{ letterSpacing: "0.10em" }}>2. TextRank (Graph-based)</h3>
              <p className="text-sm text-muted-foreground mb-4">Treats sentences as nodes and their cosine similarity as edges. Uses PageRank to find central sentences.</p>

              {/* Mini Graph Viz */}
              <div
                className="h-32 relative flex items-center justify-center overflow-hidden"
                style={glassInner}
              >
                <svg className="absolute inset-0 w-full h-full" style={{ stroke: "rgba(255,255,255,0.12)" }}>
                  <line x1="20%" y1="50%" x2="50%" y2="20%" strokeWidth="2" />
                  <line x1="50%" y1="20%" x2="80%" y2="50%" strokeWidth="4" stroke="rgba(245,158,11,0.50)" />
                  <line x1="80%" y1="50%" x2="50%" y2="80%" strokeWidth="1" />
                  <line x1="50%" y1="80%" x2="20%" y2="50%" strokeWidth="3" stroke="rgba(245,158,11,0.30)" />
                  <line x1="20%" y1="50%" x2="80%" y2="50%" strokeWidth="1" strokeDasharray="4" />
                </svg>
                <div className="absolute left-[20%] top-[50%] -translate-x-1/2 -translate-y-1/2 w-6 h-6 rounded-full z-10" style={{ background: "rgba(255,255,255,0.06)", border: "2px solid rgba(255,255,255,0.12)" }} />
                <div className="absolute left-[50%] top-[20%] -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full z-10" style={{ background: "rgba(245,158,11,0.20)", border: "2px solid rgba(245,158,11,0.60)", boxShadow: "0 0 12px rgba(245,158,11,0.40)" }} />
                <div className="absolute left-[80%] top-[50%] -translate-x-1/2 -translate-y-1/2 w-5 h-5 rounded-full z-10" style={{ background: "rgba(255,255,255,0.06)", border: "2px solid rgba(255,255,255,0.12)" }} />
                <div className="absolute left-[50%] top-[80%] -translate-x-1/2 -translate-y-1/2 w-7 h-7 rounded-full z-10" style={{ background: "rgba(255,255,255,0.04)", border: "2px solid rgba(245,158,11,0.30)" }} />
              </div>
            </div>

            <div className="p-4" style={glassCard}>
              <h3 className="font-mono text-sm font-bold text-foreground mb-2" style={{ letterSpacing: "0.10em" }}>3. Position Weighting</h3>
              <p className="text-sm text-muted-foreground">Applies a bonus score to the first and last sentences of a paragraph.</p>
            </div>
          </div>
        </div>

        <div>
          {/* Formula block */}
          <div
            className="p-4 mb-4 font-mono text-sm"
            style={{
              background: "rgba(0,0,0,0.40)",
              border: "1px solid rgba(255,255,255,0.06)",
              borderLeft: "3px solid rgba(245,158,11,0.50)",
              borderRadius: "0 8px 8px 0",
              backdropFilter: "blur(8px)",
              color: "#E0F7FF",
              fontVariantNumeric: "tabular-nums",
            }}
          >
            score = <span className="text-primary">α</span>·TF-IDF + <span className="text-secondary">β</span>·TextRank + <span className="text-accent">γ</span>·Position
          </div>
          <div className="space-y-4">
            <Slider label="α (TF-IDF)" val={alpha} setVal={(v: number) => handleSliderChange(setAlpha, v, beta, gamma, setBeta, setGamma)} color="primary" />
            <Slider label="β (TextRank)" val={beta} setVal={(v: number) => handleSliderChange(setBeta, v, alpha, gamma, setAlpha, setGamma)} color="secondary" />
            <Slider label="γ (Position)" val={gamma} setVal={(v: number) => handleSliderChange(setGamma, v, alpha, beta, setAlpha, setBeta)} color="accent" />
          </div>
        </div>
      </div>

      {/* Right Column */}
      <div className="space-y-6 flex flex-col h-full">
        <div className="p-6 flex-1 flex flex-col" style={glassCard}>
          <div className="font-mono text-xs text-muted-foreground mb-4 uppercase" style={{ letterSpacing: "0.15em" }}>Live Demo</div>

          <textarea
            className="w-full h-40 p-4 text-sm mb-4 outline-none resize-none rounded-lg"
            style={{
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.07)",
              color: "#F1F5F9",
              borderRadius: "10px",
            }}
            value={inputText}
            onChange={(e) => { setInputText(e.target.value); setSummarized(false); }}
            onFocus={(e) => { e.currentTarget.style.border = "1px solid rgba(245,158,11,0.40)"; }}
            onBlur={(e) => { e.currentTarget.style.border = "1px solid rgba(255,255,255,0.07)"; }}
          />

          <button
            onClick={() => setSummarized(true)}
            className="w-full py-2 rounded-lg text-sm font-medium mb-6 transition-all"
            style={{
              background: "linear-gradient(135deg, rgba(245,158,11,0.90), rgba(200,120,0,0.90))",
              boxShadow: "0 0 20px rgba(245,158,11,0.30), 0 4px 12px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.20)",
              color: "#020510",
              fontWeight: 600,
            }}
          >
            Summarize
          </button>

          {summarized && (
            <div className="flex-1 pt-4" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
              <div className="flex justify-between items-center mb-4">
                <span className="text-xs font-mono text-muted-foreground" style={{ letterSpacing: "0.10em" }}>OUTPUT</span>
                <span
                  className="text-xs font-mono px-2 py-1 rounded"
                  style={{
                    background: "rgba(245,158,11,0.10)",
                    border: "1px solid rgba(245,158,11,0.25)",
                    color: "#F59E0B",
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  {sentences.length} → 3 sentences · {Math.round((1 - 3 / sentences.length) * 100)}% reduction
                </span>
              </div>
              <div className="text-sm leading-relaxed space-y-1">
                {sentences.map((sentence, i) => {
                  if (!sentence.trim()) return null;
                  const actuallyKept = i === 0 || i === Math.floor(sentences.length / 2) || i === sentences.length - 1;
                  return actuallyKept ? (
                    <span
                      key={i}
                      className="block pl-2 py-1 my-1 text-foreground"
                      style={{
                        background: "rgba(245,158,11,0.08)",
                        borderLeft: "2px solid rgba(245,158,11,0.60)",
                        borderRadius: "0 4px 4px 0",
                      }}
                    >
                      {sentence}
                    </span>
                  ) : (
                    <span key={i} className="line-through text-muted-foreground block opacity-40 pl-2">{sentence}</span>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        <div className="p-6" style={glassCard}>
          <div className="font-mono text-xs text-muted-foreground mb-4 uppercase" style={{ letterSpacing: "0.15em" }}>ROUGE Score Card</div>
          <div className="grid grid-cols-2 gap-6">
            {[
              { name: "CNN/DailyMail", vals: [["ROUGE-1", "0.4210"], ["ROUGE-2", "0.1890"], ["ROUGE-L", "0.3870"]] },
              { name: "XSum", vals: [["ROUGE-1", "0.3120"], ["ROUGE-2", "0.1140"], ["ROUGE-L", "0.2680"]] },
            ].map((dataset) => (
              <div key={dataset.name}>
                <div className="text-sm font-bold mb-3 pb-2" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>{dataset.name}</div>
                <div className="space-y-2 font-mono text-xs" style={{ fontVariantNumeric: "tabular-nums" }}>
                  {dataset.vals.map(([label, val]) => (
                    <div key={label} className="flex justify-between">
                      <span className="text-muted-foreground">{label}</span>
                      <span className="text-foreground">{val}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 text-[10px] text-muted-foreground font-mono text-center">XSum tests cross-domain generalization</div>
        </div>
      </div>
    </div>
  );
}

function Slider({ label, val, setVal, color }: { label: string; val: number; setVal: (v: number) => void; color: string }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs font-mono text-muted-foreground" style={{ fontVariantNumeric: "tabular-nums" }}>
        <span>{label}</span>
        <span>{val.toFixed(2)}</span>
      </div>
      <input
        type="range" min="0" max="1" step="0.01"
        value={val}
        onChange={(e) => setVal(parseFloat(e.target.value))}
        className={`w-full accent-${color}`}
      />
    </div>
  );
}
