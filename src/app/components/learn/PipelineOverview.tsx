import { ArrowRight } from "lucide-react";

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
  borderRadius: "8px",
} as const;

export function PipelineOverview() {
  return (
    <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="text-center space-y-4 pt-8">
        <h1
          className="text-4xl font-display font-bold"
          style={{
            letterSpacing: "-0.02em",
            background: "linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text",
          }}
        >
          How the Pipeline Works
        </h1>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          An open-domain question answering system composed of three cascading machine learning modules.
          Each stage refines and filters information from a massive corpus to extract the final exact answer.
        </p>
      </div>

      <div className="relative py-12">
        {/* Connection line */}
        <div
          className="absolute top-1/2 left-0 right-0 -translate-y-1/2 -z-10 hidden md:block"
          style={{ height: "1px", background: "linear-gradient(90deg, transparent, rgba(0,212,255,0.20), rgba(245,158,11,0.20), rgba(167,139,250,0.20), transparent)" }}
        />

        <div className="grid md:grid-cols-5 gap-4 items-center">

          <div className="flex flex-col items-center">
            <div
              className="w-16 h-16 rounded-2xl flex items-center justify-center text-2xl mb-4"
              style={glassCard}
            >
              👤
            </div>
            <div className="text-center">
              <div className="font-mono text-sm font-bold">User Question</div>
            </div>
          </div>

          <div className="hidden md:flex justify-center text-muted-foreground">
            <ArrowRight className="w-6 h-6" style={{ color: "rgba(0,212,255,0.40)" }} />
          </div>

          <div className="flex flex-col items-center">
            <div
              className="w-full p-4 text-center mb-4"
              style={{
                ...glassCard,
                borderColor: "rgba(0,212,255,0.25)",
                borderTopColor: "rgba(0,212,255,0.40)",
                boxShadow: "0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08), 0 0 30px rgba(0,212,255,0.08)",
              }}
            >
              <div className="text-primary font-mono text-xs mb-1" style={{ letterSpacing: "0.15em" }}>MODULE 1</div>
              <div className="font-bold text-foreground">IR Module</div>
            </div>
            <div className="text-center text-xs text-muted-foreground font-mono">
              Finds relevant passages<br />TF-IDF + BM25
            </div>
            <div
              className="text-[10px] mt-2 px-2 py-1 rounded-full text-primary font-mono"
              style={{ background: "rgba(0,212,255,0.06)", border: "1px solid rgba(0,212,255,0.20)" }}
            >
              top-k passages
            </div>
          </div>

          <div className="hidden md:flex justify-center text-muted-foreground">
            <ArrowRight className="w-6 h-6" style={{ color: "rgba(245,158,11,0.40)" }} />
          </div>

          <div className="flex flex-col items-center">
            <div
              className="w-full p-4 text-center mb-4"
              style={{
                ...glassCard,
                borderColor: "rgba(245,158,11,0.25)",
                borderTopColor: "rgba(245,158,11,0.40)",
                boxShadow: "0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08), 0 0 30px rgba(245,158,11,0.08)",
              }}
            >
              <div className="text-secondary font-mono text-xs mb-1" style={{ letterSpacing: "0.15em" }}>MODULE 2</div>
              <div className="font-bold text-foreground">Summarizer</div>
            </div>
            <div className="text-center text-xs text-muted-foreground font-mono">
              Compresses passages<br />TextRank hybrid
            </div>
            <div
              className="text-[10px] mt-2 px-2 py-1 rounded-full text-secondary font-mono"
              style={{ background: "rgba(245,158,11,0.06)", border: "1px solid rgba(245,158,11,0.20)" }}
            >
              compressed text
            </div>
          </div>

          <div className="hidden md:flex justify-center text-muted-foreground">
            <ArrowRight className="w-6 h-6" style={{ color: "rgba(167,139,250,0.40)" }} />
          </div>

          <div className="flex flex-col items-center">
            <div
              className="w-full p-4 text-center mb-4"
              style={{
                ...glassCard,
                borderColor: "rgba(167,139,250,0.25)",
                borderTopColor: "rgba(167,139,250,0.40)",
                boxShadow: "0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08), 0 0 30px rgba(167,139,250,0.08)",
              }}
            >
              <div className="text-accent font-mono text-xs mb-1" style={{ letterSpacing: "0.15em" }}>MODULE 3</div>
              <div className="font-bold text-foreground">QA Model</div>
            </div>
            <div className="text-center text-xs text-muted-foreground font-mono">
              Extracts answer span<br />BiLSTM + Attention
            </div>
            <div
              className="text-[10px] mt-2 px-2 py-1 rounded-full text-accent font-mono"
              style={{ background: "rgba(167,139,250,0.06)", border: "1px solid rgba(167,139,250,0.20)" }}
            >
              exact answer
            </div>
          </div>

        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        <MetricCard title="Retrieval Performance" val="0.834" label="Recall@10" accentRgb="0,212,255" />
        <MetricCard title="Summarization Quality" val="0.421" label="ROUGE-1" accentRgb="245,158,11" />
        <MetricCard title="QA Accuracy" val="73.4%" label="F1 Score" accentRgb="167,139,250" />
      </div>

      <div
        className="p-8 mt-12 flex flex-col md:flex-row gap-8 items-center"
        style={glassCard}
      >
        <div className="flex-1 space-y-4">
          <h3 className="font-display text-xl" style={{ letterSpacing: "-0.02em" }}>Datasets Used</h3>
          <p className="text-sm text-muted-foreground">
            The pipeline is trained on distinct datasets to handle different tasks. SQuAD 2.0 provides
            the reading comprehension foundation, while CNN/DailyMail is used to train the abstractive summarization metrics.
          </p>
        </div>
        <div className="flex-1 grid grid-cols-2 gap-4">
          {[
            { name: "SQuAD 2.0", desc: "Feeds IR + QA Modules" },
            { name: "CNN/DailyMail", desc: "Feeds Summarization" },
          ].map((d) => (
            <div key={d.name} className="p-4 text-center" style={glassInner}>
              <div className="font-mono text-foreground font-bold mb-1">{d.name}</div>
              <div className="text-xs text-muted-foreground">{d.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function MetricCard({ title, val, label, accentRgb }: { title: string; val: string; label: string; accentRgb: string }) {
  return (
    <div
      className="p-6"
      style={{
        background: "rgba(10,20,40,0.45)",
        backdropFilter: "blur(24px) saturate(180%)",
        border: `1px solid rgba(${accentRgb},0.20)`,
        borderTop: `1px solid rgba(${accentRgb},0.35)`,
        borderLeft: "1px solid rgba(255,255,255,0.10)",
        borderRadius: "16px",
        boxShadow: `0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08), 0 0 30px rgba(${accentRgb},0.06)`,
        transform: "translateZ(0)",
        transition: "all 200ms cubic-bezier(0.34, 1.56, 0.64, 1)",
      }}
    >
      <div className="text-sm text-muted-foreground mb-4">{title}</div>
      <div className="flex items-baseline gap-2">
        <div
          className="text-3xl font-display font-bold"
          style={{ letterSpacing: "-0.02em", color: `rgba(${accentRgb},1)`, fontVariantNumeric: "tabular-nums" }}
        >
          {val}
        </div>
        <div
          className="text-xs font-mono px-2 py-1"
          style={{
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: "6px",
            color: "#94A3B8",
          }}
        >
          {label}
        </div>
      </div>
    </div>
  );
}
