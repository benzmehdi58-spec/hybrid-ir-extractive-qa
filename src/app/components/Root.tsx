import { Outlet, useLocation, useNavigate } from "react-router";
import { Github } from "lucide-react";
import { motion } from "motion/react";
import { BackgroundScene } from "./BackgroundScene";

export function Root() {
  const location = useLocation();
  const navigate = useNavigate();
  const isLearnMode = location.pathname.startsWith("/learn");

  return (
    <div className="min-h-screen text-foreground font-sans selection:bg-primary/30" style={{ backgroundColor: "#020510" }}>
      <BackgroundScene />

      {/* Frosted glass nav */}
      <nav
        className="fixed top-0 left-0 right-0 h-14 z-50 flex items-center justify-between px-6"
        style={{
          background: "rgba(2, 5, 16, 0.70)",
          backdropFilter: "blur(20px) saturate(160%)",
          borderBottom: "1px solid rgba(255, 255, 255, 0.06)",
          boxShadow: "0 1px 0 rgba(0, 212, 255, 0.05), 0 4px 24px rgba(0, 0, 0, 0.4)",
        }}
      >
        {/* Left */}
        <div className="flex items-center gap-4">
          <div className="flex flex-col gap-[3px]">
            <div className="w-5 h-1 bg-primary rounded-full" style={{ boxShadow: "0 0 6px rgba(0,212,255,0.6)" }} />
            <div className="w-4 h-1 bg-secondary rounded-full" style={{ boxShadow: "0 0 6px rgba(245,158,11,0.6)" }} />
            <div className="w-3 h-1 bg-accent rounded-full" style={{ boxShadow: "0 0 6px rgba(167,139,250,0.6)" }} />
          </div>
          <div className="flex items-baseline gap-2">
            <span
              className="font-display font-bold text-lg tracking-tight"
              style={{
                background: "linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
                letterSpacing: "-0.02em",
              }}
            >
              NLP Pipeline
            </span>
            <span className="text-muted-foreground text-xs font-medium tracking-wide">Open-Domain QA System</span>
          </div>
        </div>

        {/* Center toggle */}
        <div
          className="flex rounded-full p-[3px] relative"
          style={{
            background: "rgba(0, 0, 0, 0.40)",
            border: "1px solid rgba(255,255,255,0.10)",
            boxShadow: "inset 0 1px 3px rgba(0,0,0,0.5)",
          }}
        >
          {/* Sliding active pill */}
          <motion.div
            layoutId="nav-pill"
            className="absolute top-[3px] bottom-[3px] rounded-full z-0"
            style={{
              background: isLearnMode
                ? "rgba(0, 212, 255, 0.15)"
                : "rgba(245, 158, 11, 0.15)",
              border: isLearnMode
                ? "1px solid rgba(0, 212, 255, 0.40)"
                : "1px solid rgba(245, 158, 11, 0.40)",
              boxShadow: isLearnMode
                ? "0 0 12px rgba(0,212,255,0.18), inset 0 1px 0 rgba(255,255,255,0.08)"
                : "0 0 12px rgba(245,158,11,0.18), inset 0 1px 0 rgba(255,255,255,0.08)",
              backdropFilter: "blur(8px)",
            }}
            animate={{
              left: isLearnMode ? "calc(50% + 3px)" : "3px",
              right: isLearnMode ? "3px" : "calc(50% + 3px)",
            }}
            transition={{ type: "spring", stiffness: 400, damping: 30 }}
          />

          <button
            onClick={() => navigate("/")}
            className="relative z-10 flex items-center gap-1.5 px-4 py-1.5 text-sm font-medium rounded-full transition-colors"
            style={{
              color: !isLearnMode ? "#F59E0B" : "rgba(148,163,184,0.70)",
              minWidth: "110px",
              justifyContent: "center",
            }}
          >
            <span style={{ fontSize: "13px" }}>⚡</span>
            Demo Mode
          </button>

          <button
            onClick={() => navigate("/learn")}
            className="relative z-10 flex items-center gap-1.5 px-4 py-1.5 text-sm font-medium rounded-full transition-colors"
            style={{
              color: isLearnMode ? "#00D4FF" : "rgba(148,163,184,0.70)",
              minWidth: "110px",
              justifyContent: "center",
            }}
          >
            <span style={{ fontSize: "13px" }}>📖</span>
            Learn Mode
          </button>
        </div>

        {/* Right */}
        <div className="flex items-center gap-6">
          <div
            className="font-mono text-[10px] text-muted-foreground px-2 py-1 rounded"
            style={{
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.08)",
              letterSpacing: "0.05em",
            }}
          >
            SQuAD 2.0 + CNN/DailyMail
          </div>
          <div className="flex items-center gap-3">
            <a
              href="https://github.com/benzmehdi58-spec/hybrid-ir-extractive-qa"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="GitHub Repository"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <Github className="w-5 h-5" />
            </a>
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="pt-14 h-screen overflow-hidden flex flex-col" style={{ position: "relative", zIndex: 1 }}>
        <Outlet />
      </main>
    </div>
  );
}
