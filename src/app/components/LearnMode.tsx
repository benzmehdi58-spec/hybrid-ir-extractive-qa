import { Outlet, NavLink } from "react-router";

export function LearnMode() {
  return (
    <div className="flex-1 flex flex-col overflow-hidden" style={{ position: "relative", zIndex: 1 }}>
      {/* Frosted glass tab bar */}
      <div
        className="px-8"
        style={{
          background: "rgba(10,20,40,0.50)",
          backdropFilter: "blur(20px) saturate(160%)",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
          boxShadow: "0 1px 0 rgba(0,212,255,0.04)",
        }}
      >
        <div className="max-w-5xl mx-auto flex">
          <TabLink to="/learn" end>Pipeline Overview</TabLink>
          <TabLink to="/learn/ir">Module 1: IR</TabLink>
          <TabLink to="/learn/summ">Module 2: Summarization</TabLink>
          <TabLink to="/learn/qa">Module 3: QA — BiLSTM</TabLink>
          <TabLink to="/learn/qa-distilbert">Module 4: QA — DistilBERT</TabLink>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-5xl mx-auto">
          <Outlet />
        </div>
      </div>
    </div>
  );
}

function TabLink({ to, children, end }: { to: string; children: React.ReactNode; end?: boolean }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        `px-6 py-4 text-sm font-mono border-b-2 transition-all duration-200 ${
          isActive
            ? "border-primary text-primary"
            : "border-transparent text-muted-foreground hover:text-foreground hover:border-[rgba(255,255,255,0.15)]"
        }`
      }
      style={({ isActive }) => ({
        letterSpacing: "0.05em",
        boxShadow: isActive ? "0 2px 0 rgba(0,212,255,0.20)" : "none",
      })}
    >
      {children}
    </NavLink>
  );
}
