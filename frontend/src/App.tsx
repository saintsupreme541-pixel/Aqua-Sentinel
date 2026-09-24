import { useEffect, useState } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import { api } from "./api";
import {
  Activity,
  FileText,
  FlaskConical,
  LayoutDashboard,
  Map,
  ScanSearch,
  TriangleAlert,
  Upload,
} from "lucide-react";
import logo from "./assets/logo.png";
import { AmbientBackground } from "./components/AmbientBackground";
import DashboardPage from "./pages/DashboardPage";
import UploadPage from "./pages/UploadPage";
import LabPage from "./pages/LabPage";
import ExplorerPage from "./pages/ExplorerPage";
import MapPage from "./pages/MapPage";
import HazardPage from "./pages/HazardPage";
import ReportsPage from "./pages/ReportsPage";

// Real routes only — the reference navigation names are adapted to the pages
// that actually exist; nothing is fabricated (sidebar contract).
const NAV: Array<{ section: string; items: Array<{ to: string; label: string; icon: typeof Map; end?: boolean }> }> = [
  {
    section: "Operations",
    items: [
      { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
      { to: "/upload", label: "Upload Survey", icon: Upload },
      { to: "/lab", label: "Sonar Intelligence Lab", icon: FlaskConical },
    ],
  },
  {
    section: "Intelligence",
    items: [
      { to: "/explorer", label: "Detection Explorer", icon: ScanSearch },
      { to: "/map", label: "Marine Intelligence Map", icon: Map },
      { to: "/hazards", label: "Hazard Intelligence", icon: TriangleAlert },
      { to: "/reports", label: "Reports", icon: FileText },
    ],
  },
];

interface HealthState {
  ok: boolean;
  mode: string | null;
  backends: Record<string, string>;
}

export default function App() {
  // Honest connectivity/degraded-mode surface: never silently present
  // heuristic or stale data as trained-model results.
  const [health, setHealth] = useState<HealthState>({ ok: true, mode: null, backends: {} });
  useEffect(() => {
    api
      .health()
      .then((h) => setHealth({ ok: true, mode: h.mode ?? null, backends: h.model_backends ?? {} }))
      .catch(() => setHealth({ ok: false, mode: null, backends: {} }));
  }, []);

  // §5: subtle page-enter transition on every route change (opacity+8px rise,
  // ~300–450ms — never a dramatic fly-in). Reduced-motion disables it in CSS.
  const location = useLocation();
  const [pageKey, setPageKey] = useState(location.pathname);
  useEffect(() => {
    setPageKey(location.pathname);
  }, [location.pathname]);

  const backendLabel = Object.entries(health.backends)
    .map(([k, v]) => `${k} ${v.toUpperCase()}`)
    .join(" · ");

  return (
    <div className="flex h-screen flex-col aqua-vignette">
      <AmbientBackground />
      {/* ── Top header ─────────────────────────────────────────────────── */}
      <header className="relative z-10 flex h-14 shrink-0 items-center gap-4 border-b border-slate-700/50 bg-abyss-900/90 px-4 shadow-lg shadow-black/30 backdrop-blur-sm">
        <div className="flex items-center gap-2.5">
          <img
            src={logo}
            alt="AQUA-SENTINEL logo"
            className="h-9 w-9 shrink-0 rounded-lg object-contain ring-1 ring-sonar-500/25"
          />
          <div className="leading-tight">
            <div className="aqua-shine text-sm font-bold tracking-wide text-slate-100">AQUA-SENTINEL</div>
            <div className="text-[10px] font-medium uppercase tracking-widest text-sonar-500">
              AI for Cleaner Oceans
            </div>
          </div>
        </div>

        <div className="hidden min-w-0 flex-1 px-2 md:block">
          <div className="truncate text-[13px] font-semibold text-slate-300">
            Underwater Debris Detection &amp; Marine Intelligence
          </div>
          <div className="text-[10px] uppercase tracking-widest text-slate-500">Scan · Detect · Locate · Protect</div>
        </div>

        {/* Neutral system identity — no fake users or organizations. */}
        <div className="ml-auto flex items-center gap-3">
          {health.ok && (
            <span
              className="hidden items-center gap-1.5 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-1 text-[11px] font-medium text-emerald-300 sm:inline-flex"
              title={backendLabel ? `Model backends: ${backendLabel}` : "Backend reachable"}
            >
              <Activity className="h-3 w-3 animate-status-pulse" />
              {health.mode === "degraded-heuristic" ? "Heuristic baseline" : "System nominal"}
            </span>
          )}
          {!health.ok && (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-rose-500/40 bg-rose-500/10 px-2.5 py-1 text-[11px] font-medium text-rose-300">
              <Activity className="h-3 w-3" />
              Backend unreachable
            </span>
          )}
          <div className="text-right leading-tight">
            <div className="text-[12px] font-semibold text-slate-200">MarineOps</div>
            <div className="text-[10px] text-slate-500">Prototype System</div>
          </div>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* ── Left sidebar ───────────────────────────────────────────────── */}
        <aside className="flex w-60 shrink-0 flex-col border-r border-slate-700/50 bg-abyss-900/90 backdrop-blur-sm">
          <nav className="flex-1 overflow-y-auto scroll-thin px-3 py-4">
            {NAV.map((group) => (
              <div key={group.section} className="mb-4">
                <div className="mb-1.5 px-3 text-[10px] font-semibold uppercase tracking-widest text-slate-500">
                  {group.section}
                </div>
                <div className="space-y-1">
                  {group.items.map((n) => (
                    <NavLink
                      key={n.to}
                      to={n.to}
                      end={n.end}
                      className={({ isActive }) =>
                        `relative flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13px] font-medium transition-all duration-200 ${
                          isActive
                            ? "bg-sonar-500/15 text-sonar-400 shadow-[0_0_10px_rgba(34,211,238,0.12)]"
                            : "text-slate-300 hover:bg-abyss-800 hover:text-slate-100 hover:translate-x-0.5"
                        }`
                      }
                    >
                      {({ isActive }) => (
                        <>
                          {isActive && (
                            <span
                              aria-hidden
                              className="absolute left-0 top-1/2 h-5 w-1 -translate-y-1/2 rounded-r-full bg-sonar-400"
                            />
                          )}
                          <n.icon className="h-4 w-4 shrink-0" strokeWidth={1.75} />
                          {n.label}
                        </>
                      )}
                    </NavLink>
                  ))}
                </div>
              </div>
            ))}
          </nav>
          <div className="border-t border-slate-700/40 px-4 py-3 text-[10px] leading-relaxed text-slate-500">
            Side-scan sonar analysis with multi-stage evidence verification. Estimated locations only — never substitutes
            for a survey-grade fix.
          </div>
        </aside>

        {/* ── Main content ───────────────────────────────────────────────── */}
        <main className="relative z-10 min-w-0 flex-1 overflow-y-auto scroll-thin">
          {!health.ok && (
            <div className="border-b border-rose-500/40 bg-rose-500/15 px-6 py-2 text-xs text-rose-200">
              ⚠ Backend unreachable — analysis is unavailable. Anything displayed is cached data only; results are never
              substituted or faked.
            </div>
          )}
          {health.ok && health.mode === "degraded-heuristic" && (
            <div className="border-b border-amber-500/40 bg-amber-500/10 px-6 py-2 text-xs text-amber-200">
              Running on the deterministic heuristic baseline — no trained weights are loaded. Detections are clearly
              labeled as heuristic, never presented as trained-AI output.
            </div>
          )}
          <div key={pageKey} className="animate-fade-in mx-auto max-w-[1500px] px-6 py-6">
            <Routes>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/lab" element={<LabPage />} />
              <Route path="/lab/:surveyId" element={<LabPage />} />
              <Route path="/explorer" element={<ExplorerPage />} />
              <Route path="/map" element={<MapPage />} />
              <Route path="/hazards" element={<HazardPage />} />
              <Route path="/reports" element={<ReportsPage />} />
            </Routes>
          </div>
        </main>
      </div>

      {/* ── Footer / system status (real /api/health data only) ──────────── */}
      <footer className="relative z-10 flex h-8 shrink-0 items-center gap-4 border-t border-slate-700/50 bg-abyss-900/90 px-4 text-[10px] text-slate-500">
        <span className="flex items-center gap-1.5">
          <span
            className={`h-1.5 w-1.5 rounded-full ${health.ok ? "animate-status-pulse bg-emerald-400" : "bg-rose-500"}`}
            aria-hidden
          />
          {health.ok ? `Backend ${health.mode ?? "ok"}` : "Backend offline"}
        </span>
        {backendLabel && <span className="hidden sm:inline">{backendLabel}</span>}
        <span className="hidden md:inline">Taxonomy: wreck · debris · structure · tire</span>
        <span className="ml-auto hidden lg:inline">Estimated locations only — never substitutes for a survey-grade fix.</span>
      </footer>
    </div>
  );
}
