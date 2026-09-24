import { ReactNode, useCallback, useEffect, useState } from "react";

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): { data: T | null; error: string | null; loading: boolean; reload: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  useEffect(() => {
    let live = true;
    setLoading(true);
    setError(null);
    fn()
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, ...deps]);

  return { data, error, loading, reload };
}

export function useInterval(cb: () => void, ms: number | null) {
  useEffect(() => {
    if (ms === null) return;
    const id = window.setInterval(cb, ms);
    return () => window.clearInterval(id);
  }, [ms, cb]);
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-400">
      <div className="h-4 w-4 animate-spin rounded-full border-2 border-sonar-500 border-t-transparent" />
      {label && <span>{label}</span>}
    </div>
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: ReactNode }) {
  return (
    <div className="card card-pad flex flex-col items-center justify-center gap-2 py-16 text-center">
      <div className="text-3xl">🌊</div>
      <div className="font-medium text-slate-300">{title}</div>
      {hint && <div className="max-w-md text-sm text-slate-500">{hint}</div>}
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
      <span className="font-semibold">Error — </span>
      {message}
    </div>
  );
}

const STATUS_STYLE: Record<string, string> = {
  confirmed: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
  review: "bg-amber-500/15 text-amber-300 border-amber-500/40",
  candidate: "bg-sky-500/15 text-sky-300 border-sky-500/40",
  human_review_required: "bg-rose-500/15 text-rose-300 border-rose-500/40",
  done: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
  running: "bg-cyan-500/15 text-cyan-300 border-cyan-500/40",
  queued: "bg-slate-500/15 text-slate-300 border-slate-500/40",
  failed: "bg-rose-500/15 text-rose-300 border-rose-500/40",
  created: "bg-slate-500/15 text-slate-300 border-slate-500/40",
};

export function Badge({ value, label }: { value: string; label?: string }) {
  const cls = STATUS_STYLE[value] ?? "bg-slate-500/15 text-slate-300 border-slate-500/40";
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium ${cls}`}>
      {label ?? value.replace(/_/g, " ")}
    </span>
  );
}

const TIER_STYLE: Record<string, string> = {
  critical: "text-rose-400 border-rose-500/50 bg-rose-500/10",
  high: "text-orange-400 border-orange-500/50 bg-orange-500/10",
  medium: "text-amber-300 border-amber-500/50 bg-amber-500/10",
  low: "text-slate-400 border-slate-600 bg-slate-600/10",
};

export function TierBadge({ tier, score }: { tier: string; score?: number }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${TIER_STYLE[tier] ?? TIER_STYLE.low}`}>
      {tier}
      {score !== undefined && <span className="font-normal opacity-80">{Math.round(score)}</span>}
    </span>
  );
}

export function PctBar({ value, color = "bg-sonar-500", className = "" }: { value: number; color?: string; className?: string }) {
  const v = Math.max(0, Math.min(100, value * 100));
  return (
    <div className={`h-1.5 w-full overflow-hidden rounded-full bg-slate-700/60 ${className}`}>
      <div className={`h-full rounded-full ${color}`} style={{ width: `${v}%` }} />
    </div>
  );
}

export function Stat({ label, value, sub, accent }: { label: string; value: ReactNode; sub?: string; accent?: string }) {
  return (
    <div className="card card-hover card-pad">
      <div className="label">{label}</div>
      <div className={`mt-1 text-3xl font-bold ${accent ?? "text-slate-100"}`}>{value}</div>
      {sub && <div className="mt-0.5 text-xs text-slate-500">{sub}</div>}
    </div>
  );
}

/** §9: staggered entrance — card N rises ~60ms after card N-1. */
export function Stagger({ index, children, className = "" }: { index: number; children: ReactNode; className?: string }) {
  return (
    <div className={`animate-rise ${className}`} style={{ animationDelay: `${Math.min(index, 8) * 60}ms` }}>
      {children}
    </div>
  );
}

/** §27: subtle skeleton block (dark blue-gray base + moving highlight). */
export function Skeleton({ className = "h-4 w-full" }: { className?: string }) {
  return <div className={`aqua-skeleton rounded ${className}`} aria-hidden />;
}

export function SectionTitle({ title, sub, right }: { title: string; sub?: string; right?: ReactNode }) {
  return (
    <div className="mb-4 flex items-end justify-between gap-4">
      <div>
        <h1 className="text-xl font-bold text-slate-100">{title}</h1>
        {sub && <p className="mt-0.5 text-sm text-slate-400">{sub}</p>}
      </div>
      {right}
    </div>
  );
}

// Geolocation v2 (§13): status labels — every coordinate carries one.
export const LOCATION_STATUS_STYLE: Record<string, { symbol: string; cls: string }> = {
  VERIFIED: { symbol: "\u2713", cls: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40" },
  DERIVED: { symbol: "\u2713", cls: "bg-cyan-500/15 text-cyan-300 border-cyan-500/40" },
  MANUAL: { symbol: "\u2691", cls: "bg-violet-500/15 text-violet-300 border-violet-500/40" },
  VESSEL_ONLY: { symbol: "\u25d0", cls: "bg-sky-500/15 text-sky-300 border-sky-500/40" },
  UNAVAILABLE: { symbol: "\u2014", cls: "bg-slate-500/15 text-slate-300 border-slate-500/40" },
  INVALID: { symbol: "\u2715", cls: "bg-rose-500/15 text-rose-300 border-rose-500/40" },
};

export function LocationStatusBadge({ status }: { status: string | null | undefined }) {
  const key = (status ?? "UNAVAILABLE").toUpperCase();
  const s = LOCATION_STATUS_STYLE[key] ?? LOCATION_STATUS_STYLE.UNAVAILABLE;
  const label = key === "VESSEL_ONLY" ? "VESSEL POSITION ONLY" : key;
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold tracking-wide ${s.cls}`}
      title={`Geolocation provenance state: ${key}`}
    >
      <span aria-hidden>{s.symbol}</span>
      {label}
    </span>
  );
}

export function fmtDate(iso: string) {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleString();
}

export function fmtBytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function classColor(cls: string): string {
  // Approved v3 taxonomy display colors. ghost_net/pipe/cylinder/net_like are
  // intentionally unmapped — no supervised classes exist (semantic review v3).
  const map: Record<string, string> = {
    tire: "#2dd4bf",
    wreck: "#fb7185",
    debris: "#facc15",
    structure: "#94a3b8",
    rock: "#64748b",
    unknown: "#94a3b8",
    unknown_anomaly: "#f43f5e",
  };
  return map[cls] ?? "#94a3b8";
}
