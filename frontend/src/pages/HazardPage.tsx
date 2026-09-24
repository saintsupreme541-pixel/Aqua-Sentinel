import { useMemo, useState } from "react";
import { Detection } from "../api";
import { api } from "../api";
import { Badge, ErrorBox, PctBar, SectionTitle, TierBadge, classColor, useAsync } from "../components/ui";

export default function HazardPage() {
  const surveys = useAsync(() => api.surveys(), []);
  const [surveyId, setSurveyId] = useState<string>("");
  const dets = useAsync<Detection[]>(() => (surveyId ? api.surveyDetections(surveyId) : Promise.resolve([])), [surveyId]);
  const [openId, setOpenId] = useState<string | null>(null);

  const choices = surveys.data ?? [];
  const rows = useMemo(() => [...(dets.data ?? [])].sort((a, b) => b.priority.score - a.priority.score), [dets.data]);
  const critical = rows.filter((r) => r.priority.tier === "critical").length;
  const high = rows.filter((r) => r.priority.tier === "high").length;
  const netlike = rows.filter((r) => r.class_name === "ghost_net" || r.class_name === "net_like").length;

  return (
    <div>
      <SectionTitle
        title="Hazard Intelligence"
        sub="Ranked marine cleanup priorities. Every score is the transparent weighted sum of six explainable factors."
        right={
          <select className="input !w-80" value={surveyId} onChange={(e) => { setSurveyId(e.target.value); setOpenId(null); }}>
            <option value="">Select a survey…</option>
            {choices.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
        }
      />

      {surveys.error && <ErrorBox message={surveys.error} />}
      {!surveyId && <div className="text-sm text-slate-400">Pick a survey to rank its hazards for cleanup planning.</div>}

      {surveyId && (
        <>
          <div className="mb-4 grid grid-cols-2 gap-4 md:grid-cols-4">
            <Mini label="Objects" value={rows.length} />
            <Mini label="Critical" value={critical} accent="text-rose-400" />
            <Mini label="High" value={high} accent="text-orange-400" />
            <Mini label="Net-like (entanglement)" value={netlike} accent="text-sonar-400" />
          </div>

          <div className="space-y-2">
            {rows.length === 0 && <div className="card card-pad text-sm text-slate-500">No detections in this survey.</div>}
            {rows.map((d, idx) => {
              const open = openId === d.id;
              return (
                <div key={d.id} className="card overflow-hidden">
                  <button className="flex w-full items-center gap-3 px-4 py-3 text-left" onClick={() => setOpenId(open ? null : d.id)}>
                    <span className="w-8 shrink-0 text-right mono text-sm text-slate-600">{idx + 1}</span>
                    <span className="h-3 w-3 shrink-0 rounded-full" style={{ background: classColor(d.class_name) }} />
                    <span className="w-36 shrink-0 truncate text-sm font-semibold text-slate-100">
                      {d.class_name.replace(/_/g, " ")}
                    </span>
                    <div className="hidden w-40 shrink-0 md:block">
                      <Badge value={d.status} />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-700/60">
                          <div
                            className={`h-full rounded ${
                              d.priority.tier === "critical" ? "bg-rose-500" : d.priority.tier === "high" ? "bg-orange-500" : d.priority.tier === "medium" ? "bg-amber-400" : "bg-slate-500"
                            }`}
                            style={{ width: `${d.priority.score}%` }}
                          />
                        </div>
                        <span className="w-12 text-right mono text-sm font-bold text-slate-200">{Math.round(d.priority.score)}</span>
                      </div>
                      <div className="mt-0.5 truncate text-[11px] text-slate-600">{d.image_filename}</div>
                    </div>
                    <TierBadge tier={d.priority.tier} />
                  </button>
                  {open && <Factors d={d} />}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

const FACTOR_LABEL: Record<string, string> = {
  type_risk: "Type risk",
  size: "Size",
  entanglement: "Entanglement risk",
  environmental: "Environmental risk",
  confidence: "Detection confidence",
  location: "Location priority",
};

const FACTOR_W = { type_risk: 0.2, size: 0.1, entanglement: 0.25, environmental: 0.2, confidence: 0.15, location: 0.1 };

function Factors({ d }: { d: Detection }) {
  return (
    <div className="border-t border-slate-700/50 bg-abyss-850/40 px-4 py-3">
      <div className="grid gap-3 md:grid-cols-[1.3fr_1fr]">
        <div>
          <div className="label mb-1">Why this score?</div>
          <div className="space-y-1">
            {Object.entries(FACTOR_LABEL).map(([k, label]) => {
              const v = d.priority.factors?.[k] ?? 0;
              return (
                <div key={k} className="flex items-center gap-2 text-xs">
                  <span className="w-36 shrink-0 text-slate-400">{label}</span>
                  <PctBar value={Number(v)} className="flex-1" color={k === "entanglement" ? "bg-rose-400" : k === "type_risk" ? "bg-orange-400" : "bg-sonar-400"} />
                  <span className="w-16 text-right mono text-[11px] text-slate-400">×{FACTOR_W[k as keyof typeof FACTOR_W]}</span>
                </div>
              );
            })}
          </div>
        </div>
        <div className="space-y-1 text-[11px] text-slate-500">
          <p>
            <span className="text-slate-400">Location:</span>{" "}
            {d.geolocation.known ? (
              <span className="mono">
                {d.geolocation.lat?.toFixed(6)}, {d.geolocation.lon?.toFixed(6)} ±{d.geolocation.uncertainty_m?.toFixed(0)} m
              </span>
            ) : (
              d.geolocation.note || "no metadata fix"
            )}
          </p>
          <p>
            <span className="text-slate-400">Extent:</span>{" "}
            {d.dimensions.estimable
              ? `${d.dimensions.width_m?.toFixed(1) ?? "?"} m wide${d.dimensions.length_m ? ` × ${d.dimensions.length_m.toFixed(1)} m` : ""}${d.dimensions.height_m != null ? `, ~${d.dimensions.height_m.toFixed(1)} m tall` : ""}`
              : d.dimensions.note || "not estimable"}
          </p>
          <p>
            <span className="text-slate-400">Evidence:</span> fusion {Math.round(d.evidence.fusion * 100)}% ·{" "}
            {d.evidence.availability?.shadow ? "shadow ✓" : "no shadow geometry"} ·{" "}
            {d.evidence.availability?.consistency ? "multi-frame ✓" : "single-frame"}
          </p>
          <p className="text-slate-600">
            Guidance: {d.priority.tier === "critical" ? "dispatch cleanup team — prioritize for survey dive" : d.priority.tier === "high" ? "schedule removal within the current campaign" : d.priority.tier === "medium" ? "log for follow-up survey" : "low urgency — verify before dispatching"}
          </p>
        </div>
      </div>
    </div>
  );
}

function Mini({ label, value, accent }: { label: string; value: number; accent?: string }) {
  return (
    <div className="card card-pad">
      <div className="label">{label}</div>
      <div className={`mt-1 text-2xl font-bold ${accent ?? "text-slate-100"}`}>{value}</div>
    </div>
  );
}
