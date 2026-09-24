import { useMemo, useState } from "react";
import { Detection } from "../api";

const NO_DETECTIONS: Detection[] = [];
import { api } from "../api";
import { Badge, ErrorBox, PctBar, SectionTitle, TierBadge, classColor, useAsync } from "../components/ui";

export default function ExplorerPage() {
  const surveys = useAsync(() => api.surveys(), []);
  const [surveyId, setSurveyId] = useState<string>("all");
  const [q, setQ] = useState("");
  const [status, setStatus] = useState("all");
  const [tier, setTier] = useState("all");
  const [cls, setCls] = useState("all");
  const [selId, setSelId] = useState<string | null>(null);

  const surveyChoices = surveys.data ?? [];
  const allDets = useAsync<Detection[]>(
    () => {
      if (surveyId !== "all") {
        return api.surveyDetections(surveyId);
      }
      const ids = surveyChoices.map((s) => s.id);
      if (ids.length === 0) return Promise.resolve([]);
      return Promise.all(ids.map((id) => api.surveyDetections(id))).then((arr) => arr.flat());
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [surveyId, surveyChoices.length]
  );

  const rows = allDets.data ?? NO_DETECTIONS;
  const classes = Array.from(new Set(rows.map((d) => d.class_name))).sort();
  const filtered = useMemo(
    () =>
      rows.filter((d) => {
        if (status !== "all" && d.status !== status) return false;
        if (tier !== "all" && d.priority.tier !== tier) return false;
        if (cls !== "all" && d.class_name !== cls) return false;
        if (q && !`${d.image_filename} ${d.class_name} ${d.id}`.toLowerCase().includes(q.toLowerCase())) return false;
        return true;
      }),
    [rows, status, tier, cls, q]
  );

  const sel = rows.find((d) => d.id === selId) ?? filtered[0] ?? null;

  return (
    <div>
      <SectionTitle title="Detection Explorer" sub="Every detection across surveys — raw signal, fused evidence, geolocation and dimensions." />

      <div className="card card-pad mb-4">
        <div className="flex flex-wrap items-center gap-3">
          <select className="input !w-60" value={surveyId} onChange={(e) => setSurveyId(e.target.value)}>
            <option value="all">All surveys</option>
            {surveyChoices.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
          <input className="input !w-56" placeholder="Search filename / class…" value={q} onChange={(e) => setQ(e.target.value)} />
          <select className="input !w-40" value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="all">All statuses</option>
            <option value="confirmed">Confirmed</option>
            <option value="review">Review</option>
            <option value="candidate">Candidate</option>
            <option value="human_review_required">Human review</option>
          </select>
          <select className="input !w-40" value={tier} onChange={(e) => setTier(e.target.value)}>
            <option value="all">All tiers</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
          <select className="input !w-40" value={cls} onChange={(e) => setCls(e.target.value)}>
            <option value="all">All classes</option>
            {classes.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <span className="text-xs text-slate-500">{filtered.length} of {rows.length}</span>
        </div>
      </div>

      {allDets.error && <ErrorBox message={allDets.error} />}

      <div className="grid gap-4 xl:grid-cols-[1.6fr_1fr]">
        <div className="card overflow-hidden">
          <div className="max-h-[70vh] overflow-y-auto scroll-thin">
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-abyss-900 text-[10px] uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="px-3 py-2">Class</th>
                  <th className="px-3 py-2">Survey / image</th>
                  <th className="px-3 py-2">Status</th>
                  <th className="px-3 py-2 text-right">Fusion</th>
                  <th className="px-3 py-2 text-right">Priority</th>
                  <th className="px-3 py-2 text-right">Geo</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700/40">
                {filtered.map((d) => (
                  <tr
                    key={d.id}
                    onClick={() => setSelId(d.id)}
                    className={`cursor-pointer transition-colors ${sel?.id === d.id ? "bg-sonar-500/10" : "hover:bg-abyss-850/60"}`}
                  >
                    <td className="px-3 py-2">
                      <div className="flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full" style={{ background: classColor(d.class_name) }} />
                        <span className="font-medium text-slate-200">{d.class_name.replace(/_/g, " ")}</span>
                      </div>
                    </td>
                    <td className="px-3 py-2 text-slate-400">
                      <div className="max-w-44 truncate">{d.image_filename}</div>
                      <div className="text-[10px] text-slate-600">{d.survey_id}</div>
                    </td>
                    <td className="px-3 py-2"><Badge value={d.status} /></td>
                    <td className="px-3 py-2 text-right">
                      <span className="mono font-semibold text-sonar-400">{Math.round(d.evidence.fusion * 100)}%</span>
                    </td>
                    <td className="px-3 py-2 text-right"><TierBadge tier={d.priority.tier} score={d.priority.score} /></td>
                    <td className="px-3 py-2 text-right mono text-slate-500">
                      {d.geolocation.known ? "✓ fix" : "—"}
                    </td>
                  </tr>
                ))}
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-3 py-10 text-center text-slate-500">
                      No detections match the filters.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card card-pad h-fit sticky top-0">
          {!sel ? (
            <p className="text-sm text-slate-500">Select a detection to inspect its full record.</p>
          ) : (
            <DetailPanel d={sel} />
          )}
        </div>
      </div>
    </div>
  );
}

function DetailPanel({ d }: { d: Detection }) {
  const signals = (d.evidence.signals ?? {}) as Record<string, number | null | undefined>;
  const label: Record<string, string> = { detection: "Detector", segmentation: "Seg.", natural: "Artificial", shadow: "Shadow", consistency: "Consistency" };
  return (
    <div>
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-sm font-bold text-slate-100">{d.class_name.replace(/_/g, " ")}</div>
          <div className="mono text-[10px] text-slate-500">{d.id}</div>
          <div className="mt-0.5 text-[11px] text-slate-500">
            {d.image_filename} · {d.survey_id}
          </div>
        </div>
        <TierBadge tier={d.priority.tier} score={d.priority.score} />
      </div>

      <div className="mt-3 grid grid-cols-2 gap-1.5 text-[11px]">
        <Info k="Status" v={d.status.replace(/_/g, " ")} />
        <Info k="Class confidence" v={`${Math.round(d.class_confidence * 100)}%`} />
        <Info k="Box" v={`${Math.round(d.box.x)},${Math.round(d.box.y)} ${Math.round(d.box.w)}×${Math.round(d.box.h)}`} />
        <Info k="Mask coverage" v={d.mask_area_frac != null ? `${(d.mask_area_frac * 100).toFixed(0)}% of box` : "—"} />
      </div>

      <h4 className="label mb-1 mt-4">Evidence signals</h4>
      <div className="space-y-1">
        {Object.entries(label).map(([k, l]) => {
          const v = signals[k];
          if (v == null) return null;
          return (
            <div key={k} className="flex items-center gap-2">
              <span className="w-20 text-[11px] text-slate-400">{l}</span>
              <PctBar value={Number(v)} className="flex-1" />
              <span className="w-8 text-right mono text-[11px]">{Math.round(Number(v) * 100)}</span>
            </div>
          );
        })}
      </div>

      <h4 className="label mb-1 mt-4">Backends</h4>
      <div className="mono text-[10px] text-slate-500">
        {Object.entries(d.evidence.backend_used ?? {})
          .map(([k, v]) => `${k}:${v}`)
          .join(" · ")}
      </div>

      <h4 className="label mb-1 mt-4">Geolocation</h4>
      {d.geolocation.known ? (
        <div className="mono text-[11px] text-slate-300">
          {d.geolocation.lat?.toFixed(6)}, {d.geolocation.lon?.toFixed(6)}
          {d.geolocation.uncertainty_m != null && <div className="text-slate-500">±{d.geolocation.uncertainty_m.toFixed(0)} m (2σ ellipse)</div>}
        </div>
      ) : (
        <p className="text-[11px] text-slate-500">{d.geolocation.note || "no fix (metadata incomplete)"}</p>
      )}

      <h4 className="label mb-1 mt-4">Dimensions</h4>
      <p className="text-[11px] text-slate-400">
        {d.dimensions.estimable
          ? `w ${d.dimensions.width_m?.toFixed(1)} m${d.dimensions.height_m != null ? ` · h ${d.dimensions.height_m.toFixed(1)} m (shadow)` : ""}`
          : d.dimensions.note || "not estimable without range metadata"}
      </p>

      <h4 className="label mb-1 mt-4">Priority factors</h4>
      <div className="grid grid-cols-2 gap-1 text-[11px] text-slate-400">
        {Object.entries(d.priority.factors ?? {}).map(([k, v]) => (
          <div key={k} className="flex justify-between rounded bg-abyss-850/60 px-1.5 py-0.5">
            <span className="capitalize text-slate-500">{k.replace(/_/g, " ")}</span>
            <span className="mono">{Number(v).toFixed(2)}</span>
          </div>
        ))}
      </div>

      {d.shadow?.available && (
        <>
          <h4 className="label mb-1 mt-4">Acoustic shadow</h4>
          <p className="text-[11px] text-slate-400">
            {d.shadow.valid ? `valid shadow, ${d.shadow.length_m?.toFixed(1) ?? "?"} m` : "candidate region present"}
            {d.shadow.height_estimate_m != null ? ` → height ~${d.shadow.height_estimate_m.toFixed(1)} m` : ""} · {d.shadow.note}
          </p>
        </>
      )}
      <div className="mt-3">
        <a
          href={`#/lab/${d.survey_id}`}
          className="text-[11px] text-sonar-400 hover:underline"
          onClick={() => {
            /* LabPage keeps same survey */
          }}
        >
          Open in Sonar Lab →
        </a>
      </div>
    </div>
  );
}

function Info({ k, v }: { k: string; v: string }) {
  return (
    <div className="rounded bg-abyss-850/60 px-2 py-1">
      <span className="text-slate-500">{k}: </span>
      <span className="font-medium text-slate-300">{v}</span>
    </div>
  );
}
