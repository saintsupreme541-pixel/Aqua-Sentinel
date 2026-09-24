import { useState } from "react";
import { ReportInfo } from "../api";
import { api } from "../api";
import { Badge, ErrorBox, SectionTitle, fmtBytes, fmtDate, useAsync } from "../components/ui";

const FORMATS: Array<{ id: string; label: string; desc: string }> = [
  { id: "csv", label: "CSV", desc: "Flat table of every detection for Excel / GIS" },
  { id: "json", label: "JSON", desc: "Complete structured survey record" },
  { id: "geojson", label: "GeoJSON", desc: "Hazard points + uncertainty polygons for GIS" },
  { id: "pdf", label: "PDF", desc: "Formatted field summary for stakeholders" },
];

export default function ReportsPage() {
  const surveys = useAsync(() => api.surveys(), []);
  const [surveyId, setSurveyId] = useState<string>("");
  const reports = useAsync<ReportInfo[]>(() => (surveyId ? api.reports(surveyId) : Promise.resolve([])), [surveyId]);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const choices = surveys.data ?? [];
  const rows = reports.data ?? [];

  async function generate(fmt: string) {
    if (!surveyId) return;
    setBusy(fmt);
    setErr(null);
    try {
      await api.generateReports(surveyId, [fmt]);
      reports.reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div>
      <SectionTitle
        title="Reports"
        sub="Export the survey analysis for cleanup planning, GIS ingestion and stakeholder review."
        right={
          <select className="input !w-80" value={surveyId} onChange={(e) => setSurveyId(e.target.value)}>
            <option value="">Select a survey…</option>
            {choices.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
        }
      />

      {err && <div className="mb-4"><ErrorBox message={err} /></div>}
      {!surveyId && <div className="text-sm text-slate-400">Pick a survey to generate reports from its detections.</div>}

      {surveyId && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {FORMATS.map((f) => (
              <div key={f.id} className="card card-pad flex flex-col">
                <div className="flex items-center justify-between">
                  <span className="text-lg font-bold uppercase tracking-wide text-sonar-400">{f.label}</span>
                  <span className="rounded bg-abyss-800 px-1.5 py-0.5 mono text-[10px] text-slate-400">.{f.id}</span>
                </div>
                <p className="mt-1 flex-1 text-xs text-slate-500">{f.desc}</p>
                <button className="btn-ghost mt-3 w-full justify-center" disabled={busy === f.id} onClick={() => generate(f.id)}>
                  {busy === f.id ? "Generating…" : `Generate ${f.label}`}
                </button>
              </div>
            ))}
          </div>

          <div className="card card-pad mt-4">
            <h2 className="label mb-2">Generated reports</h2>
            {rows.length === 0 && <p className="text-sm text-slate-500">No reports yet — generate one above.</p>}
            <div className="divide-y divide-slate-700/40">
              {rows.map((r) => (
                <div key={r.id} className="flex items-center gap-3 py-2.5">
                  <Badge value={r.format} label={r.format.toUpperCase()} />
                  <span className="mono min-w-0 flex-1 truncate text-xs text-slate-400">
                    aqua-sentinel-{r.survey_id}.{r.format}
                  </span>
                  <span className="hidden text-xs text-slate-500 sm:block">{fmtBytes(r.size_bytes)}</span>
                  <span className="hidden text-xs text-slate-600 lg:block">{fmtDate(r.created_at)}</span>
                  <a className="btn-ghost !px-2.5 !py-1 text-xs" href={api.reportDownloadUrl(r.id)} download>
                    Download
                  </a>
                </div>
              ))}
            </div>
          </div>

          <div className="card card-pad mt-4 text-[11px] leading-relaxed text-slate-500">
            <span className="label mb-1">Honesty notes on exported data</span>
            <ul className="list-disc space-y-0.5 pl-4">
              <li>Geolocation columns are <em>estimates</em> from survey metadata + sonar geometry; rows without a valid fix carry an explicit "known=false / no fix" marker and a null coordinate — never a fabricated one.</li>
              <li>Each detection records the backend that produced it (heuristic baseline vs ONNX/ultralytics model) and the availability of each evidence signal.</li>
              <li>"Unknown anomaly — human review required" objects are exported in their own status so downstream tools can route them to an analyst.</li>
            </ul>
          </div>
        </>
      )}
    </div>
  );
}
