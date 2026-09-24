import { FormEvent, useState } from "react";
import { Link } from "react-router-dom";
import { api, SampleInfo } from "../api";
import JobProgress from "../components/JobProgress";
import { ErrorBox, SectionTitle, useAsync } from "../components/ui";

export default function UploadPage() {
  const samples = useAsync(() => api.samples(), []);
  const [files, setFiles] = useState<File[]>([]);
  const [meta, setMeta] = useState({
    name: "",
    sonar_type: "sss",
    lat: "",
    lon: "",
    heading_deg: "",
    altitude_m: "",
    range_m: "",
    side: "starboard",
  });
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [started, setStarted] = useState<{ surveyId: string; jobId: string } | null>(null);
  const [importing, setImporting] = useState<string | null>(null);

  const num = (v: string) => (v.trim() === "" ? null : Number(v));

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (files.length === 0) {
      setErr("Select at least one image file.");
      return;
    }
    setErr(null);
    setBusy(true);
    setStarted(null);
    try {
      const payload: Record<string, unknown> = {
        name: meta.name || files[0].name.replace(/\.[^.]+$/, ""),
        sonar_type: meta.sonar_type,
        preprocess_preset: "light",
      };
      for (const [k, v] of Object.entries({
        lat: num(meta.lat),
        lon: num(meta.lon),
        heading_deg: num(meta.heading_deg),
        altitude_m: num(meta.altitude_m),
        range_m: num(meta.range_m),
      })) {
        if (v !== null) payload[k] = v;
      }
      if (meta.side) payload.side = meta.side;
      const res = await api.createSurvey(files, payload, {});
      setStarted({ surveyId: res.survey_id, jobId: res.job_id });
      setFiles([]);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function importSample(s: SampleInfo) {
    setImporting(s.id);
    setErr(null);
    try {
      const res = await api.importSample(s.id);
      setStarted({ surveyId: res.survey_id, jobId: res.job_id });
    } catch (e) {
      setErr(`Sample import failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setImporting(null);
    }
  }

  const field = "w-24 shrink-0 text-xs font-semibold uppercase tracking-wide text-slate-400 self-center";

  return (
    <div>
      <SectionTitle title="Upload Survey" sub="Upload side-scan sonar (SSS/FLS) images plus vessel metadata. Raw files are preserved untouched." />

      {err && <div className="mb-4"><ErrorBox message={err} /></div>}

      <div className="grid gap-4 lg:grid-cols-2">
        <form onSubmit={submit} className="card card-pad space-y-4">
          <div>
            <label className="label mb-1">Survey name</label>
            <input className="input" value={meta.name} onChange={(e) => setMeta({ ...meta, name: e.target.value })} placeholder="e.g. Goa shelf survey — line 03" />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="label mb-1">Sonar type</label>
              <select className="input" value={meta.sonar_type} onChange={(e) => setMeta({ ...meta, sonar_type: e.target.value })}>
                <option value="sss">Side-scan sonar (SSS)</option>
                <option value="fls">Forward-looking sonar (FLS)</option>
                <option value="unknown">Unknown</option>
              </select>
            </div>
            <div>
              <label className="label mb-1">Imaging side</label>
              <select className="input" value={meta.side} onChange={(e) => setMeta({ ...meta, side: e.target.value })}>
                <option value="starboard">Starboard</option>
                <option value="port">Port</option>
                <option value="unknown">Unknown</option>
              </select>
            </div>
          </div>

          <div className="rounded-lg border border-slate-700 bg-abyss-850/60 p-3">
            <div className="label mb-2">Vessel / survey metadata (optional)</div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-2">
              {(
                [
                  ["lat", "Vessel lat", "e.g. 17.68"],
                  ["lon", "Vessel lon", "e.g. 83.31"],
                  ["heading_deg", "Heading °", "e.g. 90"],
                  ["altitude_m", "Altitude m", "e.g. 15"],
                  ["range_m", "Range m", "e.g. 60"],
                ] as const
              ).map(([k, label, ph]) => (
                <div key={k} className="flex items-center gap-2">
                  <label className={field}>{label}</label>
                  <input
                    className="input"
                    type="number"
                    step="any"
                    value={String(meta[k as keyof typeof meta])}
                    placeholder={ph}
                    onChange={(e) => setMeta({ ...meta, [k]: e.target.value })}
                  />
                </div>
              ))}
            </div>
            <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
              The system never fabricates coordinates: without GPS/heading/altitude, objects are flagged with geolocation{" "}
              <em>unavailable</em> instead of an invented fix.
            </p>
          </div>

          <div>
            <label className="label mb-1">Sonar images</label>
            <label
              className={`flex cursor-pointer flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed px-4 py-8 text-center transition-all duration-200 ${
                dragOver
                  ? "scale-[1.01] border-sonar-400 bg-sonar-500/10 shadow-[0_0_24px_rgba(34,211,238,0.15)]"
                  : "border-slate-600 bg-abyss-850/40 hover:border-sonar-500"
              }`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                if (e.dataTransfer.files.length) setFiles(Array.from(e.dataTransfer.files));
              }}
            >
              <span className="text-sm text-slate-300">{files.length ? `${files.length} file(s) selected` : "Click or drop image files here"}</span>
              <span className="text-xs text-slate-500">PNG / JPG / TIFF — raw bytes stored unchanged</span>
              <input
                type="file"
                multiple
                accept="image/*"
                className="hidden"
                onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
              />
            </label>
            {files.length > 0 && (
              <ul className="mt-2 max-h-28 space-y-0.5 overflow-y-auto text-xs text-slate-400 scroll-thin">
                {files.map((f) => (
                  <li key={f.name} className="animate-fade-in truncate mono">
                    {f.name} <span className="text-slate-600">({Math.round(f.size / 1024)} KB)</span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <button className="btn-primary w-full" disabled={busy}>
            {busy ? "Uploading…" : "Upload & analyze survey"}
          </button>
        </form>

        <div>
          <div className="card card-pad">
            <h2 className="label mb-2">One-click demo samples</h2>
            <p className="mb-3 text-xs text-slate-500">
              Pre-bundled surveys — a synthetic SSS line with GPS metadata and a subset of the real MD-FLS watertank dataset —
              so the full pipeline can be demonstrated without any upload.
            </p>
            {samples.loading && <p className="text-sm text-slate-500">Loading samples…</p>}
            {samples.error && <ErrorBox message={samples.error} />}
            <div className="space-y-2">
              {(samples.data ?? []).map((s) => (
                <div key={s.id} className="rounded-lg border border-slate-700 bg-abyss-850/60 p-3">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 text-sm font-medium text-slate-200">
                        <span className="truncate">{s.name}</span>
                        {s.synthetic && <span className="rounded bg-violet-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-violet-300">SYNTHETIC</span>}
                      </div>
                      <div className="mt-0.5 text-xs text-slate-500">
                        {s.image_count} images · {s.size_kb} KB
                      </div>
                    </div>
                    <button className="btn-ghost shrink-0 !px-2.5 !py-1 text-xs" disabled={importing === s.id} onClick={() => importSample(s)}>
                      {importing === s.id ? "Importing…" : "Import"}
                    </button>
                  </div>
                  <p className="mt-1 line-clamp-2 text-[11px] leading-relaxed text-slate-500">{s.description}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {started && (
        <div className="mt-4 space-y-2">
          <JobProgress jobId={started.jobId} />
          <div className="text-sm text-slate-400">
            When analysis finishes, open it in the{" "}
            <Link to={`/lab/${started.surveyId}`} className="text-sonar-400 hover:underline">
              Sonar Intelligence Lab
            </Link>
            .
          </div>
        </div>
      )}
    </div>
  );
}
