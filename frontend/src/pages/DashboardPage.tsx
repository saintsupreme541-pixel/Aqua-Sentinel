import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { FileDown, Search } from "lucide-react";
import {
  DashboardData,
  Detection,
  ImageResult,
  PersistentTarget,
  SpatialSummary,
  SurveyDetail,
} from "../api";
import { api } from "../api";
import {
  Badge,
  ErrorBox,
  PctBar,
  SectionTitle,
  Spinner,
  Stat,
  Stagger,
  TierBadge,
  classColor,
  fmtDate,
  useAsync,
} from "../components/ui";
import { TargetMap } from "../components/TargetMap";

const TIER_COLOR: Record<string, string> = {
  critical: "bg-rose-500",
  high: "bg-orange-500",
  medium: "bg-amber-400",
  low: "bg-slate-500",
};

const CLASS_LABEL: Record<string, string> = {
  tire: "Tire",
  wreck: "Wreck",
  structure: "Structure",
  debris: "Debris",
  unknown_anomaly: "Unknown anomaly",
  unknown: "Unknown",
};

const REPORT_FORMATS: Array<{ id: string; label: string }> = [
  { id: "csv", label: "CSV" },
  { id: "json", label: "JSON" },
  { id: "geojson", label: "GeoJSON" },
  { id: "pdf", label: "PDF" },
];

type ViewMode = "original" | "processed" | "detections" | "segmentation" | "evidence";

const VIEWS: Array<{ id: ViewMode; label: string }> = [
  { id: "original", label: "Original" },
  { id: "processed", label: "Processed" },
  { id: "detections", label: "Detections" },
  { id: "segmentation", label: "Segmentation" },
  { id: "evidence", label: "Evidence" },
];

export default function DashboardPage() {
  const navigate = useNavigate();
  const stats = useAsync<DashboardData>(() => api.dashboard(), []);

  // Survey-scoped panels: selection is real (dropdown of actual surveys).
  const surveys = useAsync(() => api.surveys(), []);
  const [surveyId, setSurveyId] = useState<string>("");
  useEffect(() => {
    if (!surveyId && surveys.data && surveys.data.length > 0) setSurveyId(surveys.data[0].id);
  }, [surveys.data, surveyId]);

  const survey = useAsync<SurveyDetail | null>(
    () => (surveyId ? api.survey(surveyId) : Promise.resolve(null)),
    [surveyId]
  );
  const detections = useAsync<Detection[]>(
    () => (surveyId ? api.surveyDetections(surveyId) : Promise.resolve([])),
    [surveyId]
  );
  const spatial = useAsync<SpatialSummary | null>(
    () => (surveyId ? api.spatial(surveyId) : Promise.resolve(null)),
    [surveyId]
  );
  const targets = useAsync<PersistentTarget[]>(
    () => (surveyId ? api.targets(surveyId) : Promise.resolve([])),
    [surveyId]
  );
  const reports = useAsync(() => (surveyId ? api.reports(surveyId) : Promise.resolve([])), [surveyId]);

  // Analysis panel state — real images and views only.
  const [imageId, setImageId] = useState<string | null>(null);
  const images = useMemo(() => survey.data?.images ?? [], [survey.data]);
  // Auto-select a frame only from the LOADED survey matching the current
  // selection — prevents the previous survey's image id being fetched against
  // the newly selected survey while its data is still in flight.
  const surveyMatchesSelection = survey.data?.id === surveyId;
  useEffect(() => {
    if (!surveyMatchesSelection || images.length === 0) return;
    if (!imageId || !images.some((i) => i.id === imageId)) setImageId(images[0].id);
  }, [images, imageId, surveyMatchesSelection]);
  const [view, setView] = useState<ViewMode>("detections");
  const [selDetId, setSelDetId] = useState<string | null>(null);
  const imDets = useMemo(() => (detections.data ?? []).filter((d) => d.image_id === imageId), [detections.data, imageId]);
  const selDet = imDets.find((d) => d.id === selDetId) ?? imDets[0] ?? null;
  // Real pipeline artifacts for the analysis panel (same endpoint as the Lab).
  // Gated on the image belonging to the CURRENT survey — prevents a stale
  // cross-survey fetch (404) while the new survey's data is still loading.
  const imageResult = useAsync<ImageResult | null>(
    () =>
      surveyId && imageId && surveyMatchesSelection && images.some((i) => i.id === imageId)
        ? api.imageResult(surveyId, imageId)
        : Promise.resolve(null),
    [surveyId, imageId, surveyMatchesSelection, images]
  );
  const [selectedTargetId, setSelectedTargetId] = useState<string | null>(null);
  const [targetQuery, setTargetQuery] = useState("");

  const filteredTargets = useMemo(() => {
    const q = targetQuery.trim().toLowerCase();
    return (targets.data ?? []).filter(
      (t) => !q || t.id.toLowerCase().includes(q) || t.canonical_class.toLowerCase().includes(q)
    );
  }, [targets.data, targetQuery]);

  const classes = Object.entries(stats.data?.by_class ?? {}).sort((a, b) => b[1] - a[1]);
  const maxClass = Math.max(1, ...classes.map(([, v]) => v));
  const tiers = stats.data?.by_priority_tier ?? {};

  const openTargetInLab = (tid: string) => {
    if (!surveyId) return;
    navigate(`/lab/${surveyId}?target=${encodeURIComponent(tid)}`);
  };

  if (stats.error) return <ErrorBox message={stats.error} />;
  if (stats.loading || !stats.data) return <Spinner label="Loading dashboard…" />;

  return (
    <div>
      <SectionTitle
        title="Marine Intelligence Dashboard"
        sub="Scan. Detect. Locate. Protect. — every value below is read live from the survey database."
      />

      {/* ── Summary cards (real backend metrics; tier names are the backend's own) ── */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
        {[
          <Stat key="f" label="Survey frames" value={stats.data.image_count} sub={`${stats.data.survey_count} surveys analyzed`} />,
          <Stat key="d" label="Objects detected" value={stats.data.detection_count} sub="evidence-fused detections" />,
          <Stat
            key="c"
            label="Critical priority"
            value={tiers.critical ?? 0}
            accent={(tiers.critical ?? 0) > 0 ? "text-rose-400" : "text-slate-100"}
            sub="cleanup priority tier"
          />,
          <Stat
            key="h"
            label="High priority"
            value={tiers.high ?? 0}
            accent={(tiers.high ?? 0) > 0 ? "text-orange-400" : "text-slate-100"}
            sub="cleanup priority tier"
          />,
          <Stat
            key="r"
            label="Human review"
            value={stats.data.human_review_count}
            accent={stats.data.human_review_count > 0 ? "text-amber-300" : "text-emerald-400"}
            sub="detections flagged for review"
          />,
        ].map((card, i) => (
          <Stagger key={i} index={i}>
            {card}
          </Stagger>
        ))}
      </div>

      <div className="mt-5 grid gap-4 xl:grid-cols-3">
        {/* ── Sonar Image Analysis (real artifacts, real pipeline outputs) ── */}
        <div className="card xl:col-span-2">
          <div className="flex flex-wrap items-center gap-3 border-b border-slate-700/50 px-4 py-3">
            <h2 className="text-sm font-bold tracking-wide text-slate-100">Sonar Image Analysis</h2>
            {/* Honest state indicator — mirrors the real survey job_status
                (queued/running/done/failed); nothing is faked while idle. */}
            <SonarStatusChip status={survey.data?.job_status ?? null} />
            <select
              className="input !w-56 !py-1 text-xs"
              value={surveyId}
              onChange={(e) => {
                setSurveyId(e.target.value);
                setImageId(null);
                setSelDetId(null);
              }}
              aria-label="Select survey"
            >
              <option value="">Select a survey…</option>
              {(surveys.data ?? []).map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
            <select
              className="input !w-64 !py-1 text-xs"
              value={imageId ?? ""}
              onChange={(e) => {
                setImageId(e.target.value || null);
                setSelDetId(null);
              }}
              aria-label="Select frame"
              disabled={images.length === 0}
            >
              {images.length === 0 && <option value="">No frames</option>}
              {images.map((i) => (
                <option key={i.id} value={i.id}>
                  {i.filename}
                </option>
              ))}
            </select>
            <div className="ml-auto flex flex-wrap gap-1">
              {VIEWS.map((v) => (
                <button
                  key={v.id}
                  onClick={() => setView(v.id)}
                  aria-pressed={view === v.id}
                  className={`rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors ${
                    view === v.id
                      ? "bg-sonar-500/20 text-sonar-300 ring-1 ring-sonar-500/50"
                      : "text-slate-400 hover:bg-abyss-800 hover:text-slate-200"
                  }`}
                >
                  {v.label}
                </button>
              ))}
            </div>
          </div>

          <div className="grid gap-4 p-4 lg:grid-cols-[1fr_260px]">
            <div className="relative flex min-h-[320px] items-center justify-center overflow-hidden rounded-lg border border-sonar-500/20 bg-abyss-950/60 p-2 shadow-[inset_0_0_40px_rgba(6,182,212,0.05)]">
              {!surveyId ? (
                <p className="text-sm text-slate-500">Select a survey to view its sonar analysis.</p>
              ) : survey.loading || detections.loading ? (
                // §12/§23: motion only while data is genuinely loading —
                // the sweep stops the moment content is ready (never fake-idle).
                <div className="flex flex-col items-center gap-3">
                  <div className="aqua-sweep relative flex h-16 w-16 items-center justify-center rounded-full border border-sonar-500/30">
                    <span className="h-1.5 w-1.5 rounded-full bg-sonar-400" />
                  </div>
                  <p className="text-xs text-slate-500">Loading sonar frame…</p>
                </div>
              ) : survey.error ? (
                <ErrorBox message={`Unable to load survey — ${survey.error}`} />
              ) : images.length === 0 ? (
                <p className="text-sm text-slate-500">This survey has no frames.</p>
              ) : (
                <SonarView
                  view={view}
                  image={images.find((i) => i.id === imageId) ?? null}
                  imageResult={imageResult.data}
                  dets={imDets}
                  selDet={selDet}
                />
              )}
            </div>

            <div>
              <h3 className="label mb-2">Detections in this frame</h3>
              {imDets.length === 0 ? (
                <p className="text-xs leading-relaxed text-slate-500">
                  No objects detected in this frame. This does NOT mean the seabed is clear — absence of detections is
                  never reported as a verified clear survey.
                </p>
              ) : (
                <div className="space-y-1">
                  {imDets.map((d) => (
                    <button
                      key={d.id}
                      onClick={() => {
                        setSelDetId(d.id);
                        setView("detections");
                      }}
                      className={`flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs ${
                        selDet?.id === d.id ? "bg-sonar-500/15 ring-1 ring-sonar-500/40" : "hover:bg-abyss-800"
                      }`}
                      aria-label={`Inspect detection ${d.class_name} ${Math.round(d.evidence.fusion * 100)} percent`}
                    >
                      <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: classColor(d.class_name) }} />
                      <span className="min-w-0 flex-1 truncate capitalize text-slate-200">{d.class_name.replace(/_/g, " ")}</span>
                      <span className="mono text-slate-400">{Math.round(d.evidence.fusion * 100)}%</span>
                      <TierBadge tier={d.priority.tier} score={d.priority.score} />
                    </button>
                  ))}
                </div>
              )}
              <Link to={`/lab/${surveyId}`} className="btn-ghost btn mt-3 w-full justify-center !text-xs">
                Open in Lab →
              </Link>
            </div>
          </div>
        </div>

        {/* ── Map + Targets (authoritative spatial API; no fake markers) ── */}
        <div className="space-y-4">
          <div className="card card-pad">
            <div className="mb-2 flex items-center justify-between gap-2">
              <h2 className="text-sm font-bold tracking-wide text-slate-100">Geospatial View</h2>
              {spatial.data && (
                <span className="text-[11px] text-slate-500">
                  Located {spatial.data.targets_with_location} / {spatial.data.targets_total}
                </span>
              )}
            </div>
            {spatial.error ? (
              <ErrorBox message={`Unable to load spatial data — ${spatial.error}`} />
            ) : spatial.data && (spatial.data.targets_with_location > 0 || spatial.data.frames_with_location > 0) ? (
              <TargetMap
                markers={(spatial.data.targets ?? [])
                  .filter((t) => t.latitude != null && t.longitude != null)
                  .map((t) => ({
                    id: t.id,
                    lng: t.longitude as number,
                    lat: t.latitude as number,
                    color: classColor(t.canonical_class),
                    label: `${t.id.slice(0, 12)} · ${t.canonical_class} · ${t.status} · location ${t.geolocation_status}`,
                  }))}
                track={spatial.data.track.available ? (spatial.data.track.coordinates as [number, number][]) : []}
                bounds={spatial.data.bounds}
                selectedId={selectedTargetId}
                onSelect={setSelectedTargetId}
                className="h-[300px] w-full overflow-hidden rounded-lg"
              />
            ) : spatial.loading ? (
              <div className="flex h-[300px] items-center justify-center rounded-lg border border-slate-700/50 bg-abyss-950/60">
                <Spinner label="Loading spatial data…" />
              </div>
            ) : (
              <div className="flex h-[300px] flex-col items-center justify-center gap-1 rounded-lg border border-dashed border-slate-700/60 text-center">
                <p className="text-xs font-medium text-slate-400">Location data unavailable</p>
                <p className="max-w-[240px] text-[11px] leading-relaxed text-slate-500">
                  No valid navigation metadata for this survey — targets are never placed with invented coordinates.
                </p>
              </div>
            )}
            <p className="mt-2 text-[10px] text-slate-600">
              Markers = persistent target representative locations ({spatial.data?.targets?.find((t) => t.geolocation_status === "approximate") ? "approximate" : "as reported by the backend"}).
            </p>
          </div>

          <div className="card card-pad">
            <div className="mb-2 flex items-center gap-2">
              <h2 className="text-sm font-bold tracking-wide text-slate-100">Detected Objects</h2>
              <span className="text-[11px] text-slate-500">{filteredTargets.length} / {targets.data?.length ?? 0}</span>
              <div className="relative ml-auto">
                <Search className="pointer-events-none absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-slate-500" />
                <input
                  className="input !w-40 !py-1 !pl-7 text-[11px]"
                  placeholder="Search id or class…"
                  value={targetQuery}
                  onChange={(e) => setTargetQuery(e.target.value)}
                  aria-label="Search targets"
                />
              </div>
            </div>
            {targets.error ? (
              <ErrorBox message={`Unable to load targets — ${targets.error}`} />
            ) : targets.loading ? (
              <Spinner label="Loading targets…" />
            ) : filteredTargets.length === 0 ? (
              <p className="text-xs text-slate-500">
                {targetQuery ? "No targets match the search." : "No persistent targets yet — run association in the Lab."}
              </p>
            ) : (
              <div className="max-h-[280px] space-y-1 overflow-y-auto scroll-thin pr-1">
                {filteredTargets.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => setSelectedTargetId(selectedTargetId === t.id ? null : t.id)}
                    onDoubleClick={() => openTargetInLab(t.id)}
                    className={`flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-[11px] ${
                      selectedTargetId === t.id ? "bg-sonar-500/15 ring-1 ring-sonar-500/40" : "hover:bg-abyss-800"
                    }`}
                  >
                    <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: classColor(t.canonical_class) }} />
                    <span className="mono shrink-0 text-slate-300">{t.id.slice(0, 10)}</span>
                    <span className="min-w-0 flex-1 truncate capitalize text-slate-400">{t.canonical_class}</span>
                    <span className="shrink-0 text-slate-500">
                      {t.geolocation_status === "unknown" ? "no location" : t.geolocation_status}
                    </span>
                    <span className="w-16 shrink-0 text-right">
                      <Badge value={t.status} />
                    </span>
                  </button>
                ))}
              </div>
            )}
            {selectedTargetId && (
              <button className="btn-ghost btn mt-2 w-full justify-center !text-xs" onClick={() => openTargetInLab(selectedTargetId)}>
                Open selected target in Lab →
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        {/* ── Map + TARGETS + REPORT band: composition report panel ── */}
        <div className="card card-pad xl:col-span-2">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-bold tracking-wide text-slate-100">Survey Reports</h2>
            <Link to="/reports" className="text-[11px] text-sonar-400 hover:underline">
              Full report center →
            </Link>
          </div>
          {!surveyId ? (
            <p className="text-sm text-slate-500">Select a survey to generate or download reports.</p>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {REPORT_FORMATS.map((f) => (
                  <button
                    key={f.id}
                    className="btn-ghost btn justify-center !py-2 !text-xs"
                    onClick={async () => {
                      await api.generateReports(surveyId, [f.id]);
                      reports.reload();
                    }}
                  >
                    <FileDown className="h-3.5 w-3.5" /> {f.label}
                  </button>
                ))}
              </div>
              {reports.data && reports.data.length > 0 && (
                <div className="mt-3 divide-y divide-slate-700/40">
                  {reports.data.slice(-4).reverse().map((r) => (
                    <a
                      key={r.id}
                      href={api.reportDownloadUrl(r.id)}
                      className="flex items-center gap-2 py-1.5 text-[11px] text-slate-300 hover:text-sonar-400"
                    >
                      <span className="mono uppercase text-slate-500">{r.format}</span>
                      <span className="text-slate-600">{fmtDate(r.created_at)}</span>
                      <span className="ml-auto text-sonar-400">download ↓</span>
                    </a>
                  ))}
                </div>
              )}
            </>
          )}
        </div>

        {/* Priority tiers — the backend's real classification (no invented risk model) */}
        <div className="card card-pad">
          <h2 className="label mb-3">Cleanup priority tiers</h2>
          <div className="space-y-2">
            {(["critical", "high", "medium", "low"] as const).map((t) => {
              const n = tiers[t] ?? 0;
              const total = Object.values(tiers).reduce((a, b) => a + b, 0);
              return (
                <div key={t} className="flex items-center gap-3">
                  <span className={`h-2.5 w-2.5 rounded-full ${TIER_COLOR[t]}`} />
                  <span className="w-14 text-xs capitalize text-slate-300">{t}</span>
                  <PctBar value={total ? n / total : 0} color={TIER_COLOR[t]} />
                  <span className="w-6 text-right text-xs font-semibold text-slate-200">{n}</span>
                </div>
              );
            })}
          </div>
          <h2 className="label mb-3 mt-4">Detected classes</h2>
          {classes.length === 0 ? (
            <p className="text-xs text-slate-500">No detections yet.</p>
          ) : (
            <div className="space-y-2">
              {classes.map(([cls, count]) => (
                <div key={cls} className="flex items-center gap-3">
                  <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: classColor(cls) }} />
                  <span className="w-16 shrink-0 text-xs capitalize text-slate-300">{CLASS_LABEL[cls] ?? cls}</span>
                  <PctBar value={count / maxClass} color="bg-sonar-500/80" />
                  <span className="w-6 text-right text-xs font-semibold text-slate-200">{count}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Recent surveys ── */}
      <div className="card card-pad mt-4">
        <h2 className="label mb-3">Recent surveys</h2>
        {stats.data.recent_surveys.length === 0 && (
          <p className="text-sm text-slate-500">
            Nothing analyzed yet.{" "}
            <Link to="/upload" className="text-sonar-400 hover:underline">
              Upload a survey
            </Link>{" "}
            or import a sample.
          </p>
        )}
        <div className="divide-y divide-slate-700/40">
          {stats.data.recent_surveys.map((s) => (
            <div key={s.id} className="flex items-center gap-3 py-2.5">
              <div className="min-w-0 flex-1">
                <Link to={`/lab/${s.id}`} className="truncate text-sm font-medium text-slate-200 hover:text-sonar-400">
                  {s.name}
                </Link>
                <div className="text-xs text-slate-500">{fmtDate(s.created_at)}</div>
              </div>
              <div className="hidden text-xs text-slate-400 sm:block">{s.image_count} frames</div>
              <div className="w-16 text-right text-xs font-semibold text-slate-300">{s.detection_count} det</div>
              <Badge value={s.job_status ?? "created"} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/**
 * LIVE SONAR ANALYSIS indicator — strictly derived from the survey's real
 * job_status. Idle (no job / done long ago) shows a calm "LIVE SONAR
 * ANALYSIS" label with a very slow pulse; an actual running/queued job
 * switches to "ANALYZING SONAR DATA"; done/failed are shown as transient
 * final states. Never fakes a processing state (§4).
 */
function SonarStatusChip({ status }: { status: string | null }) {
  const analyzing = status === "running" || status === "queued";
  const cls = analyzing
    ? "border-sonar-500/50 bg-sonar-500/10 text-sonar-300"
    : status === "failed"
      ? "border-rose-500/40 bg-rose-500/10 text-rose-300"
      : status === "done"
        ? "border-emerald-500/35 bg-emerald-500/8 text-emerald-300"
        : "border-slate-600/50 bg-abyss-900/60 text-slate-400";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-widest ${cls}`}
      title={status ? `Survey job status: ${status}` : "No analysis job yet"}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          analyzing
            ? "bg-sonar-400 animate-status-pulse"
            : status === "failed"
              ? "bg-rose-400"
              : status === "done"
                ? "bg-emerald-400"
                : "bg-slate-500"
        }`}
      />
      {analyzing ? "Analyzing sonar data" : status === "done" ? "Analysis complete" : status === "failed" ? "Analysis failed" : "Live sonar analysis"}
    </span>
  );
}

/** Sonar image viewer for the dashboard — real artifacts and real detections only. */
function SonarView({
  view,
  image,
  imageResult,
  dets,
  selDet,
}: {
  view: ViewMode;
  image: SurveyDetail["images"][number] | null;
  imageResult: ImageResult | null;
  dets: Detection[];
  selDet: Detection | null;
}) {
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);

  if (!image) return <p className="text-sm text-slate-500">Select a frame.</p>;

  if (view === "segmentation") {
    const masks = dets.map((d) => d.mask_path).filter((p): p is string => !!p);
    if (masks.length === 0)
      return <p className="text-sm text-slate-500">No segmentation masks for this frame — masks persist when detections are analyzed.</p>;
    return (
      <div className="relative inline-block leading-none">
        <img
          src={image.raw_url}
          alt="sonar frame with U-Net mask overlay"
          className="max-h-[52vh] w-auto"
          onLoad={(e) => setImgSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
        />
        {masks.map((mp) => (
          <img
            key={mp}
            src={api.mediaUrl(mp)}
            alt="U-Net segmentation mask"
            className="pointer-events-none absolute inset-0 h-full w-full mix-blend-screen"
            style={{ opacity: 0.7 }}
          />
        ))}
      </div>
    );
  }

  if (view === "evidence") {
    if (!selDet) return <p className="text-sm text-slate-500">Select a detection to inspect its evidence.</p>;
    const sig = selDet.evidence.signals;
    const avail = selDet.evidence.availability;
    return (
      <div className="w-full max-w-md text-left text-xs">
        <div className="mb-3 flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: classColor(selDet.class_name) }} />
          <span className="font-semibold capitalize text-slate-100">{selDet.class_name.replace(/_/g, " ")}</span>
          <Badge value={selDet.status} />
          <TierBadge tier={selDet.priority.tier} score={selDet.priority.score} />
        </div>
        <div className="mb-1 flex items-center justify-between text-[11px] text-slate-400">
          <span>Fused evidence score</span>
          <span className="mono text-slate-200">{Math.round(selDet.evidence.fusion * 100)}%</span>
        </div>
        <PctBar value={selDet.evidence.fusion} className="mb-3" />
        <div className="grid grid-cols-2 gap-1.5">
          {(["detection", "segmentation", "natural", "shadow", "physics", "consistency", "anomaly"] as const).map((k) => (
            <div key={k} className="flex items-center justify-between rounded bg-abyss-850/60 px-2 py-1">
              <span className="capitalize text-slate-400">{k}</span>
              <span className={avail[k] ? "mono text-slate-200" : "text-[10px] uppercase text-slate-600"}>
                {avail[k] && sig[k] != null ? `${Math.round((sig[k] as number) * 100)}%` : "unavailable"}
              </span>
            </div>
          ))}
        </div>
        <p className="mt-2 text-[10px] leading-relaxed text-slate-600">
          Unavailable evidence is excluded and renormalized — never scored as zero or fabricated.
        </p>
      </div>
    );
  }

  // original / processed / detections → real stage artifacts (+ real boxes)
  const stageUrl =
    view === "original"
      ? (imageResult?.raw_url ?? image.raw_url)
      : view === "processed"
        ? imageResult?.processed_url ?? null
        : (imageResult?.raw_url ?? image.raw_url);

  if (!stageUrl)
    return (
      <p className="text-sm text-slate-500">
        {view === "processed" ? "Processed image not available for this frame." : "Loading frame…"}
      </p>
    );

  return (
    <div className="relative">
      <img
        src={stageUrl}
        alt={`${view} sonar frame`}
        className="animate-fade-in max-h-[52vh] w-auto"
        onLoad={(e) => setImgSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
      />
      {view === "detections" && imgSize && (
        <svg
          viewBox={`0 0 ${imgSize.w} ${imgSize.h}`}
          className="pointer-events-none absolute inset-0 h-full w-full"
          preserveAspectRatio="xMidYMid meet"
        >
          {dets.map((d, i) => {
            const c = d.status === "human_review_required" ? "#fb7185" : classColor(d.class_name);
            const selected = selDet?.id === d.id;
            return (
              // §14: one entrance animation (opacity + 0.98→1 scale), staggered
              // per box — never a continuous pulse; the data stays inspectable.
              <g
                key={d.id}
                className="animate-detection-in"
                style={{ animationDelay: `${Math.min(i, 6) * 70}ms`, transformOrigin: `${d.box.x + d.box.w / 2}px ${d.box.y + d.box.h / 2}px` }}
              >
                <rect
                  x={d.box.x}
                  y={d.box.y}
                  width={d.box.w}
                  height={d.box.h}
                  fill="none"
                  stroke={c}
                  strokeWidth={selected ? 3.5 : 2}
                  opacity={0.95}
                />
                <text
                  x={d.box.x}
                  y={Math.max(10, d.box.y - 4)}
                  fill={c}
                  fontSize={Math.max(9, imgSize.w / 90)}
                  fontFamily="ui-monospace, monospace"
                >
                  {d.class_name.replace(/_/g, " ")} {Math.round(d.class_confidence * 100)}%
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}
