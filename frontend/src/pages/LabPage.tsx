import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import {
  api,
  Detection,
  ImageResult,
  LocationStatus,
  NavigationDiagnostics,
  PersistentTarget,
  SurveyCard,
  SurveyDetail,
  TargetHistory,
} from "../api";
import {
  Badge,
  ErrorBox,
  LocationStatusBadge,
  PctBar,
  SectionTitle,
  Spinner,
  TierBadge,
  classColor,
  useAsync,
  useInterval,
} from "../components/ui";

type View = "overlay" | "segmentation" | "processed" | "raw";

export default function LabPage() {
  const { surveyId } = useParams();
  const navigate = useNavigate();
  // Phase 4: Map → Lab hand-off — ?target=<id> preselects a persistent target.
  const [search, setSearch] = useSearchParams();
  const requestedTarget = search.get("target");
  const surveys = useAsync(() => api.surveys(), []);
  const activeId = surveyId ?? surveys.data?.[0]?.id ?? null;
  const survey = useAsync<SurveyDetail | null>(
    () => (activeId ? api.survey(activeId) : Promise.resolve(null)),
    [activeId]
  );
  const detections = useAsync<Detection[]>(() => (activeId ? api.surveyDetections(activeId) : Promise.resolve([])), [activeId]);

  const [imageId, setImageId] = useState<string | null>(null);
  const [view, setView] = useState<View>("overlay");
  const [selDetId, setSelDetId] = useState<string | null>(null);

  // auto-reload while the job is running
  const jobRunning = survey.data?.job_status === "running" || survey.data?.job_status === "queued";
  const [jobDoneTick, setJobDoneTick] = useState(0);
  useEffect(() => {
    if (!jobRunning) setJobDoneTick((t) => t + 1);
  }, [jobRunning]);
  // Phase 2: persistent targets from the backend association engine
  const targets = useAsync<PersistentTarget[]>(() => (activeId ? api.targets(activeId) : Promise.resolve([])), [activeId, jobDoneTick]);
  useInterval(() => {
    if (jobRunning) {
      surveys.reload();
      survey.reload();
      detections.reload();
    }
  }, 2500);

  const [selectedTargetId, setSelectedTargetId] = useState<string | null>(null);
  // Consume the Map's ?target=<persistent target id> hand-off once targets
  // are loaded (§14).  Stale/unknown ids are ignored safely.
  useEffect(() => {
    if (requestedTarget && (targets.data ?? []).some((t) => t.id === requestedTarget)) {
      setSelectedTargetId(requestedTarget);
      setSearch({}, { replace: true });
    }
  }, [requestedTarget, targets.data, setSearch]);

  const images = useMemo(() => survey.data?.images ?? [], [survey.data?.images]);
  useEffect(() => {
    if (!imageId && images.length > 0) setImageId(images[0].id);
  }, [images, imageId]);

  const image = images.find((i) => i.id === imageId) ?? null;
  const result = useAsync<ImageResult | null>(
    () => (activeId && imageId ? api.imageResult(activeId, imageId) : Promise.resolve(null)),
    [activeId, imageId, jobRunning]
  );
  const imDets = useMemo(
    () => (detections.data ?? []).filter((d) => d.image_id === imageId),
    [detections.data, imageId]
  );
  // Real U-Net masks are persisted per detection by the backend (mask_path →
  // /api/media) — the Segmentation view overlays exactly those files.
  const maskPaths = useMemo(
    () => imDets.map((d) => d.mask_path).filter((p): p is string => !!p),
    [imDets]
  );
  const sel = imDets.find((d) => d.id === selDetId) ?? imDets[0] ?? null;
  useEffect(() => {
    if (selDetId && !imDets.some((d) => d.id === selDetId)) setSelDetId(imDets[0]?.id ?? null);
  }, [imDets, selDetId]);

  const pickSurvey = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setImageId(null);
    setSelDetId(null);
    navigate(`/lab/${e.target.value}`);
  };

  // v2 §14: geolocation diagnostics — navigation availability + position quality
  const geoDiag = useAsync<NavigationDiagnostics | null>(
    () => (activeId ? api.geoDiagnostics(activeId) : Promise.resolve(null)),
    [activeId, jobDoneTick]
  );
  const [showGeoDiag, setShowGeoDiag] = useState(false);

  const current = (surveys.data ?? []).find((s) => s.id === activeId);

  return (
    <div>
      <SectionTitle
        title="Sonar Intelligence Lab"
        sub="Inspect every stage of the pipeline for a single image: raw sonar → preprocessing → detection → segmentation → verification → fusion."
        right={
          <select className="input !w-80" value={activeId ?? ""} onChange={pickSurvey} disabled={surveys.loading}>
            <option value="" disabled>
              {surveys.loading ? "Loading surveys…" : "Select a survey"}
            </option>
            {(surveys.data ?? []).map((s: SurveyCard) => (
              <option key={s.id} value={s.id}>
                {s.name} ({s.detection_count} det)
              </option>
            ))}
          </select>
        }
      />

      {surveys.error && <ErrorBox message={surveys.error} />}
      {!surveys.loading && !surveys.error && surveys.data?.length === 0 && (
        <div className="text-sm text-slate-400">No surveys yet — go to Upload Survey to create one.</div>
      )}

      {current && survey.data && (
        <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-slate-400">
          <span className="font-medium text-slate-300">{survey.data.name}</span>
          <span>{survey.data.image_count} images</span>
          <span>{survey.data.detection_count} detections</span>
          <Badge value={survey.data.job_status ?? "created"} />
          {survey.data.meta.synthetic === true && (
            <span className="rounded bg-violet-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-violet-300">SYNTHETIC</span>
          )}
        </div>
      )}

      {current && survey.data && (
        <div className="mb-3">
          <button
            className="btn btn-sm"
            onClick={() => setShowGeoDiag((v) => !v)}
            aria-expanded={showGeoDiag}
          >
            {showGeoDiag ? "Hide geolocation diagnostics" : "Geolocation diagnostics"}
          </button>
          {showGeoDiag && geoDiag.data && <GeoDiagnosticsPanel diag={geoDiag.data} />}
          {showGeoDiag && geoDiag.loading && <Spinner label="Loading diagnostics…" />}
        </div>
      )}

      <div className="grid gap-4 xl:grid-cols-[210px_1fr]">
        {/* image strip */}
        <div className="card card-pad h-fit">
          <div className="label mb-2">Images</div>
          {images.length === 0 && <p className="text-xs text-slate-500">None.</p>}
          <div className="grid max-h-[70vh] grid-cols-3 gap-1.5 overflow-y-auto scroll-thin xl:grid-cols-1 xl:max-h-[calc(100vh-220px)]">
            {images.map((im) => (
              <button
                key={im.id}
                onClick={() => {
                  setImageId(im.id);
                  setSelDetId(null);
                }}
                className={`flex items-center gap-2 rounded-lg border px-2 py-1.5 text-left transition-colors ${
                  im.id === imageId ? "border-sonar-500 bg-sonar-500/10" : "border-slate-700/60 bg-abyss-850/40 hover:border-slate-500"
                }`}
              >
                <span className="h-8 w-8 shrink-0 overflow-hidden rounded bg-slate-800">
                  <img src={im.raw_url} alt="" className="h-full w-full object-cover" loading="lazy" />
                </span>
                <span className="hidden min-w-0 flex-1 truncate text-[11px] text-slate-300 xl:block">{im.filename}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="min-w-0 space-y-4">
          {result.loading && <p className="text-sm text-slate-500">Loading image result…</p>}
          {result.error && <ErrorBox message={result.error} />}
          {image && result.data && (
            <>
              <div className="card card-pad">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-slate-200">{image.filename}</div>
                    <div className="text-xs text-slate-500">
                      {image.width}×{image.height} · {result.data.quality ? `quality ${result.data.quality.score.toFixed(0)}/100` : "not yet analyzed"}
                    </div>
                  </div>
                  <div className="flex overflow-hidden rounded-lg border border-slate-600 text-xs font-medium">
                    {(
                      [
                        ["overlay", "Detections"],
                        ["segmentation", "Segmentation"],
                        ["processed", "Processed"],
                        ["raw", "Original"],
                      ] as [View, string][]
                    ).map(([v, label]) => (
                      <button
                        key={v}
                        onClick={() => setView(v)}
                        className={`px-3 py-1.5 transition-colors ${view === v ? "bg-sonar-500 text-abyss-950" : "bg-abyss-850 text-slate-300 hover:text-slate-100"}`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="relative w-full overflow-hidden rounded-lg border border-slate-700 bg-black">
                  {view === "overlay" && <StageImg url={result.data.overlay_url} label="detection overlay" dets={imDets} />}
                  {view === "segmentation" &&
                    (maskPaths.length > 0 ? (
                      <MaskView base={result.data.processed_url ?? result.data.raw_url} masks={maskPaths} />
                    ) : (
                      <div className="p-8 text-center text-sm text-slate-500">
                        No segmentation masks for this image{imDets.length > 0 ? " (masks are stored when a detection is analyzed)" : " — no detections were returned for it"}.
                      </div>
                    ))}
                  {view === "processed" && result.data.processed_url && <StageImg url={result.data.processed_url} label="preprocessed" />}
                  {view === "processed" && !result.data.processed_url && <div className="p-8 text-center text-sm text-slate-500">Not processed yet.</div>}
                  {view === "raw" && <StageImg url={result.data.raw_url} label="original raw" />}
                </div>

                <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-slate-500">
                  {result.data.quality?.flags.length ? (
                    <span>
                      flags: {result.data.quality.flags.map((f) => f.replace(/_/g, " ")).join(" · ")}
                    </span>
                  ) : (
                    <span>no quality flags</span>
                  )}
                  <span>
                    preprocessing: {(result.data.preprocess_params.steps as string[] | undefined)?.join(" → ") ?? "—"}
                  </span>
                  <span>backends: {JSON.stringify(result.data.backends)}</span>
                </div>
              </div>

              <div className="grid gap-4 lg:grid-cols-2">
                <DetectionTable dets={imDets} selectedId={sel?.id ?? null} onSelect={setSelDetId} />
                {sel ? <DetectionDetail det={sel} /> : <div className="card card-pad text-sm text-slate-500">Select a detection to inspect its evidence.</div>}
              </div>
            </>
          )}

          {/* Phase 2: persistent targets — everything below comes from the
              backend association engine via /api (no fabricated cards). */}
          {activeId && (
            <TargetsPanel
              surveyId={activeId}
              targets={targets.data ?? []}
              loading={targets.loading}
              selectedTargetId={selectedTargetId}
              onSelect={setSelectedTargetId}
              onAssociated={() => {
                targets.reload();
                detections.reload();
              }}
              frames={images.map((im) => ({ id: im.id, filename: im.filename }))}
            />
          )}
        </div>
      </div>
    </div>
  );
}

/** Backend U-Net masks (PNG, white mask on black) overlaid on the image. */
function MaskView({ base, masks }: { base: string | null; masks: string[] }) {
  const [opacity, setOpacity] = useState(70);
  return (
    <div>
      <div className="relative inline-block leading-none">
        {base ? (
          <img src={base} alt="sonar image with mask overlay" className="block max-h-[62vh] w-auto" />
        ) : (
          <div className="p-8 text-sm text-slate-500">Processed image not available.</div>
        )}
        {base &&
          masks.map((mp) => (
            <img
              key={mp}
              src={api.mediaUrl(mp)}
              alt="U-Net segmentation mask"
              className="pointer-events-none absolute inset-0 h-full w-full mix-blend-screen"
              style={{ opacity: opacity / 100 }}
            />
          ))}
      </div>
      <div className="mt-2 flex items-center justify-end gap-2 text-[11px] text-slate-500">
        <span>mask opacity</span>
        <input
          type="range"
          min={0}
          max={100}
          value={opacity}
          onChange={(e) => setOpacity(Number(e.target.value))}
          className="w-32 accent-sonar-400"
        />
        <span className="w-8 text-right mono">{opacity}%</span>
      </div>
    </div>
  );
}

function StageImg({ url, label, dets }: { url: string; label: string; dets?: Detection[] }) {
  const [failed, setFailed] = useState(false);
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  if (failed) return <div className="p-8 text-center text-sm text-slate-500">No {label} available yet.</div>;
  return (
    <div className="relative">
      <img
        src={url}
        alt={label}
        className="mx-auto max-h-[62vh] w-auto"
        onError={() => setFailed(true)}
        onLoad={(e) => {
          const el = e.currentTarget;
          setImgSize({ w: el.naturalWidth, h: el.naturalHeight });
        }}
      />
      {dets && imgSize && (
        <svg viewBox={`0 0 ${imgSize.w} ${imgSize.h}`} className="pointer-events-none absolute inset-0 mx-auto h-full max-h-[62vh]" preserveAspectRatio="xMidYMid meet">
          {dets.map((d) => (
            <g key={d.id} opacity={0.95} className="animate-detection-in">
              <rect
                x={d.box.x}
                y={d.box.y}
                width={d.box.w}
                height={d.box.h}
                fill="none"
                stroke={d.status === "human_review_required" ? "#fb7185" : classColor(d.class_name)}
                strokeWidth={Math.max(1.5, Math.min(4, imgSize.w / 400))}
              />
            </g>
          ))}
        </svg>
      )}
    </div>
  );
}

function DetectionTable({ dets, selectedId, onSelect }: { dets: Detection[]; selectedId: string | null; onSelect: (id: string) => void }) {
  return (
    <div className="card card-pad">
      <h2 className="label mb-2">Detections in this image</h2>
      {dets.length === 0 && (
        <div className="rounded-lg border border-slate-700/60 bg-abyss-850/40 px-4 py-3 text-xs leading-relaxed text-slate-500">
          <span className="font-medium text-slate-400">No objects detected.</span> This does NOT mean the seabed is
          guaranteed to be clear — the pipeline found no candidate matching the current detection criteria. Absence of
          detections is never reported as a verified clear survey.
        </div>
      )}
      <div className="divide-y divide-slate-700/40">
        {dets.map((d) => (
          <button
            key={d.id}
            onClick={() => onSelect(d.id)}
            className={`flex w-full items-center gap-3 px-1 py-2 text-left ${selectedId === d.id ? "" : ""}`}
          >
            <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: classColor(d.class_name) }} />
            <span className="w-24 shrink-0 truncate text-xs font-medium text-slate-200">{d.class_name.replace(/_/g, " ")}</span>
            <span className="hidden w-10 shrink-0 mono text-[11px] text-slate-400 sm:block">{Math.round(d.evidence.fusion * 100)}%</span>
            <span className="w-20 shrink-0">
              <Badge value={d.status} />
            </span>
            <span className="w-14 shrink-0 text-right">
              <TierBadge tier={d.priority.tier} score={d.priority.score} />
            </span>
            <span className={`ml-auto text-[11px] ${selectedId === d.id ? "text-sonar-400" : "text-slate-600"}`}>
              {selectedId === d.id ? "selected" : "inspect"}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

const SIGNAL_LABEL: Record<string, string> = {
  detection: "Detector score",
  segmentation: "Seg. agreement",
  natural: "Artificial (1−natural)",
  shadow: "Shadow evidence",
  physics: "Sonar geometry",
  consistency: "Multi-frame",
};

function DetectionDetail({ det }: { det: Detection }) {
  const sig = (det.evidence.signals ?? {}) as Record<string, number | null | undefined>;
  const order = ["detection", "segmentation", "natural", "shadow", "physics", "consistency"];
  return (
    <div className="card card-pad">
      <div className="flex items-center justify-between gap-2">
        <div>
          <div className="text-sm font-bold text-slate-100">{det.class_name.replace(/_/g, " ")}</div>
          <div className="mono text-[10px] text-slate-500">{det.id} · {det.image_filename}</div>
        </div>
        <TierBadge tier={det.priority.tier} score={det.priority.score} />
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
        <KV k="Status" v={<Badge value={det.status} />} />
        <KV k="Fusion confidence" v={`${Math.round(det.evidence.fusion * 100)}%`} />
        <KV k="Class conf (raw)" v={`${Math.round(det.class_confidence * 100)}%`} />
        <KV k="Mask area frac" v={det.mask_area_frac != null ? det.mask_area_frac.toFixed(2) : "—"} />
      </div>

      <h3 className="label mb-1 mt-4">Evidence fusion breakdown</h3>
      <div className="space-y-1.5">
        {order.map((k) => {
          const v = sig[k];
          if (v == null) return null;
          const avail = det.evidence.availability?.[k];
          const weight = det.evidence.breakdown?.[k] ?? 0;
          const totalW = Object.values(det.evidence.breakdown ?? {}).reduce((a, b) => a + b, 0);
          return (
            <div key={k} className="flex items-center gap-2">
              <span className="w-28 shrink-0 text-[11px] text-slate-400">{SIGNAL_LABEL[k] ?? k}</span>
              <PctBar value={Number(v)} color={avail === false ? "bg-slate-600" : k === "shadow" ? "bg-amber-400" : "bg-sonar-400"} className="flex-1" />
              <span className="w-9 text-right mono text-[11px] text-slate-300">{Math.round(Number(v) * 100)}</span>
              <span className="w-16 text-right mono text-[10px] text-slate-600">{(totalW ? (weight / totalW) * 100 : 0).toFixed(0)}% w</span>
            </div>
          );
        })}
        {!det.evidence.availability?.shadow && <p className="text-[10px] text-slate-600">shadow evidence not computed (geometry unavailable) — absence is not counted against the object.</p>}
        {!det.evidence.availability?.physics && <p className="text-[10px] text-slate-600">physics evidence not computed (missing altitude/range metadata) — renormalized out of the fusion, never counted as zero.</p>}
      </div>

      <h3 className="label mb-1 mt-4">Physics-informed analysis</h3>
      {det.physics?.metadata_sufficient ? (
        <div className="grid grid-cols-3 gap-2 text-center">
          <MiniStat label="Grazing angle" value={det.physics.grazing_angle_deg != null ? `${det.physics.grazing_angle_deg.toFixed(1)}°` : "—"} />
          <MiniStat label="Height est." value={det.physics.estimated_height_m != null ? `${det.physics.estimated_height_m.toFixed(1)} m` : "—"} />
          <MiniStat label="Geometry score" value={det.physics.geometry_score != null ? `${Math.round(det.physics.geometry_score * 100)}%` : "—"} />
        </div>
      ) : (
        <p className="text-xs text-slate-500">
          Required sonar metadata unavailable — physical height not estimated. Image-based shadow analysis remains
          available above; nothing is fabricated.
        </p>
      )}
      {det.physics?.note && <p className="mt-1.5 text-[10px] leading-relaxed text-slate-600">{det.physics.note}</p>}

      <h3 className="label mb-1 mt-4">Verification</h3>
      <div className="space-y-1 text-xs text-slate-400">
        {det.shadow?.note && <p><span className="text-slate-500">Shadow:</span> {det.shadow.note}</p>}
        {det.dimensions?.note && <p><span className="text-slate-500">Size:</span> {det.dimensions.note}</p>}
        {det.geolocation?.note && <p><span className="text-slate-500">Position:</span> {det.geolocation.note}</p>}
        {det.status === "human_review_required" && (
          <p className="text-rose-300">⚠ Anomalous signature outside known classes — human review required.</p>
        )}
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2 text-center">
        <MiniStat label="Width" value={det.dimensions?.width_m != null ? `${det.dimensions.width_m.toFixed(1)} m` : "—"} />
        <MiniStat label="Height (shadow)" value={det.dimensions?.height_m != null ? `${det.dimensions.height_m.toFixed(1)} m` : "—"} />
        <MiniStat label="Uncertainty" value={det.geolocation?.uncertainty_m != null ? `±${det.geolocation.uncertainty_m.toFixed(0)} m` : "—"} />
      </div>
      {det.raw?.anomaly_score != null && (
        <p className="mt-2 text-[10px] text-slate-600">
          anomaly score (OOD): {(Number(det.raw.anomaly_score)).toFixed(2)} — used only to flag human-review cases, never as positive evidence.
        </p>
      )}
    </div>
  );
}

/**
 * §14 geolocation diagnostics panel — navigation availability, per-frame
 * sync and position quality.  Missing items show as MISSING, never as
 * zeros (§15); every status comes verbatim from the backend.
 */
function GeoDiagnosticsPanel({ diag }: { diag: NavigationDiagnostics }) {
  const n = diag.navigation;
  const rows: Array<[string, string]> = [
    ["GNSS data", n.gnss_data],
    ["Survey track", n.survey_track],
    ["Timestamp", n.timestamp],
    ["CRS", n.crs],
    ["Heading", n.heading],
    ["Towfish data", n.towfish_data],
    ["Layback", n.layback],
    ["Sonar geometry", n.sonar_geometry],
  ];
  return (
    <div className="card card-pad mt-2 animate-fade-in">
      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <h3 className="label mb-2">Navigation</h3>
          <div className="space-y-1 text-xs">
            {rows.map(([k, v]) => (
              <div key={k} className="flex items-center justify-between rounded bg-abyss-850/50 px-2 py-1">
                <span className="text-slate-400">{k}</span>
                {v === "ok" ? (
                  <span className="flex items-center gap-1 text-emerald-300">
                    <span aria-hidden>\u2713</span> available
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-amber-300">
                    <span aria-hidden>\u26a0</span> missing
                  </span>
                )}
              </div>
            ))}
          </div>
          <div className="mt-2 text-[11px] text-slate-500">
            {n.stats.valid} valid fix(es) stored · {n.stats.rejected} rejected at ingestion · {n.record_count} total
          </div>
        </div>
        <div>
          <h3 className="label mb-2">Position quality</h3>
          {diag.position_quality.targets.length === 0 ? (
            <p className="text-xs text-slate-500">No persistent targets yet — run analysis and association first.</p>
          ) : (
            <div className="space-y-1">
              {diag.position_quality.targets.map((t) => (
                <div key={t.target_id} className="flex items-center justify-between rounded bg-abyss-850/50 px-2 py-1 text-xs">
                  <span className="flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full" style={{ background: classColor(t.canonical_class) }} />
                    <span className="mono text-slate-300">{t.target_id.slice(0, 12)}</span>
                  </span>
                  <LocationStatusBadge status={t.location_status} />
                </div>
              ))}
            </div>
          )}
          <div className="mt-2 text-[11px] text-slate-500">
            Located {diag.position_quality.located} / {diag.position_quality.targets_total}
            {Object.keys(diag.position_quality.status_counts).length > 0 &&
              ` · ${Object.entries(diag.position_quality.status_counts)
                .map(([k, v]) => `${k} ${v}`)
                .join(" · ")}`}
          </div>
        </div>
      </div>
      {diag.frames.length > 0 && (
        <details className="mt-3">
          <summary className="cursor-pointer text-xs text-slate-400 hover:text-slate-200">Frame synchronization (\u00a74)</summary>
          <div className="mt-1.5 space-y-1">
            {diag.frames.map((f) => (
              <div key={f.frame_id} className="flex flex-wrap items-center gap-x-3 rounded bg-abyss-850/40 px-2 py-1 text-[11px]">
                <span className="mono w-40 truncate text-slate-400">{f.filename}</span>
                <span className="text-slate-500">
                  {f.frame_position.latitude != null
                    ? `fix ${f.frame_position.latitude.toFixed(4)}, ${f.frame_position.longitude?.toFixed(4)}`
                    : "no position"}
                </span>
                <span className={f.sync_method ? "text-emerald-300" : "text-slate-600"}>
                  {f.sync_method ? `sync: ${f.sync_method}` : "sync: none"}
                </span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

function KV({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2 rounded bg-abyss-850/60 px-2 py-1">
      <span className="text-slate-500">{k}</span>
      <span className="font-medium text-slate-200">{v}</span>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-700/60 bg-abyss-850/50 px-1 py-2">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mono text-xs font-semibold text-slate-200">{value}</div>
    </div>
  );
}

/**
 * Phase 2: persistent targets panel — lists the survey's targets with their
 * observation counts and lets the operator open a target's frame-by-frame
 * history.  All data comes from /api/surveys/{id}/targets(+/history); the
 * "Associate frames" button simply triggers the backend's deterministic
 * association engine (idempotent).
 */
function TargetsPanel({
  surveyId,
  targets,
  loading,
  selectedTargetId,
  onSelect,
  onAssociated,
}: {
  surveyId: string;
  targets: PersistentTarget[];
  loading: boolean;
  selectedTargetId: string | null;
  onSelect: (id: string | null) => void;
  onAssociated: () => void;
  frames: Array<{ id: string; filename: string }>;
}) {
  const [busy, setBusy] = useState(false);
  const [summary, setSummary] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runAssociation = async (reset: boolean) => {
    setBusy(true);
    setError(null);
    try {
      const s = await api.associate(surveyId, reset);
      setSummary(
        `${s.detections_processed} detections → ${s.target_ids.length} targets ` +
          `(${s.targets_created} created now, ${s.already_associated} reused, ${s.ambiguous} ambiguous)`
      );
      onAssociated();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card card-pad">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h2 className="label">Persistent targets — multi-frame association</h2>
        <div className="flex items-center gap-2">
          <button className="btn btn-sm" disabled={busy} onClick={() => runAssociation(false)}>
            {busy ? "Associating…" : "Associate frames"}
          </button>
          <button
            className="btn btn-sm text-slate-400"
            disabled={busy}
            onClick={() => runAssociation(true)}
            title="Clear all targets in this survey and rebuild from the current detections"
          >
            Rebuild
          </button>
        </div>
      </div>
      {summary && <p className="mb-2 text-xs text-sonar-400">{summary}</p>}
      {error && <ErrorBox message={error} />}
      <p className="mb-3 text-[11px] leading-relaxed text-slate-500">
        Deterministic, explainable grouping: detections in different frames that share class, frame proximity,
        and (when available) geolocation are linked as one physical object. Same car in four pictures = one car,
        not four. Ambiguous matches are never merged arbitrarily.
      </p>

      {loading && <p className="text-sm text-slate-500">Loading targets…</p>}
      {!loading && targets.length === 0 && (
        <p className="text-xs text-slate-500">
          No persistent targets yet — run “Associate frames” to group the survey's detections across frames.
        </p>
      )}

      <div className="space-y-2">
        {targets.map((t) => (
          <TargetRow key={t.id} surveyId={surveyId} target={t} open={selectedTargetId === t.id} onToggle={() => onSelect(selectedTargetId === t.id ? null : t.id)} />
        ))}
      </div>
    </div>
  );
}

function TargetRow({
  surveyId,
  target,
  open,
  onToggle,
}: {
  surveyId: string;
  target: PersistentTarget;
  open: boolean;
  onToggle: () => void;
}) {
  const history = useAsync<TargetHistory | null>(
    () => (open ? api.targetHistory(surveyId, target.id) : Promise.resolve(null)),
    [surveyId, target.id, open]
  );
  const framesObserved = new Set((history.data?.history ?? []).map((h) => h.frame_id)).size;
  const firstIdx = history.data?.history?.[0]?.frame_index;
  const lastIdx = history.data?.history?.[history.data.history.length - 1]?.frame_index;

  return (
    <div className="rounded-lg border border-slate-700/60 bg-abyss-850/40">
      <button onClick={onToggle} className="flex w-full items-center gap-3 px-3 py-2 text-left">
        <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: classColor(target.canonical_class) }} />
        <span className="mono w-28 shrink-0 text-xs font-semibold text-slate-100">{target.id.slice(0, 12)}</span>
        <span className="w-20 shrink-0 truncate text-xs capitalize text-slate-300">{target.canonical_class}</span>
        <Badge value={target.status} />
        <span className="hidden text-[11px] text-slate-400 sm:block">
          {history.data ? `${framesObserved} frame${framesObserved === 1 ? "" : "s"}` : ""}
          {firstIdx != null && lastIdx != null && firstIdx !== lastIdx ? ` (frame ${firstIdx}–${lastIdx})` : firstIdx != null ? ` (frame ${firstIdx})` : ""}
        </span>
        <span className="ml-auto text-right text-[11px] text-slate-400">
          {target.confidence != null ? `conf ${Math.round(target.confidence * 100)}%` : ""}
          {" · "}
          {target.geolocation_status === "unknown" ? "location unavailable" : `location ${target.geolocation_status}`}
        </span>
        <span className={`ml-2 text-[11px] ${open ? "text-sonar-400" : "text-slate-600"}`}>{open ? "hide" : "history"}</span>
      </button>
      {open && (
        <div className="border-t border-slate-700/50 px-3 py-2">
          {history.loading && <p className="text-xs text-slate-500">Loading history…</p>}
          {history.error && <ErrorBox message={history.error} />}
          {history.data && (
            <>
              <div className="mb-2 grid grid-cols-2 gap-2 text-[11px] md:grid-cols-4">
                <KV k="Observations" v={String(history.data.n_observations)} />
                <KV
                  k="First seen"
                  v={history.data.history[0] ? `frame ${history.data.history[0].frame_index ?? "?"}` : "—"}
                />
                <KV
                  k="Last seen"
                  v={
                    history.data.history[history.data.history.length - 1]
                      ? `frame ${history.data.history[history.data.history.length - 1].frame_index ?? "?"}`
                      : "—"
                  }
                />
                <KV k="Representative" v={history.data.representative ? history.data.representative.detection_id.slice(0, 16) : "—"} />
              </div>

              {/* Phase 3: traceable location block — status, source, provenance. */}
              <div className="mb-2 rounded-lg border border-slate-700/50 bg-abyss-900/50 p-2">
                <div className="mb-1 flex items-center justify-between gap-2">
                  <span className="label text-[10px]">Location</span>
                  <LocationStatusBadge
                    status={
                      (history.data.geolocation_evidence?.status as LocationStatus | undefined) ??
                      (history.data.geolocation_status === "approximate"
                        ? "DERIVED"
                        : history.data.geolocation_status === "known"
                          ? "VERIFIED"
                          : "UNAVAILABLE")
                    }
                  />
                </div>
                {history.data.latitude != null && history.data.longitude != null ? (
                  <div className="space-y-0.5 text-[11px]">
                    <div className="mono text-slate-200">
                      {history.data.latitude.toFixed(4)}, {history.data.longitude.toFixed(4)}
                      <span className="ml-2 text-[10px] text-slate-500">(±{history.data.representative?.geolocation?.uncertainty_m ?? "?"} m)</span>
                    </div>
                    <div className="text-slate-500">
                      Source: {history.data.geolocation_source === "frame_navigation_plus_sonar_geometry"
                        ? "frame navigation + sonar geometry"
                        : history.data.geolocation_source ?? "—"}
                    </div>
                    {history.data.location_consistency.available && (
                      <div className={history.data.location_consistency.status === "consistent" ? "text-emerald-400" : "text-amber-400"}>
                        Location consistency: {history.data.location_consistency.status}
                        {history.data.location_consistency.dispersion_m != null &&
                          ` · spread ${history.data.location_consistency.dispersion_m} m over ${history.data.location_consistency.n} fixes`}
                      </div>
                    )}
                    <div className="text-[10px] text-slate-600">{history.data.location_note}</div>
                  </div>
                ) : (
                  <div className="text-[11px] text-slate-500">
                    <div>Location: unavailable</div>
                    <div className="text-slate-600">Reason: {history.data.location_note.replace("location unavailable — ", "")}</div>
                  </div>
                )}
              </div>

              <p className="mb-2 text-[10px] text-slate-600">{history.data.confidence_definition}</p>
              <div className="divide-y divide-slate-700/40">
                {history.data.history.map((h) => (
                  <div key={h.detection_id} className="flex items-center gap-3 py-1.5 text-[11px]">
                    <span className="mono w-20 shrink-0 text-slate-500">{h.frame_index != null ? `frame ${h.frame_index}` : h.frame_id.slice(0, 10)}</span>
                    <span className="w-16 shrink-0 capitalize text-slate-300">{h.class_name}</span>
                    <span className="mono w-10 shrink-0 text-slate-400">{h.fusion != null ? `${Math.round(h.fusion * 100)}%` : "—"}</span>
                    <span className="w-20 shrink-0">
                      <Badge value={h.status} />
                    </span>
                    <span className="mono w-40 shrink-0 truncate text-slate-500">
                      {h.geolocation?.known && h.geolocation.lat != null
                        ? `${h.geolocation.lat.toFixed(4)}, ${h.geolocation.lon?.toFixed(4)}`
                        : "location unavailable"}
                    </span>
                    <span className="truncate text-slate-500">
                      {h.association?.reasons?.slice(0, 2).join(" · ") || h.association?.status || ""}
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
