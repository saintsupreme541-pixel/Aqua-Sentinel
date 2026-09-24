import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Crosshair, MapPinOff, Navigation as NavIcon, RefreshCw, Crosshair as FitIcon, Flag } from "lucide-react";
import {
  LocationStatus,
  NavigationDiagnostics,
  NavUploadResult,
  PersistentTarget,
  SpatialSummary,
  SurveyDetail,
  TargetHistory,
} from "../api";
import { api } from "../api";
import { TargetMap, MAP_STYLE_URL } from "../components/TargetMap";
import {
  Badge,
  ErrorBox,
  LocationStatusBadge,
  SectionTitle,
  Spinner,
  classColor,
  useAsync,
} from "../components/ui";

const CLASSES = ["wreck", "debris", "structure", "tire"] as const;
const STATUSES = ["active", "review", "confirmed", "rejected"] as const;

type ClassFilter = "all" | (typeof CLASSES)[number];
type StatusFilter = "all" | (typeof STATUSES)[number];
type LocationFilter = "all" | "located" | "unavailable";

const SOURCE_LABEL: Record<string, string> = {
  frame_navigation_plus_sonar_geometry: "Sonar geometry from frame navigation (DERIVED, approximate)",
  frame_navigation: "Frame navigation (GNSS)",
  uploaded_track: "Uploaded navigation track (GNSS)",
  uploaded_track_interpolated: "Uploaded navigation track, time-interpolated",
  user_supplied: "Operator-supplied manual georeference",
  observed: "Observed frame navigation",
};

/** v2 status for a target: backend-normalized field, with client fallback. */
function targetStatus(t: PersistentTarget): LocationStatus {
  if (t.location_status) return t.location_status;
  if (t.geolocation_source === "user_supplied") return "MANUAL";
  if (t.latitude != null && t.longitude != null) return "DERIVED";
  return "UNAVAILABLE";
}

export default function MapPage() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const surveys = useAsync(() => api.surveys(), []);
  const surveyId = params.get("survey") ?? "";
  const spatial = useAsync<SpatialSummary | null>(
    () => (surveyId ? api.spatial(surveyId) : Promise.resolve(null)),
    [surveyId]
  );
  const [cls, setCls] = useState<ClassFilter>("all");
  const [status, setStatus] = useState<StatusFilter>("all");
  const [loc, setLoc] = useState<LocationFilter>("all");
  const [selectedTargetId, setSelectedTargetId] = useState<string | null>(null);

  const survey = useAsync<SurveyDetail | null>(
    () => (surveyId ? api.survey(surveyId) : Promise.resolve(null)),
    [surveyId]
  );
  // v2: the genuine uploaded navigation track + diagnostics (§5/§14)
  const navigation = useAsync(
    () => (surveyId ? api.navigation(surveyId) : Promise.resolve(null)),
    [surveyId]
  );
  const diagnostics = useAsync<NavigationDiagnostics | null>(
    () => (surveyId ? api.geoDiagnostics(surveyId) : Promise.resolve(null)),
    [surveyId]
  );

  // §30: stale-selection guard — clear when the target vanishes from the set.
  useEffect(() => {
    if (selectedTargetId && spatial.data && !spatial.data.targets.some((t) => t.id === selectedTargetId)) {
      setSelectedTargetId(null);
    }
  }, [spatial.data, selectedTargetId]);

  const pickSurvey = (id: string) => {
    setParams(id ? { survey: id } : {});
    setSelectedTargetId(null);
  };

  const allTargets = useMemo(() => spatial.data?.targets ?? [], [spatial.data]);
  // §10: counts come from the backend, never recomputed from marker lists.
  const locatedBackend = spatial.data?.targets_with_location ?? 0;
  const totalBackend = spatial.data?.targets_total ?? 0;

  // §11: client-side filtering of the authoritative dataset (server-side
  // filtering unnecessary at prototype scale; no fake data involved).
  const filtered = useMemo(
    () =>
      allTargets.filter((t) => {
        if (cls !== "all" && t.canonical_class !== cls) return false;
        if (status !== "all" && t.status !== status) return false;
        const isLocated = t.latitude != null && t.longitude != null;
        if (loc === "located" && !isLocated) return false;
        if (loc === "unavailable" && isLocated) return false;
        return true;
      }),
    [allTargets, cls, status, loc]
  );

  const hasAnyGeo =
    (spatial.data?.targets_with_location ?? 0) > 0 ||
    (spatial.data?.frames_with_location ?? 0) > 0 ||
    (navigation.data?.available ?? false);
  const track: [number, number][] = useMemo(
    () =>
      spatial.data?.track.available ? (spatial.data.track.coordinates as [number, number][]) : [],
    [spatial.data]
  );
  // §5: the genuine uploaded navigation track — rendered as its own line.
  const navTrack: [number, number][] = useMemo(
    () => (navigation.data?.available ? (navigation.data.geojson?.geometry?.coordinates as [number, number][]) ?? [] : []),
    [navigation.data]
  );
  // The map canvas renders when the survey has any geography AND the filtered
  // target set has at least one mappable marker (or a drawable track). When the
  // canvas is replaced by an empty state, the empty state itself owns the
  // messaging — the overlay banner must not duplicate it.
  const canvasVisible =
    hasAnyGeo &&
    (filtered.some((t) => t.latitude != null && t.longitude != null) || track.length >= 2);

  return (
    <div>
      <SectionTitle
        title="Survey Intelligence Map"
        sub="Persistent targets and the observed survey track — only genuine backend geolocation is plotted; nothing is fabricated."
        right={
          <select
            className="input !w-80"
            value={surveyId}
            onChange={(e) => pickSurvey(e.target.value)}
            disabled={surveys.loading}
            aria-label="Select survey"
          >
            <option value="">{surveys.loading ? "Loading surveys…" : "Select a survey…"}</option>
            {(surveys.data ?? []).map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
        }
      />

      {surveys.error && <ErrorBox message={surveys.error} />}
      {!surveyId && (
        <div className="card card-pad flex items-center gap-3 text-sm text-slate-400">
          <Crosshair className="h-5 w-5 text-sonar-500" />
          Pick a survey to view its persistent targets and survey track.
        </div>
      )}

      {surveyId && spatial.loading && (
        <div className="card card-pad">
          <Spinner label="Loading survey spatial data…" />
        </div>
      )}
      {surveyId && spatial.error && (
        <ErrorBox
          message={
            spatial.error.includes("404") || spatial.error.toLowerCase().includes("not found")
              ? "Survey not found — it may have been deleted. Pick another survey."
              : `Unable to load spatial survey data — ${spatial.error}`
          }
        />
      )}

      {surveyId && spatial.data && survey.data && (
        <>
          {/* survey summary — all values from the spatial API (§10) */}
          <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-400">
            <span className="font-semibold text-slate-200">{survey.data.name}</span>
            <span>
              Frames: <span className="mono text-slate-300">{spatial.data.frames_total}</span>
            </span>
            <span>
              Targets: <span className="mono text-slate-300">{totalBackend}</span>
            </span>
            <span>
              Located:{" "}
              <span className="mono text-slate-300">
                {locatedBackend} / {totalBackend}
              </span>
            </span>
            <span>
              Track:{" "}
              <span className="mono text-slate-300">
                {spatial.data.track.available ? "Available" : spatial.data.track.point_count > 0 ? "Incomplete" : "Unavailable"}
              </span>
            </span>
            <span>
              GNSS track:{" "}
              <span className="mono text-slate-300">
                {navigation.data ? (navigation.data.available ? `${navigation.data.record_count} fixes` : "None uploaded") : "…"}
              </span>
            </span>
            <button
              className="btn btn-sm"
              onClick={() => spatial.reload()}
              title="Reload spatial data from the backend"
              aria-label="Refresh spatial data"
            >
              <RefreshCw className="h-3.5 w-3.5" /> Refresh
            </button>
            {spatial.data.bounds && (
              <button
                className="btn btn-sm"
                onClick={() => window.dispatchEvent(new CustomEvent("aqua-fit-survey"))}
                title="Fit the map to the survey's track and target extent"
                aria-label="Fit survey extent"
              >
                <FitIcon className="h-3.5 w-3.5" /> Fit survey
              </button>
            )}
            {survey.data.meta.synthetic === true && (
              <span className="rounded bg-violet-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-violet-300">
                SYNTHETIC COORDINATES
              </span>
            )}
          </div>

          {/* filters (§11) */}
          <div className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs">
            <FilterGroup
              label="Class"
              value={cls}
              options={["all", ...CLASSES]}
              onChange={(v) => setCls(v as ClassFilter)}
            />
            <FilterGroup
              label="Status"
              value={status}
              options={["all", ...STATUSES]}
              onChange={(v) => setStatus(v as StatusFilter)}
            />
            <FilterGroup
              label="Location"
              value={loc}
              options={["all", "located", "unavailable"]}
              onChange={(v) => setLoc(v as LocationFilter)}
            />
            <span className="ml-auto text-slate-500">
              Showing {filtered.filter((t) => t.latitude != null).length} marker(s) ·{" "}
              {filtered.length} target(s) match filters
            </span>
          </div>

          <div className="grid gap-4 xl:grid-cols-[1fr_360px]">
            <div className="relative overflow-hidden rounded-xl border border-slate-700">
              {/* §27: explicit empty-filter notice over the track, so a visible
                  survey path never hides it. Only when the canvas is visible —
                  otherwise the empty states below own the messaging. */}
              {filtered.length === 0 && canvasVisible && (
                <div className="absolute inset-x-0 top-0 z-10 flex justify-center p-2">
                  <span className="rounded-lg border border-slate-600 bg-abyss-950/90 px-3 py-1.5 text-xs text-slate-300 shadow">
                    No targets match the current filters.
                  </span>
                </div>
              )}
              {!hasAnyGeo ? (
                <NoGeoState />
              ) : !canvasVisible ? (
                // §27 honesty: distinguish "nothing matches" from "matches exist
                // but none is mappable" — never claim zero matches falsely.
                <EmptyFilterState matchCount={filtered.length} />
              ) : (
                <TargetMap
                  style={MAP_STYLE_URL}
                  markers={filtered.flatMap((t) =>
                    t.latitude != null && t.longitude != null
                      ? [
                          {
                            id: t.id,
                            lng: t.longitude,
                            lat: t.latitude,
                            color: classColor(t.canonical_class),
                            label: `${t.id.slice(0, 12)} · ${t.canonical_class} · ${t.status} · location ${targetStatus(t)}`,
                          },
                        ]
                      : []
                  )}
                  track={track}
                  navTrack={navTrack}
                  bounds={spatial.data.bounds}
                  selectedId={selectedTargetId}
                  onSelect={setSelectedTargetId}
                />
              )}
            </div>

            {/* right column: navigation + legend + detail drawer (§13) */}
            <div className="space-y-4">
              <NavigationPanel
                surveyId={surveyId}
                navigation={navigation}
                diagnostics={diagnostics}
                onUploaded={() => {
                  navigation.reload();
                  diagnostics.reload();
                  spatial.reload();
                }}
              />
              <Legend />
              {selectedTargetId ? (
                <TargetDrawer
                  surveyId={surveyId}
                  target={allTargets.find((t) => t.id === selectedTargetId) ?? null}
                  onOpenLab={(det) =>
                    navigate(`/lab/${surveyId}${det ? `?target=${encodeURIComponent(det)}` : ""}`)
                  }
                  onClose={() => setSelectedTargetId(null)}
                  onManualSet={() => {
                    spatial.reload();
                    diagnostics.reload();
                  }}
                />
              ) : (
                <div className="card card-pad text-[11px] leading-relaxed text-slate-500">
                  Click a target marker to inspect its summary, provenance and observation history — then open it in
                  the Sonar Intelligence Lab for the full evidence. Unlocated targets are never plotted; use the
                  Location filter to review them in the Lab instead.
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function FilterGroup({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: readonly string[];
  onChange: (v: string) => void;
}) {
  return (
    <label className="flex items-center gap-1.5">
      <span className="text-slate-500">{label}</span>
      <select
        className="input !w-auto !py-0.5 text-xs"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        aria-label={`Filter targets by ${label.toLowerCase()}`}
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o === "all" ? "All" : o[0].toUpperCase() + o.slice(1)}
          </option>
        ))}
      </select>
    </label>
  );
}

function Legend() {
  return (
    <div className="card card-pad">
      <h2 className="label mb-2">Legend</h2>
      <div className="space-y-1.5 text-xs">
        {CLASSES.map((c) => (
          <div key={c} className="flex items-center gap-2">
            <span className="h-3 w-3 rounded-full border-2" style={{ borderColor: classColor(c) }} />
            <span className="capitalize text-slate-300">{c}</span>
            <span className="text-slate-600">— target class</span>
          </div>
        ))}
        <div className="flex items-center gap-2 pt-1">
          <svg width="46" height="10">
            <line x1="0" y1="5" x2="46" y2="5" stroke="#22d3ee" strokeDasharray="6 5" strokeWidth="2" />
          </svg>
          <span className="text-slate-500">observed survey track (frame positions; gaps not interpolated)</span>
        </div>
        <div className="flex items-center gap-2">
          <svg width="46" height="10">
            <line x1="0" y1="5" x2="46" y2="5" stroke="#34d399" strokeWidth="2.5" />
          </svg>
          <span className="text-slate-500">GNSS navigation track (uploaded navigation file)</span>
        </div>
        <div className="flex items-center gap-2">
          <svg width="46" height="14">
            <circle cx="23" cy="7" r="6" fill="#0ea5e9" stroke="#e2e8f0" strokeWidth="2" />
          </svg>
          <span className="text-slate-500">selected target</span>
        </div>
      </div>
    </div>
  );
}

/** §22: the honest no-GPS state — no arbitrary map location, no fake markers. */
function NoGeoState() {
  return (
    <div className="flex min-h-[420px] flex-col items-center justify-center gap-3 px-6 py-24 text-center">
      {/* §29: calm radial/sonar visualization — communicates absence of
          metadata; purely visual, implies nothing about confidence. */}
      <div className="relative mb-1 flex h-20 w-20 items-center justify-center">
        <span className="absolute inset-0 rounded-full border border-slate-700/70" />
        <span className="absolute inset-3 rounded-full border border-slate-700/50" />
        <span className="absolute inset-6 rounded-full border border-slate-700/40" />
        <MapPinOff className="h-6 w-6 text-slate-600" />
      </div>
      <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
        No georeferenced observations available for this survey
      </h3>
      <p className="max-w-md text-sm text-slate-500">
        AQUA analyzed the survey successfully, but neither navigation metadata nor an uploaded GNSS track was
        available to place the results geographically. Upload a navigation file (CSV / GeoJSON / GPX) in the panel on
        the right, or review targets in the Sonar Intelligence Lab — the system never invents coordinates.
      </p>
    </div>
  );
}

/**
 * §2/§5 navigation panel: upload a genuine track (CSV/GeoJSON/GPX), inspect
 * ingestion diagnostics, or remove it.  Everything shown is real backend
 * state — rejected rows are listed with reasons, never hidden.
 */
function NavigationPanel({
  surveyId,
  navigation,
  diagnostics,
  onUploaded,
}: {
  surveyId: string;
  navigation: ReturnType<typeof useAsync<Awaited<ReturnType<typeof api.navigation>> | null>>;
  diagnostics: ReturnType<typeof useAsync<NavigationDiagnostics | null>>;
  onUploaded: () => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [crs, setCrs] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<NavUploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const upload = async (file: File) => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.uploadNavigation(surveyId, file, crs || undefined);
      setResult(r);
      onUploaded();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    setBusy(true);
    try {
      await api.deleteNavigation(surveyId);
      setResult(null);
      onUploaded();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const stats = navigation.data?.available ? diagnostics.data?.navigation.stats : null;
  return (
    <div className="card card-pad">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="label">Navigation track (GNSS)</h2>
        <NavIcon className="h-4 w-4 text-slate-500" />
      </div>
      <p className="mb-2 text-[11px] leading-relaxed text-slate-500">
        Upload the vessel's genuine navigation recording. Timestamps synchronize frames to the track; targets are
        geolocated only from real navigation plus sonar geometry — never fabricated.
      </p>
      <input
        ref={fileRef}
        type="file"
        accept=".csv,.txt,.geojson,.json,.gpx"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void upload(f);
          e.target.value = "";
        }}
      />
      <div className="flex flex-wrap items-center gap-2">
        <button className="btn btn-sm" disabled={busy} onClick={() => fileRef.current?.click()}>
          {busy ? "Uploading…" : "Upload CSV / GeoJSON / GPX"}
        </button>
        <input
          className="input !w-36 !py-1 text-xs"
          placeholder="CRS (EPSG:4326)"
          value={crs}
          onChange={(e) => setCrs(e.target.value)}
          aria-label="Source coordinate reference system (optional)"
        />
        {navigation.data && navigation.data.record_count > 0 && (
          <button className="btn btn-sm text-rose-300" disabled={busy} onClick={() => void remove()}>
            Remove track
          </button>
        )}
      </div>
      {error && <ErrorBox message={error} />}
      {result && (
        <div className="mt-2 rounded border border-emerald-500/30 bg-emerald-500/5 p-2 text-[11px]">
          <div className="font-medium text-emerald-300">
            Stored {result.stored} navigation fix(es) · CRS {result.crs}
          </div>
          <div className="text-slate-400">
            Rejected {result.stats.rejected} · duplicate timestamps {result.stats.duplicate_timestamps} · impossible
            jumps {result.stats.impossible_jumps} · gaps &gt;60 s {result.stats.gaps_over_60s}
          </div>
          {result.rejected.length > 0 && (
            <div className="mt-1 text-amber-300">
              Rejected rows (never silently repaired):{" "}
              {result.rejected.slice(0, 3).map((r) => `#${r.index} ${r.issues[0]}`).join("; ")}
              {result.rejected.length > 3 && " …"}
            </div>
          )}
        </div>
      )}
      {stats && (
        <div className="mt-2 grid grid-cols-3 gap-1.5 text-center text-[11px]">
          <div className="rounded bg-abyss-850/60 px-1 py-1">
            <div className="font-semibold text-slate-200">{stats.valid}</div>
            <div className="text-[10px] text-slate-500">valid fixes</div>
          </div>
          <div className="rounded bg-abyss-850/60 px-1 py-1">
            <div className="font-semibold text-slate-200">{stats.rejected}</div>
            <div className="text-[10px] text-slate-500">rejected</div>
          </div>
          <div className="rounded bg-abyss-850/60 px-1 py-1">
            <div className="font-semibold text-slate-200">{diagnostics.data?.navigation.stats.total ?? 0}</div>
            <div className="text-[10px] text-slate-500">total rows</div>
          </div>
        </div>
      )}
      {navigation.data && !navigation.data.available && navigation.data.record_count === 0 && (
        <div className="mt-2 text-[11px] text-slate-500">
          No navigation uploaded — frames keep their operator-supplied metadata only.
        </div>
      )}
    </div>
  );
}

function EmptyFilterState({ matchCount }: { matchCount: number }) {
  return (
    <div className="animate-fade-in flex min-h-[420px] flex-col items-center justify-center gap-3 px-6 py-24 text-center">
      <MapPinOff className="h-10 w-10 text-slate-600" />
      <h3 className="text-sm font-semibold text-slate-400">
        {matchCount === 0 ? "No targets match the current filters." : `${matchCount} target(s) match, but none has a mappable location.`}
      </h3>
      <p className="max-w-sm text-sm text-slate-500">
        {matchCount === 0
          ? "Adjust or reset the class / status / location filters."
          : "Their targets and evidence remain available in the Lab — AQUA never invents coordinates to place them on the map."}
      </p>
    </div>
  );
}

/**
 * §13 target detail drawer.  Compact by design: identity, class, status,
 * location + provenance summary, history, and the Open in Lab workflow (§14).
 * History is fetched only when the drawer opens (§47).
 */
function TargetDrawer({
  surveyId,
  target,
  onOpenLab,
  onClose,
  onManualSet,
}: {
  surveyId: string;
  target: PersistentTarget | null;
  /** Navigates to the Lab with the persistent target preselected (?target=<target id>). */
  onOpenLab: (targetId?: string) => void;
  onClose: () => void;
  /** Called after a successful manual georeference so parents can reload. */
  onManualSet?: () => void;
}) {
  const history = useAsync<TargetHistory | null>(
    () => (target ? api.targetHistory(surveyId, target.id) : Promise.resolve(null)),
    [surveyId, target?.id]
  );

  // §16 manual georeference — explicit, optional, always labelled MANUAL.
  const [manualOpen, setManualOpen] = useState(false);
  const [manualLat, setManualLat] = useState("");
  const [manualLon, setManualLon] = useState("");
  const [manualNote, setManualNote] = useState("");
  const [manualBusy, setManualBusy] = useState(false);
  const [manualError, setManualError] = useState<string | null>(null);

  const submitManual = async () => {
    if (!target) return;
    const la = Number(manualLat);
    const lo = Number(manualLon);
    if (!Number.isFinite(la) || !Number.isFinite(lo) || Math.abs(la) > 90 || Math.abs(lo) > 180) {
      setManualError("Enter a valid latitude (−90…90) and longitude (−180…180).");
      return;
    }
    setManualBusy(true);
    setManualError(null);
    try {
      await api.manualGeorefTarget(surveyId, target.id, la, lo, manualNote);
      setManualOpen(false);
      setManualLat("");
      setManualLon("");
      setManualNote("");
      onManualSet?.();
    } catch (e) {
      setManualError(e instanceof Error ? e.message : String(e));
    } finally {
      setManualBusy(false);
    }
  };

  if (!target) return null;
  const tLat = target.latitude;
  const tLon = target.longitude;
  const located = tLat != null && tLon != null;
  const status = targetStatus(target);
  const lc = history.data?.location_consistency;
  const lcLabel = !lc || !lc.available ? "Insufficient data" : lc.status === "consistent" ? "Consistent" : "Inconsistent";
  const source = target.geolocation_source
    ? (SOURCE_LABEL[target.geolocation_source] ?? target.geolocation_source)
    : "—";

  return (
    <div className="card card-pad animate-drawer-in" aria-label="Selected target details">
      <div className="mb-2 flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: classColor(target.canonical_class) }} />
            <span className="mono text-sm font-bold text-slate-100">{target.id.slice(0, 12)}</span>
          </div>
          <div className="mt-0.5 text-xs capitalize text-slate-400">{target.canonical_class}</div>
        </div>
        <div className="flex flex-col items-end gap-1">
          <Badge value={target.status} />
          <button className="text-[11px] text-slate-500 hover:text-slate-300" onClick={onClose}>
            close
          </button>
        </div>
      </div>

      {/* location block — v2 provenance wording: status is always shown
          with the coordinate (§13); MANUAL is visibly operator-supplied. */}
      <div className="rounded-lg border border-slate-700/50 bg-abyss-900/50 p-2 text-[11px]">
        {located ? (
          <>
            <div className="flex items-center justify-between">
              <span className="label text-[10px]">Location</span>
              <LocationStatusBadge status={status} />
            </div>
            <div className="mono mt-1 text-slate-200">
              {Math.abs(tLat).toFixed(4)}° {tLat >= 0 ? "N" : "S"}, {Math.abs(tLon).toFixed(4)}° {tLon >= 0 ? "E" : "W"}
            </div>
            <div className="mt-0.5 text-slate-500">
              Source: {status === "MANUAL" ? "User supplied (manual)" : source}
            </div>
            {status === "MANUAL" && (
              <div className="text-violet-300">Operator-created georeference — not sensor-derived.</div>
            )}
            {history.data?.representative?.geolocation?.uncertainty_m != null && status !== "MANUAL" ? (
              <div className="text-slate-500">Uncertainty: ±{history.data.representative.geolocation.uncertainty_m} m</div>
            ) : status !== "MANUAL" ? (
              <div className="text-slate-500">Uncertainty: not available</div>
            ) : null}
            <div className={lc && lc.available ? (lc.status === "consistent" ? "text-emerald-400" : "text-amber-400") : "text-slate-500"}>
              Location consistency: {lcLabel}
            </div>
          </>
        ) : (
          <>
            <div className="flex items-center justify-between">
              <span className="label text-[10px]">Location</span>
              <LocationStatusBadge status={status} />
            </div>
            <div className="mt-1 text-slate-500">
              {status === "VESSEL_ONLY"
                ? "Vessel position known; target-level position cannot be justified from sonar geometry."
                : status === "INVALID"
                  ? "Navigation present but failed validation — no coordinate produced."
                  : "Geolocation unavailable — insufficient navigation/sonar metadata. Evidence remains available in the Lab."}
            </div>
          </>
        )}
        {/* §16: explicit optional manual workflow */}
        {!manualOpen ? (
          <button
            className="mt-2 flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-300"
            onClick={() => setManualOpen(true)}
          >
            <Flag className="h-3 w-3" /> Set location manually…
          </button>
        ) : (
          <div className="mt-2 space-y-1.5 rounded border border-violet-500/30 bg-violet-500/5 p-2">
            <div className="text-[10px] font-semibold uppercase tracking-wide text-violet-300">Manual georeference</div>
            <div className="flex gap-1.5">
              <input
                className="input !py-1 text-[11px]"
                placeholder="latitude"
                value={manualLat}
                onChange={(e) => setManualLat(e.target.value)}
                aria-label="Manual latitude"
              />
              <input
                className="input !py-1 text-[11px]"
                placeholder="longitude"
                value={manualLon}
                onChange={(e) => setManualLon(e.target.value)}
                aria-label="Manual longitude"
              />
            </div>
            <input
              className="input !py-1 text-[11px]"
              placeholder="note (optional)"
              value={manualNote}
              onChange={(e) => setManualNote(e.target.value)}
              aria-label="Manual georeference note"
            />
            {manualError && <div className="text-[11px] text-rose-300">{manualError}</div>}
            <div className="flex gap-1.5">
              <button className="btn btn-sm flex-1 justify-center" disabled={manualBusy} onClick={() => void submitManual()}>
                {manualBusy ? "Saving…" : "Save (status MANUAL)"}
              </button>
              <button className="btn btn-sm" onClick={() => setManualOpen(false)}>
                Cancel
              </button>
            </div>
            <p className="text-[10px] text-slate-500">
              Saved coordinates are visibly labelled MANUAL / user-supplied — never presented as GNSS.
            </p>
          </div>
        )}
      </div>

      <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]">
        <KV k="Observations" v={String(target.n_observations ?? target.detection_ids.length)} />
        <KV
          k="First seen"
          v={history.data?.history?.[0] ? `frame ${history.data.history[0].frame_index ?? "?"}` : "—"}
        />
        <KV
          k="Last seen"
          v={
            history.data?.history?.length
              ? `frame ${history.data.history[history.data.history.length - 1].frame_index ?? "?"}`
              : "—"
          }
        />
        <KV
          k="Confidence"
          v={target.confidence != null ? `${Math.round(target.confidence * 100)}%` : "—"}
        />
      </div>
      <p className="mt-1 text-[10px] text-slate-600">
        Confidence = max evidence-fusion of associated detections (prototype aggregate, not a calibrated probability).
      </p>

      {/* §15 compact history — the full version lives in the Lab */}
      <h3 className="label mb-1 mt-3">Observation history</h3>
      {history.loading && <Spinner label="Loading target history…" />}
      {history.error && <ErrorBox message={`Unable to load target history — ${history.error}`} />}
      {history.data && history.data.history.length > 0 && (
        <div className="divide-y divide-slate-700/40">
          {history.data.history.map((h) => (
            <div key={h.detection_id} className="flex items-center gap-2 py-1 text-[11px]">
              <span className="mono w-16 shrink-0 text-slate-500">
                {h.frame_index != null ? `frame ${h.frame_index}` : h.frame_id.slice(0, 10)}
              </span>
              <span className="w-16 shrink-0 capitalize text-slate-300">{h.class_name}</span>
              <span className="mono w-10 shrink-0 text-slate-400">
                {h.fusion != null ? `${Math.round(h.fusion * 100)}%` : "—"}
              </span>
              <span className="truncate text-slate-600">{h.association?.status ?? ""}</span>
            </div>
          ))}
        </div>
      )}

      <button className="btn mt-3 w-full justify-center" onClick={() => onOpenLab(target.id)}>
        Open in Lab →
      </button>
      <p className="mt-1 text-center text-[10px] text-slate-600">
        Opens this survey in the Sonar Intelligence Lab (target preselected).
      </p>
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
