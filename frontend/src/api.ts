// Thin typed client for the AQUA-SENTINEL FastAPI backend.
// Types mirror backend/app/schemas.py — keep field names in sync.

export interface SurveyMeta {
  name?: string;
  description?: string;
  sonar_type?: string;
  lat?: number | null;
  lon?: number | null;
  heading_deg?: number | null;
  altitude_m?: number | null;
  range_m?: number | null;
  side?: string;
  preprocess_preset?: string;
  [k: string]: unknown;
}

export interface SurveyCard {
  id: string;
  name: string;
  description: string;
  created_at: string;
  image_count: number;
  detection_count: number;
  job_status: string | null;
  job_id: string | null;
  meta: SurveyMeta;
}

export interface SurveyImage {
  id: string;
  filename: string;
  width: number;
  height: number;
  status: string;
  meta: SurveyMeta;
  raw_url: string;
}

export interface SurveyDetail extends SurveyCard {
  images: SurveyImage[];
}

// Phase 1 persistence model: Survey → Frame → FrameDetection → PersistentTarget.
// Navigation fields are null when the operator did not supply metadata —
// the backend never fabricates values.
export interface Frame {
  id: string;
  survey_id: string;
  frame_index: number | null;
  filename: string;
  width: number;
  height: number;
  status: string;
  created_at: string | null;
  captured_at: string | null;
  latitude: number | null;
  longitude: number | null;
  heading_deg: number | null;
  altitude_m: number | null;
  sonar_side: string | null;
  slant_range_m: number | null;
}

export interface FrameDetail extends Frame {
  detections: Detection[];
}

// Phase 3: structured geolocation provenance (observed vs derived).
export interface GeoProvenance {
  kind: "derived" | "unavailable" | "aggregate" | "manual" | "invalid" | "vessel_only";
  status?: string;
  source?: string | null;
  reference_frame_id?: string | null;
  inputs?: Record<string, boolean>;
  derived?: Record<string, number | undefined>;
  assumptions?: string[];
  issues?: string[];
  selection_rule?: string;
  from_detection_id?: string;
  from_frame_id?: string;
  observations_located?: number;
  n_observations?: number;
  note?: string;
  // v2: navigation synchronization audit (§4/§7/§11)
  navigation?: {
    sync_method?: "exact_timestamp" | "linear_interpolation" | "nearest" | null;
    nav_source?: string | null;
    interpolation_fraction?: number | null;
    gap_s?: number | null;
    towfish_config?: Record<string, number> | null;
    towfish_applied?: boolean;
  } | null;
}

// Geolocation v2 state model (§1): every coordinate carries exactly one.
export type LocationStatus = "VERIFIED" | "DERIVED" | "MANUAL" | "VESSEL_ONLY" | "UNAVAILABLE" | "INVALID";

// v2: survey navigation track (§5) — genuine uploaded navigation only.
export interface NavTrackInfo {
  available: boolean;
  record_count: number;
  coordinates: [number, number][] | null;
  note: string;
}

export interface NavigationDiagnostics {
  survey_id: string;
  navigation: {
    gnss_data: string;
    survey_track: string;
    timestamp: string;
    crs: string;
    heading: string;
    towfish_data: string;
    layback: string;
    sonar_geometry: string;
    stats: { total: number; valid: number; rejected: number };
    record_count: number;
  };
  position_quality: {
    targets: Array<{
      target_id: string;
      canonical_class: string;
      location_status: LocationStatus;
      latitude: number | null;
      longitude: number | null;
      uncertainty_m: number | null;
      position_source: string | null;
    }>;
    targets_total: number;
    located: number;
    status_counts: Record<string, number>;
  };
  frames: Array<{
    frame_id: string;
    filename: string;
    captured_at: string | null;
    frame_position: { latitude: number | null; longitude: number | null; position_source: string | null };
    sync_method: string | null;
    in_track_span: boolean;
  }>;
}

export interface NavUploadResult {
  survey_id: string;
  stored: number;
  stats: { total: number; valid: number; rejected: number; duplicate_timestamps: number; impossible_jumps: number; gaps_over_60s: number };
  rejected: Array<{ index: number; issues: string[] }>;
  warnings: string[];
  crs: string;
  span: { start: string; end: string } | null;
}

export interface LocationConsistency {
  available: boolean;
  status: "consistent" | "inconsistent" | "insufficient_data";
  n?: number;
  dispersion_m: number | null;
  score: number | null;
  threshold_m?: number | null;
}

export interface PersistentTarget {
  id: string;
  survey_id: string;
  canonical_class: string;
  status: "active" | "confirmed" | "rejected" | "review";
  confidence: number | null;
  first_seen_frame_id: string | null;
  last_seen_frame_id: string | null;
  representative_detection_id: string | null;
  latitude: number | null;
  longitude: number | null;
  geolocation_status: "known" | "approximate" | "unknown";
  // v2 normalized state (VERIFIED/DERIVED/MANUAL/...); backend always sends it
  location_status?: LocationStatus;
  geolocation_source?: string | null;
  geolocation_evidence?: (GeoProvenance & { location_consistency?: LocationConsistency }) | null;
  n_observations?: number;
  notes: string;
  created_at: string;
  updated_at: string;
  detection_ids: string[];
}

// Phase 2: association evidence attached to each detection (raw.association)
export interface AssociationEvidence {
  associated: boolean;
  status: "accepted" | "new_target" | "reused" | "ambiguous" | "below_threshold" | "no_candidates";
  target_id?: string | null;
  score?: number | null;
  signals: Record<string, { available: boolean; score?: number; [k: string]: unknown }>;
  unavailable: string[];
  reasons: string[];
  candidates?: Array<{ target_id: string; score: number }>;
}

export interface TargetHistoryEntry {
  detection_id: string;
  frame_id: string;
  frame_index: number | null;
  frame_filename: string | null;
  class_name: string;
  class_confidence: number | null;
  fusion: number | null;
  status: string;
  box: { x: number; y: number; w: number; h: number };
  mask_path: string | null;
  geolocation: Detection["geolocation"];
  association: AssociationEvidence | null;
}

export interface TargetHistory {
  id: string;
  survey_id: string;
  canonical_class: string;
  status: PersistentTarget["status"];
  confidence: number | null;
  confidence_definition: string;
  first_seen_frame_id: string | null;
  last_seen_frame_id: string | null;
  representative_detection_id: string | null;
  representative: TargetHistoryEntry | null;
  latitude: number | null;
  longitude: number | null;
  geolocation_status: PersistentTarget["geolocation_status"];
  geolocation_source: string | null;
  geolocation_evidence: (GeoProvenance & { location_consistency?: LocationConsistency }) | null;
  location_consistency: LocationConsistency;
  location_note: string;
  notes: string;
  created_at: string;
  updated_at: string;
  n_observations: number;
  history: TargetHistoryEntry[];
}

// Phase 3: authoritative spatial summary (survey track + target locations).
export interface SpatialSummary {
  survey_id: string;
  frames_total: number;
  frames_with_location: number;
  frames_without_location: number;
  targets_total: number;
  targets_with_location: number;
  targets_without_location: number;
  bounds: [[number, number], [number, number]] | null;
  nav_track?: NavTrackInfo;
  track: {
    available: boolean;
    point_count: number;
    note: string;
    points: Array<{
      frame_id: string;
      frame_index: number | null;
      filename: string;
      latitude: number;
      longitude: number;
      heading_deg: number | null;
      captured_at: string | null;
      position_source: "observed";
    }>;
    coordinates: [number, number][];
  };
  frame_positions: SpatialSummary["track"]["points"];
  targets: PersistentTarget[];
}

export interface AssociationSummary {
  survey_id: string;
  frames_processed: number;
  detections_processed: number;
  targets_created: number;
  detections_associated: number;
  already_associated: number;
  unassociated: number;
  ambiguous: number;
  target_ids: string[];
  config: Record<string, number>;
}

export interface Job {
  id: string;
  survey_id: string;
  status: string;
  stage: string;
  progress: number;
  message: string;
  error?: string | null;
  timings: Record<string, number>;
}

export interface QualityReport {
  score: number;
  flags: string[];
  metrics: Record<string, number>;
  metadata_completeness: Record<string, boolean>;
}

export interface EvidenceSignals {
  detection?: number | null;
  segmentation?: number | null;
  natural?: number | null;
  shadow?: number | null;
  physics?: number | null;
  consistency?: number | null;
  anomaly?: number | null;
}

export interface PhysicsInfo {
  metadata_sufficient: boolean;
  grazing_angle_deg?: number | null;
  estimated_height_m?: number | null;
  geometry_score?: number | null;
  note?: string;
}

export interface Detection {
  id: string;
  image_id: string;
  survey_id: string;
  target_id?: string | null; // set when linked to a persistent target
  image_filename: string;
  class_name: string;
  class_confidence: number;
  box: { x: number; y: number; w: number; h: number };
  mask_path?: string | null;
  mask_area_frac?: number | null;
  evidence: {
    signals: EvidenceSignals;
    availability: Record<string, boolean>;
    fusion: number;
    breakdown: Record<string, number>;
    backend_used: Record<string, string>;
  };
  status: "confirmed" | "review" | "candidate" | "human_review_required";
  shadow: {
    available: boolean;
    valid?: boolean | null;
    length_px?: number | null;
    length_m?: number | null;
    height_estimate_m?: number | null;
    note?: string;
  };
  physics: PhysicsInfo;
  geolocation: {
    known: boolean;
    status?: string;
    // v2 normalized state — present on new results; older persisted rows
    // fall back to status normalization on the backend.
    location_status?: LocationStatus;
    lat?: number | null;
    lon?: number | null;
    frame_position?: { latitude: number; longitude: number; position_source: string } | null;
    uncertainty_m?: number | null;
    ellipse?: { semi_major_m: number; semi_minor_m: number; rotation_deg: number } | null;
    provenance?: GeoProvenance | null;
    note?: string;
  };
  dimensions: {
    estimable: boolean;
    width_m?: number | null;
    length_m?: number | null;
    height_m?: number | null;
    note?: string;
  };
  priority: {
    score: number;
    tier: "critical" | "high" | "medium" | "low";
    factors: Record<string, number>;
  };
  raw: Record<string, unknown>;
}

export interface ImageResult {
  image_id: string;
  survey_id: string;
  filename: string;
  width: number;
  height: number;
  meta: SurveyMeta;
  raw_url: string;
  processed_url: string | null;
  overlay_url: string;
  quality: QualityReport | null;
  preprocess_params: Record<string, unknown>;
  backends: Record<string, string>;
  status: string;
}

export interface DashboardData {
  survey_count: number;
  image_count: number;
  detection_count: number;
  by_class: Record<string, number>;
  by_status: Record<string, number>;
  by_priority_tier: Record<string, number>;
  human_review_count: number;
  recent_surveys: Array<{
    id: string;
    name: string;
    created_at: string;
    image_count: number;
    detection_count: number;
    job_status: string | null;
  }>;
}

export interface SampleInfo {
  id: string;
  name: string;
  description: string;
  synthetic: boolean;
  attribution: string;
  license_note: string;
  image_count: number;
  size_kb: number;
  has_ground_truth: boolean;
  tags: string[];
}

export interface ReportInfo {
  id: string;
  survey_id: string;
  format: string;
  path: string;
  size_bytes: number;
  created_at: string;
}

const BASE = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () =>
    request<{ status: string; mode?: string; model_backends?: Record<string, string> }>("/api/health"),

  dashboard: () => request<DashboardData>("/api/dashboard"),
  surveys: () => request<SurveyCard[]>("/api/surveys"),
  survey: (id: string) => request<SurveyDetail>(`/api/surveys/${id}`),
  surveyDetections: (id: string) => request<Detection[]>(`/api/surveys/${id}/detections`),
  surveyGeoJSON: (id: string) => request<unknown>(`/api/surveys/${id}/geojson`),
  imageResult: (sid: string, iid: string) => request<ImageResult>(`/api/surveys/${sid}/images/${iid}`),
  detection: (id: string) => request<Detection>(`/api/detections/${id}`),

  // Phase 1 persistence APIs
  frames: (sid: string) => request<Frame[]>(`/api/surveys/${sid}/frames`),
  frame: (sid: string, fid: string) => request<FrameDetail>(`/api/surveys/${sid}/frames/${fid}`),
  frameDetections: (sid: string, fid: string) =>
    request<Detection[]>(`/api/surveys/${sid}/frames/${fid}/detections`),
  targets: (sid: string) => request<PersistentTarget[]>(`/api/surveys/${sid}/targets`),
  target: (sid: string, tid: string) => request<PersistentTarget>(`/api/surveys/${sid}/targets/${tid}`),
  targetHistory: (sid: string, tid: string) =>
    request<TargetHistory>(`/api/surveys/${sid}/targets/${tid}/history`),
  // Phase 3: authoritative spatial data (track + target locations + counts)
  spatial: (sid: string) => request<SpatialSummary>(`/api/surveys/${sid}/spatial`),

  // Geolocation v2: navigation track + diagnostics + manual georeference
  navigation: (sid: string) =>
    request<NavTrackInfo & { survey_id: string; geojson: { geometry: { coordinates: [number, number][] } | null }; records: unknown[] }>(
      `/api/surveys/${sid}/navigation`
    ),
  uploadNavigation: (sid: string, file: File, crs?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (crs) form.append("crs", crs);
    return request<NavUploadResult>(`/api/surveys/${sid}/navigation`, { method: "POST", body: form });
  },
  deleteNavigation: (sid: string) => request<{ removed: number }>(`/api/surveys/${sid}/navigation`, { method: "DELETE" }),
  geoDiagnostics: (sid: string) => request<NavigationDiagnostics>(`/api/surveys/${sid}/geolocation/diagnostics`),
  manualGeorefTarget: (sid: string, tid: string, latitude: number, longitude: number, note = "") =>
    request<{ target_id: string; geolocation: Detection["geolocation"] }>(
      `/api/surveys/${sid}/targets/${tid}/geolocation/manual`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ latitude, longitude, note }),
      },
    ),
  associate: (sid: string, reset = false) =>
    request<AssociationSummary>(`/api/surveys/${sid}/associate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reset }),
    }),

  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  jobStreamUrl: (id: string) => `${BASE}/api/jobs/${id}/stream`,

  samples: () => request<SampleInfo[]>("/api/samples"),
  importSample: (id: string) => request<{ survey_id: string; job_id: string }>(`/api/samples/${id}/import`, { method: "POST" }),

  createSurvey: (files: File[], meta: Record<string, unknown>, imageMeta: Record<string, unknown>) => {
    const form = new FormData();
    for (const f of files) form.append("files", f);
    form.append("meta", JSON.stringify(meta));
    form.append("image_meta", JSON.stringify(imageMeta));
    return request<{ survey_id: string; job_id: string; image_count: number }>("/api/surveys", {
      method: "POST",
      body: form,
    });
  },

  generateReports: (sid: string, formats: string[]) =>
    request<{ survey_id: string; report_ids: string[] }>(`/api/surveys/${sid}/reports`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formats),
    }),
  reports: (sid: string) => request<ReportInfo[]>(`/api/surveys/${sid}/reports`),
  reportDownloadUrl: (rid: string) => `${BASE}/api/reports/${rid}/download`,

  mediaUrl: (rel: string) => `${BASE}/api/media/${rel.replace(/^\/+/, "")}`,
};
