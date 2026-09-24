import { useEffect, useRef } from "react";
import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

/** Shared default style (env-overridable) — one source for Map + Dashboard.
 *
 * The previous default (demotiles.maplibre.org) is a minimal placeholder with
 * no real-world basemap detail — targets plotted on it appeared to float on a
 * blank canvas.  CARTO's dark-matter basemap is free (no API key), global and
 * matches the AQUA dark-marine theme.  Operators can override via
 * VITE_MAP_STYLE_URL (e.g. self-hosted or bathymetry styles).
 */
export const MAP_STYLE_URL =
  (import.meta.env.VITE_MAP_STYLE_URL as string | undefined) ??
  "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

export interface TargetMarker {
  id: string;
  lng: number;
  lat: number;
  color: string;
  label: string;
}

/**
 * Stable MapLibre canvas: one map instance for the component lifetime (§24);
 * markers/track/fit update in place — never a full re-init on filter or
 * selection changes.  Markers are keyboard-focusable, labeled buttons (§33).
 * Data contract: markers/track/bounds come verbatim from the backend spatial
 * API — this component performs no geolocation math and never fabricates
 * points.
 */
export function TargetMap({
  style = MAP_STYLE_URL,
  markers,
  track,
  navTrack = [],
  bounds,
  selectedId,
  onSelect,
  className = "h-[62vh] min-h-[420px] w-full",
}: {
  style?: string;
  markers: TargetMarker[];
  track: [number, number][];
  /** §5: the genuine uploaded GNSS navigation track (green line). */
  navTrack?: [number, number][];
  bounds: [[number, number], [number, number]] | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
  /** Size class for embedding contexts (dashboard panel vs full page). */
  className?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const markersRef = useRef<Map<string, maplibregl.Marker>>(new Map());
  const onSelectRef = useRef(onSelect);
  const firstFitRef = useRef(false);
  onSelectRef.current = onSelect;

  // init once
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style,
      center: [0, 0],
      zoom: 2,
      attributionControl: { compact: true },
    });
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl({ showCompass: true }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
    // "Fit survey" control is handled from React (real spatial bounds);
    // the native control block above is intentionally minimal.
    map.on("load", () => {
      if (!map.getSource("aqua-track")) {
        map.addSource("aqua-track", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
        map.addLayer({
          id: "aqua-track-line",
          type: "line",
          source: "aqua-track",
          paint: { "line-color": "#22d3ee", "line-width": 2, "line-dasharray": [2, 1.5], "line-opacity": 0.55 },
        });
      }
      // v2: genuine uploaded GNSS track (§5) — solid green, above frame track.
      if (!map.getSource("aqua-nav-track")) {
        map.addSource("aqua-nav-track", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
        map.addLayer({
          id: "aqua-nav-track-line",
          type: "line",
          source: "aqua-nav-track",
          paint: { "line-color": "#34d399", "line-width": 2.5, "line-opacity": 0.8 },
        });
      }
    });
    const store = markersRef.current;
    return () => {
      map.remove();
      mapRef.current = null;
      store.forEach((m) => m.remove());
      store.clear();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // React-side "Fit survey" handler (real bounds from the spatial API),
  // reachable from page header buttons via the aqua-fit-survey event (§21).
  useEffect(() => {
    const fit = () => {
      if (bounds) mapRef.current?.fitBounds(bounds, { padding: 60, maxZoom: 17, duration: 500 });
    };
    window.addEventListener("aqua-fit-survey", fit);
    return () => window.removeEventListener("aqua-fit-survey", fit);
  }, [bounds]);

  // fit on first data arrival (§21) — never jumps to a default city
  useEffect(() => {
    if (!bounds || firstFitRef.current) return;
    firstFitRef.current = true;
    mapRef.current?.fitBounds(bounds, { padding: 60, maxZoom: 17, duration: 600 });
  }, [bounds]);

  // sync markers in place
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const store = markersRef.current;
    const alive = new Set<string>();
    for (const m of markers) {
      alive.add(m.id);
      const existing = store.get(m.id);
      if (existing) {
        existing.setLngLat([m.lng, m.lat]);
        const span = (existing.getElement().firstChild as HTMLElement | null)?.querySelector("span");
        if (span) span.style.borderColor = m.color;
        continue;
      }
      const el = document.createElement("button");
      el.type = "button";
      el.setAttribute("aria-label", `Open target ${m.label} details`);
      el.style.cssText = "background:none;border:none;padding:0;cursor:pointer;";
      el.title = m.label;
      el.addEventListener("click", () => onSelectRef.current(m.id));
      // §26 hover: gentle lift (transition only — no continuous motion).
      el.style.transition = "transform 150ms ease";
      el.addEventListener("mouseenter", () => (el.style.transform = "scale(1.25)"));
      el.addEventListener("mouseleave", () => (el.style.transform = ""));
      el.innerHTML = `<span style="display:block;width:14px;height:14px;border-radius:9999px;border:3px solid ${m.color};background:rgba(2,6,23,0.55);box-shadow:0 0 8px ${m.color}66;"></span>`;
      const marker = new maplibregl.Marker({ element: el, anchor: "center" }).setLngLat([m.lng, m.lat]).addTo(map);
      store.set(m.id, marker);
    }
    for (const [id, marker] of store) {
      if (!alive.has(id)) {
        marker.remove();
        store.delete(id);
      }
    }
  }, [markers]);

  // track data (observed frame positions only — backend guarantees the order)
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const apply = () => {
      const src = map.getSource("aqua-track") as maplibregl.GeoJSONSource | undefined;
      src?.setData({
        type: "FeatureCollection",
        features:
          track.length > 1
            ? [{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: track } }]
            : [],
      });
      // v2: uploaded GNSS navigation track (§5)
      const navSrc = map.getSource("aqua-nav-track") as maplibregl.GeoJSONSource | undefined;
      navSrc?.setData({
        type: "FeatureCollection",
        features:
          navTrack.length > 1
            ? [{ type: "Feature", properties: { kind: "gnss_track" }, geometry: { type: "LineString", coordinates: navTrack } }]
            : [],
      });
    };
    if (map.isStyleLoaded()) apply();
    else map.once("load", apply);
  }, [track, navTrack]);

  // selected marker emphasis (§18): concentric pulse halo on the SELECTED
  // target only — selection feedback, never a confidence signal.
  useEffect(() => {
    const store = markersRef.current;
    for (const [id, marker] of store) {
      const span = marker.getElement().querySelector("span") as HTMLElement | null;
      if (span) {
        const isSelected = id === selectedId;
        span.style.outline = isSelected ? "2px solid #e2e8f0" : "";
        span.style.transform = isSelected ? "scale(1.4)" : "";
        span.classList.toggle("aqua-selected", isSelected);
      }
    }
  }, [selectedId, markers]);

  return <div ref={containerRef} className={className} />;
}
