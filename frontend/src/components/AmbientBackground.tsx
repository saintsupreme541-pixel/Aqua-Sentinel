import bgUrl from "../assets/underwater-bg.webp";
import seascapeUrl from "../assets/underwater-seascape.webp";

/**
 * Background environment (§33): two realistic underwater photographs (the
 * original deep-ocean scene and the new cinematic seascape with light
 * rays / seabed / shipwreck silhouette) as two stacked, absolutely-
 * positioned layers that slowly crossfade A→B→A on a 24s loop (8s hold ·
 * 4s fade · 8s hold · 4s fade). Sonar imagery is NEVER used here — the
 * page background is decorative underwater photography only; actual sonar
 * scans live exclusively in the analysis features (Lab/Explorer/Dashboard
 * sonar panel). Both layers are preloaded by the bundler and painted from
 * the first frame — no flashes or gaps. A navy readability overlay and a
 * static vignette sit above the images, below all UI. Nothing moves except
 * the crossfade opacity; the layer is pointer-events-none, z-0, and
 * reduced-motion shows the static original photo only.
 */
export function AmbientBackground() {
  return (
    <div className="aqua-ambient" aria-hidden>
      {/* Layer A — original underwater photograph (cover, center; static). */}
      <div className="aqua-photo" style={{ backgroundImage: `url(${bgUrl})` }} />
      {/* Layer B — cinematic underwater seascape; fades over A and back
          (opacity only). Decorative photography, never sonar imagery. */}
      <div className="aqua-photo-b" style={{ backgroundImage: `url(${seascapeUrl})` }} />
      {/* Readability overlay + static edge vignette — below UI, above images. */}
      <div className="aqua-overlay" />
      <div className="aqua-glow" />
    </div>
  );
}

