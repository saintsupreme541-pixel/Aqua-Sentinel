# Frontend Visual System

> Design philosophy: **"Subtle motion + underwater atmosphere + professional
> data visualization."** The user should feel that *something is happening* —
> never that *everything is moving*. Motion communicates system activity,
> selection, transition and hierarchy; the data stays the star.

## Color system

The existing AQUA palette is reused unchanged (`tailwind.config.js`):

- `abyss-950 … 700` — deep navy surfaces
- `sonar-400/500` — cyan accents, active states, glows
- cool white (`slate-100/200`) for data, muted gray (`slate-500/600`) for meta
- status: emerald (confirmed/online), amber (review), rose (critical/error), orange (high)
- class colors come from the shared `classColor()` tokens (wreck/debris/structure/tire)

No new colors were introduced; status colors stay consistent across Dashboard,
Map, Lab, tables and reports.

## Animation principles

1. **Transform/opacity only.** Every keyframe animates GPU-friendly
   properties — never `width`/`height`/`top`/`left`.
2. **One home for keyframes.** All keyframes live in `frontend/src/index.css`
   (`aqua-fade-in`, `aqua-slide-up`, `aqua-rise`, `aqua-drawer-in`,
   `aqua-detection-in`, `aqua-drift`, `aqua-ray`, `aqua-halo`, `aqua-status`,
   `aqua-shimmer`, `aqua-sweep`, `aqua-shine-move`). No per-file duplicates.
3. **No animation libraries.** Zero dependencies added (no Framer Motion,
   GSAP, Three.js, particles.js). Pure CSS + a few CSSProperties.
4. **Motion must mean something.** Sweep/skeleton appear only while data is
   genuinely loading; the halo marks only the *selected* target; the status
   dot pulses only for real backend health. Nothing implies confidence or
   fake processing.

## Reusable classes / components

| Utility / component        | Purpose                                                        |
| -------------------------- | -------------------------------------------------------------- |
| `AmbientBackground`        | fixed underwater layers (glow + 2 light rays + 5 slow particles) |
| `aqua-vignette`            | edge vignette focusing attention on content                    |
| `aqua-shine`               | slow specular shine across the AQUA-SENTINEL wordmark          |
| `.animate-fade-in`         | page-enter fade (route change, ~350 ms)                        |
| `.animate-slide-up`        | 8 px rise for cards/panels                                     |
| `.animate-rise`            | 12 px rise used by `Stagger` for summary cards (60 ms/card)    |
| `.animate-drawer-in`       | target drawer `translateX(15px)→0` (~300 ms)                   |
| `.animate-detection-in`    | one-shot bounding-box entrance (opacity + 0.98→1 scale)        |
| `.animate-status-pulse`    | slow 3.2 s opacity pulse for health indicators                 |
| `.aqua-selected` / `.animate-halo` | concentric pulse halo on the selected map marker only |
| `.aqua-skeleton`           | shimmer skeleton for loading panels                            |
| `.aqua-sweep`              | rotating radial wedge — **only** while data genuinely loads    |
| `.card-hover`              | hover lift (-2 px) + border glow + soft illumination           |
| `Stagger`, `Skeleton`      | small helpers in `components/ui.tsx`                           |

## Background effects

`AmbientBackground` renders (behind all content, `pointer-events: none`):

1. deep navy base (body) + two radial cyan/teal glows
2. two large, blurred, near-invisible light rays slowly rocking (±2.5°)
3. five 2–3 px particles rising over 30–50 s with slight horizontal drift
4. a fixed vignette darkening only the edges

The Dashboard uses the dense variant; the same component is available with
`dense={false}` for Map/Lab.

## Reduced motion

`@media (prefers-reduced-motion: reduce)` disables **every** animation
(particles, rays, shine, sweep, halo, pulses, skeleton shimmer, card hover
lift, entrance transitions) while keeping the static visual styling.
Particles remain as faint static specks. The application stays fully usable.

## Performance

- No JavaScript timers for animation — all layers are CSS-driven.
- Animations run on compositor-friendly properties only.
- The sweep/scan effect is a single `conic-gradient` pseudo-element rotating
  via `transform`, used only during genuine loads.
- No WebGL, canvas loops, or external animation runtime.

## Accessibility

- Map markers remain keyboard-focusable buttons with descriptive
  `aria-label`s; the halo is decorative (`aria-hidden` background layers).
- Focus rings are preserved on all controls; press/hover effects are ≤2 px
  scale/translate and never remove focus visibility.
- Status is never color-only (badges include text).
- Reduced-motion support as above.

## Known limitations

- The ambient background is decorative only; on very low-end GPUs the blur of
  the two rays is the most expensive layer (visually negligible, still
  compositor-only).
- The preview viewport is ~730 px wide in this environment; the header mission
  line hides below `md:` by design (responsive, not a defect).
