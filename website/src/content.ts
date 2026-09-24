export type SectionId = "work" | "about" | "experiments" | "lab" | "contact";

export interface Project {
  id: string;
  index: string;
  title: string;
  category: string;
  year: string;
  client: string;
  description: string;
  role: string[];
  hue: number;
  geometry: "monolith" | "torus" | "shard" | "orb" | "lattice" | "helix";
}

export const STUDIO = {
  name: "SENTINEL FORMS",
  tagline: "Independent studio for spatial interfaces & real-time worlds",
  mission:
    "We design and build digital environments — websites, installations and instruments that treat the browser as a place, not a page.",
};

export const SECTIONS: { id: SectionId; label: string; blurb: string }[] = [
  { id: "work", label: "Work", blurb: "Selected commissions & worlds" },
  { id: "about", label: "About", blurb: "The studio, the practice" },
  { id: "experiments", label: "Experiments", blurb: "Fragments from the playground" },
  { id: "lab", label: "Lab", blurb: "Open instruments & notes" },
  { id: "contact", label: "Contact", blurb: "Start a conversation" },
];

export const PROJECTS: Project[] = [
  {
    id: "tidal-memory",
    index: "01",
    title: "Tidal Memory",
    category: "Immersive Storytelling",
    year: "2026",
    client: "Museum of Oceanic Futures",
    description:
      "A WebGL exhibition where visitors drift through soundscapes recorded along disappearing coastlines. Six rooms of generative water, each reacting to live tide data.",
    role: ["Concept", "Art Direction", "WebGL", "Sound"],
    hue: 174,
    geometry: "monolith",
  },
  {
    id: "signal-garden",
    index: "02",
    title: "Signal Garden",
    category: "Interactive Installation",
    year: "2025",
    client: "Kunsthalle Nimbus",
    description:
      "A permanent lobby installation translating passing footsteps into slow-growing fields of light. Built on a custom render pipeline running at 120fps on a single mini-PC.",
    role: ["Design", "Creative Engineering"],
    hue: 262,
    geometry: "torus",
  },
  {
    id: "night-cartography",
    index: "03",
    title: "Night Cartography",
    category: "Data Experience",
    year: "2025",
    client: "Atlas Aerospace Collective",
    description:
      "Ten thousand satellite passes drawn as one navigable night sky. A cartographic instrument for understanding what watches us while we sleep.",
    role: ["UX", "Data Design", "Three.js"],
    hue: 210,
    geometry: "orb",
  },
  {
    id: "paper-engines",
    index: "04",
    title: "Paper Engines",
    category: "Editorial Platform",
    year: "2024",
    client: "Folds & Furnace Publishing",
    description:
      "A reading platform where type behaves like matter — pages bend, fold and carry weight. Commissioned to make slow journalism feel physical again.",
    role: ["Art Direction", "Creative Development"],
    hue: 38,
    geometry: "shard",
  },
  {
    id: "resonant-cities",
    index: "05",
    title: "Resonant Cities",
    category: "Real-time World",
    year: "2024",
    client: "Biennale of Moving Sound",
    description:
      "Three cities rebuilt as acoustic models. Visitors fly above rooftops hearing each street's unique resonance, computed from real architecture.",
    role: ["Concept", "Web Audio", "World Building"],
    hue: 320,
    geometry: "lattice",
  },
  {
    id: "quiet-machines",
    index: "06",
    title: "Quiet Machines",
    category: "Brand System",
    year: "2023",
    client: "Atelier Sunder",
    description:
      "A living identity for a robotics studio — the logotype is grown, not drawn, regenerated from each product's motion-capture signature.",
    role: ["Identity", "Generative Design", "Web"],
    hue: 96,
    geometry: "helix",
  },
];

export const EXPERIMENTS = [
  {
    id: "ink-eclipse",
    index: "E—01",
    title: "Ink Eclipse",
    note: "One million particles solving fluid advection in a single compute pass. Currently misbehaving beautifully.",
  },
  {
    id: "soft-machines",
    index: "E—02",
    title: "Soft Machines",
    note: "Verlet skeletons for creature-like UI. Interfaces that breathe instead of blink.",
  },
  {
    id: "glass-typography",
    index: "E—03",
    title: "Glass Typography",
    note: "Refraction-mapped headlines. Letters thick as cathedral glass, still legible at 14px.",
  },
  {
    id: "weather-chamber",
    index: "E—04",
    title: "Weather Chamber",
    note: "A room whose fog, light and hum are driven by the studio's own barometer.",
  },
];

export const LAB_NOTES = [
  {
    id: "ln-01",
    title: "The camera is a narrator",
    body: "Spatial interfaces fail when the camera plays tour guide — pointing, rushing, showing off. Ours behaves like a documentary narrator: it arrives slightly before you, then waits.",
  },
  {
    id: "ln-02",
    title: "Depth needs a reason",
    body: "Every object placed in space must earn its coordinate. If an element has no spatial relationship to what surrounds it, it belongs in a list, not a world.",
  },
  {
    id: "ln-03",
    title: "Legibility is a spatial property",
    body: "In 3D interfaces, readability is not a font-size decision — it is a camera decision, a contrast decision, a distance decision. We tune all three together.",
  },
];

export const ABOUT = {
  lead: "SENTINEL FORMS is a five-person studio working from Lisbon and Tbilisi. We come from film, engineering and type design — which is why our websites behave like places and our places behave like instruments.",
  paragraphs: [
    "Founded in 2021, the studio has shipped commissions for museums, publishers, aerospace collectives and one very patient robotics company. Our work sits where narrative meets simulation: real-time worlds with editorial discipline.",
    "We believe the web's next decade is spatial — not gimmick-3D, but interfaces that use depth, proximity and light the way print uses rhythm and white space. Every project starts with a question: what would this information want to look like if it had a body?",
    "The studio runs its own instruments: render pipelines, type tooling, sound engines — collected in the Lab and released to the public when they stop embarrassing us.",
  ],
  capabilities: [
    "Creative Direction",
    "Real-time 3D / WebGL",
    "Interactive Type",
    "Sound & Spatial Audio",
    "Installations",
    "Design Systems",
  ],
};

export const CONTACT = {
  email: "hello@sentinelforms.studio",
  phones: ["Lisbon — +351 21 000 4415", "Tbilisi — +995 32 000 8823"],
  offices: [
    { city: "Lisbon", address: "Rua do Ferragial 9, 3º, 1200-182" },
    { city: "Tbilisi", address: "12 Betlemi Street, Old Town, 1025" },
  ],
  socials: [
    { label: "Instagram", handle: "@sentinelforms" },
    { label: "Vimeo", handle: "sentinelforms" },
    { label: "Are.na", handle: "sentinel-forms" },
  ],
};
