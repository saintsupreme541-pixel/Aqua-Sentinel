import { AnimatePresence, motion } from "framer-motion";
import { StrictMode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ABOUT,
  CONTACT,
  PROJECTS,
  SECTIONS,
  STUDIO,
  type Project,
  type SectionId,
} from "./content";
import {
  AboutPanel,
  ContactPanel,
  ExperimentsPanel,
  LabPanel,
  ProjectDetail,
  WorkList,
} from "./overlays";
import { SceneEngine, type SceneNodeDef } from "./scene";
import "./styles.css";

/* section nodes -> spatial metaphors */
const SECTION_SHAPES: Record<SectionId, SceneNodeDef["shape"]> = {
  work: "cluster",
  about: "slab",
  experiments: "swarm",
  lab: "knot",
  contact: "beacon",
};
const SECTION_HUES: Record<SectionId, number> = {
  work: 200,
  about: 155,
  experiments: 280,
  lab: 45,
  contact: 335,
};

type Phase = "intro" | "home" | "section" | "project";

interface HoverInfo {
  id: string;
  kind: "section" | "project";
  title: string;
  sub: string;
  cta: string;
}

const nodeDefs: SceneNodeDef[] = [
  ...SECTIONS.map<SceneNodeDef>((s) => ({
    id: s.id,
    label: s.label,
    sub: s.blurb,
    hue: SECTION_HUES[s.id],
    shape: SECTION_SHAPES[s.id],
    group: "section",
  })),
  ...PROJECTS.map<SceneNodeDef>((p) => ({
    id: p.id,
    label: p.title,
    sub: `${p.category} — ${p.year}`,
    hue: p.hue,
    shape: p.geometry,
    group: "project",
  })),
];

function useIsMobile() {
  const [mobile, setMobile] = useState(
    () => window.matchMedia("(pointer: coarse)").matches || window.innerWidth < 720
  );
  useEffect(() => {
    const mq = window.matchMedia("(pointer: coarse)");
    const on = () => setMobile(mq.matches || window.innerWidth < 720);
    mq.addEventListener("change", on);
    window.addEventListener("resize", on);
    return () => {
      mq.removeEventListener("change", on);
      window.removeEventListener("resize", on);
    };
  }, []);
  return mobile;
}

function Clock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = window.setInterval(() => setNow(new Date()), 30_000);
    return () => window.clearInterval(t);
  }, []);
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  return (
    <span>
      {hh}
      {mm} LOCAL
    </span>
  );
}

export default function App() {
  const stageRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<SceneEngine | null>(null);
  const cursorRef = useRef<HTMLDivElement>(null);
  const dotRef = useRef<HTMLDivElement>(null);

  const isMobile = useIsMobile();
  const [phase, setPhase] = useState<Phase>("intro");
  const [activeSection, setActiveSection] = useState<SectionId | null>(null);
  const [openProject, setOpenProject] = useState<Project | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [hover, setHover] = useState<HoverInfo | null>(null);
  const hoverRef = useRef<HoverInfo | null>(null);
  const [pointer, setPointer] = useState({ x: -200, y: -200 });

  const reducedMotion = useMemo(
    () => window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    []
  );

  /* ---------- engine lifecycle ---------- */
  useEffect(() => {
    if (!stageRef.current) return;
    const engine = new SceneEngine({
      container: stageRef.current,
      nodes: nodeDefs,
      isMobile,
      onSelect: (id) => {
        const section = SECTIONS.find((s) => s.id === id);
        if (section) {
          goSection(section.id);
          return;
        }
        const project = PROJECTS.find((p) => p.id === id);
        if (project) openProjectFromList(project);
      },
      onHover: (id) => {
        if (!id) {
          hoverRef.current = null;
          setHover(null);
          return;
        }
        const section = SECTIONS.find((s) => s.id === id);
        if (section) {
          const info: HoverInfo = {
            id,
            kind: "section",
            title: section.label,
            sub: section.blurb,
            cta: "Enter",
          };
          hoverRef.current = info;
          setHover(info);
          return;
        }
        const project = PROJECTS.find((p) => p.id === id);
        if (project) {
          const info: HoverInfo = {
            id,
            kind: "project",
            title: project.title,
            sub: `${project.category} — ${project.year}`,
            cta: "Open case",
          };
          hoverRef.current = info;
          setHover(info);
        }
      },
    });
    engineRef.current = engine;
    if (import.meta.env.DEV) {
      (window as unknown as { __engine?: SceneEngine }).__engine = engine;
    }
    const introDelay = reducedMotion ? 0 : 1_500;
    const t = window.setTimeout(() => {
      engine.startIntro();
      setPhase("home");
    }, introDelay);
    return () => {
      window.clearTimeout(t);
      engine.dispose();
      engineRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isMobile]);

  /* ---------- custom cursor ---------- */
  useEffect(() => {
    if (isMobile) return;
    const move = (e: PointerEvent) => {
      // only re-render for the hover card while something is hovered;
      // the cursor itself is moved via direct DOM writes above
      if (hoverRef.current) setPointer({ x: e.clientX, y: e.clientY });
      if (cursorRef.current) cursorRef.current.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
      if (dotRef.current) dotRef.current.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
    };
    window.addEventListener("pointermove", move);
    return () => window.removeEventListener("pointermove", move);
  }, [isMobile]);

  /* ---------- ESC navigation ---------- */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (menuOpen) {
        setMenuOpen(false);
      } else if (phase === "project") {
        closeProject();
      } else if (phase === "section") {
        goHome();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [menuOpen, phase]);

  /* ---------- navigation ---------- */
  const goSection = useCallback((id: SectionId) => {
    setMenuOpen(false);
    setOpenProject(null);
    setActiveSection(id);
    setPhase("section");
    engineRef.current?.setSection(id);
    engineRef.current?.pulse();
  }, []);

  const goHome = useCallback(() => {
    setMenuOpen(false);
    setOpenProject(null);
    setActiveSection(null);
    setPhase("home");
    engineRef.current?.setSection(null);
  }, []);

  const openProjectFromList = useCallback((p: Project) => {
    console.log("[debug] openProjectFromList", p.id);
    setOpenProject(p);
    setPhase("project");
    engineRef.current?.focusProject(p.id);
  }, []);

  const closeProject = useCallback(() => {
    setOpenProject(null);
    setPhase(activeSection ? "section" : "home");
    engineRef.current?.focusOverview();
  }, [activeSection]);

  const sectionPanel = () => {
    switch (activeSection) {
      case "work":
        return <WorkList onOpen={openProjectFromList} onBack={goHome} />;
      case "about":
        return <AboutPanel onBack={goHome} />;
      case "experiments":
        return <ExperimentsPanel onBack={goHome} />;
      case "lab":
        return <LabPanel onBack={goHome} />;
      case "contact":
        return <ContactPanel onBack={goHome} />;
      default:
        return null;
    }
  };

  return (
    <>
      {/* WebGL stage */}
      <div ref={stageRef} className="stage" aria-hidden />

      {/* top chrome */}
      <header className="chrome-top">
        <button type="button" className="wordmark" onClick={goHome}>
          {STUDIO.name}
          <small>{STUDIO.tagline}</small>
        </button>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <p className="status-line">
            <span className="live-dot" />
            SYSTEMS NOMINAL
            <br />
            <Clock />
          </p>
          <button
            type="button"
            className={`menu-btn glass${menuOpen ? " open" : ""}`}
            onClick={() => setMenuOpen((v) => !v)}
            aria-expanded={menuOpen}
          >
            <span className="bars" aria-hidden>
              <span />
              <span />
            </span>
            {menuOpen ? "Close" : "Index"}
          </button>
        </div>
      </header>

      {/* footer strip */}
      <footer className="footer-strip">
        <span>Lisbon — Tbilisi</span>
        <span className="home-hint-inline">
          {phase === "home" ? "Select a structure to enter" : ""}
        </span>
        <span>{isMobile ? "Tap to explore" : "Hover to sense — click to enter"}</span>
      </footer>

      {/* home statement */}
      <AnimatePresence>
        {phase === "home" && (
          <motion.div
            className="home-statement"
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -16 }}
            transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
            style={{ pointerEvents: "none" }}
          >
            <p className="eyebrow">Independent studio — est. 2021</p>
            <h1 className="h-serif display-xl">
              We build places
              <br />
              <em style={{ fontWeight: 300 }}>inside the browser</em>
            </h1>
            <p className="body-copy" style={{ marginTop: "1.4rem" }}>
              {STUDIO.mission}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* hover card */}
      <AnimatePresence>
        {hover && phase !== "project" && (
          <motion.div
            key={hover.id}
            className="hover-card glass"
            initial={{ opacity: 0, y: 10, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.98 }}
            transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
            style={{
              left: Math.min(pointer.x + 22, window.innerWidth - 270),
              top: Math.min(pointer.y + 22, window.innerHeight - 160),
            }}
          >
            <p className="hc-label">
              {hover.kind === "section" ? "Section" : "Project"} — {hover.cta}
            </p>
            <p className="hc-title">{hover.title}</p>
            <p className="hc-sub">{hover.sub}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* section / project overlays */}
      <AnimatePresence mode="wait">
        {phase === "section" && (
          <motion.div
            key={`sec-${activeSection}`}
            className="overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.45 }}
          >
            {sectionPanel()}
          </motion.div>
        )}
        {phase === "project" && openProject && (
          <motion.div
            key={`proj-${openProject.id}`}
            className="overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.45 }}
          >
            <ProjectDetail project={openProject} onBack={closeProject} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* menu drawer */}
      <AnimatePresence>
        {menuOpen && (
          <>
            <motion.div
              className="menu-scrim"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setMenuOpen(false)}
            />
            <motion.nav
              className="menu-drawer glass"
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
            >
              {SECTIONS.map((s, i) => (
                <button
                  key={s.id}
                  type="button"
                  className={`menu-item${activeSection === s.id ? " active" : ""}`}
                  onClick={() => goSection(s.id)}
                >
                  <span className="num">0{i + 1}</span>
                  <span className="label">{s.label}</span>
                  <span className="blurb">{s.blurb}</span>
                </button>
              ))}
              <div className="menu-foot">
                {CONTACT.email}
                <br />
                {ABOUT.capabilities.length} disciplines — one studio
              </div>
            </motion.nav>
          </>
        )}
      </AnimatePresence>

      {/* intro veil */}
      <AnimatePresence>
        {phase === "intro" && (
          <motion.div
            className="intro-veil"
            exit={{ opacity: 0 }}
            transition={{ duration: 1.1, ease: "easeInOut" }}
          >
            <motion.p
              className="eyebrow"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              {STUDIO.name}
            </motion.p>
            <div className="intro-bar">
              <motion.i
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ duration: reducedMotion ? 0.1 : 1.3, ease: "easeInOut" }}
              />
            </div>
            <motion.p
              className="intro-sub"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.5 }}
            >
              Calibrating environment
            </motion.p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* custom cursor */}
      {!isMobile && (
        <>
          <div ref={cursorRef} className={`cursor${hover ? " hover" : ""}`} />
          <div ref={dotRef} className="cursor-dot" />
        </>
      )}
    </>
  );
}

/* ---------- bootstrap ---------- */

const rootEl = document.getElementById("root");
if (!rootEl) throw new Error("#root element not found");

// Re-running this module (Vite HMR) must unmount the previous root first,
// otherwise createRoot() throws on the already-claimed container.
const hot = import.meta.hot;
hot?.dispose(() => root?.unmount());

const root = createRoot(rootEl);
root.render(
  <StrictMode>
    <App />
  </StrictMode>
);
