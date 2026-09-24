import { motion } from "framer-motion";
import { ABOUT, CONTACT, EXPERIMENTS, LAB_NOTES, PROJECTS, type Project } from "./content";

const rise = {
  hidden: { opacity: 0, y: 26 },
  show: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: 0.06 * i, duration: 0.7, ease: [0.16, 1, 0.3, 1] as const },
  }),
};

function BackBar({ label, onBack }: { label: string; onBack: () => void }) {
  return (
    <motion.button
      type="button"
      className="back-btn glass"
      onClick={onBack}
      initial={{ opacity: 0, y: -12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.4 }}
    >
      <span aria-hidden>←</span> {label}
    </motion.button>
  );
}

/* ---------------- WORK index ---------------- */

export function WorkList({ onOpen, onBack }: { onOpen: (p: Project) => void; onBack: () => void }) {
  return (
    <div className="panel-pad" style={{ display: "flex", flexDirection: "column", padding: "0 clamp(1.2rem,5vw,4.5rem)" }}>
      <div style={{ position: "sticky", top: 0, paddingTop: "1.2rem", zIndex: 5, display: "flex", justifyContent: "flex-end" }}>
        <BackBar label="Environment" onBack={onBack} />
      </div>
      <motion.header
        variants={rise}
        initial="hidden"
        animate="show"
        custom={0}
        style={{ paddingTop: "4vh", paddingBottom: "3vh" }}
      >
        <p className="eyebrow">Selected Work</p>
        <h1 className="h-serif display-l" style={{ marginTop: "0.6rem" }}>
          Commissions &<br />
          <em style={{ fontWeight: 300 }}>other worlds</em>
        </h1>
      </motion.header>
      <div>
        {PROJECTS.map((p, i) => (
          <motion.button
            key={p.id}
            type="button"
            className="row-item"
            variants={rise}
            initial="hidden"
            animate="show"
            custom={i + 1}
            onClick={() => onOpen(p)}
          >
            <span className="num">{p.index}</span>
            <span className="title">{p.title}</span>
            <span className="meta">
              {p.category}
              <br />
              {p.year}
            </span>
          </motion.button>
        ))}
      </div>
      <p className="eyebrow" style={{ margin: "2.5rem 0 1rem", color: "var(--ink-faint)" }}>
        Select a structure in the environment — or a row above
      </p>
    </div>
  );
}

/* ---------------- PROJECT detail ---------------- */

export function ProjectDetail({ project, onBack }: { project: Project; onBack: () => void }) {
  return (
    <div className="panel-pad" style={{ display: "flex", flexDirection: "column", padding: "0 clamp(1.2rem,5vw,4.5rem)" }}>
      <div style={{ position: "sticky", top: 0, paddingTop: "1.2rem", zIndex: 5, display: "flex", justifyContent: "flex-end" }}>
        <BackBar label="All work" onBack={onBack} />
      </div>
      <motion.article
        key={project.id}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.5 }}
        style={{ maxWidth: "56rem", paddingBottom: "6vh" }}
      >
        <div className="project-hero">
          <motion.p className="eyebrow" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15, duration: 0.6 }}>
            {project.index} — {project.category}
          </motion.p>
          <motion.h1
            className="h-serif hero-title"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          >
            {project.title}
          </motion.h1>
        </div>
        <motion.dl className="spec-grid" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4, duration: 0.7 }}>
          <div>
            <dt>Client</dt>
            <dd>{project.client}</dd>
          </div>
          <div>
            <dt>Year</dt>
            <dd>{project.year}</dd>
          </div>
          <div>
            <dt>Status</dt>
            <dd>Shipped</dd>
          </div>
        </motion.dl>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5, duration: 0.7 }}>
          <p className="body-copy" style={{ fontSize: "1.15rem", color: "var(--ink)", maxWidth: "40rem", marginTop: "2rem" }}>
            {project.description}
          </p>
          <div className="role-tags">
            {project.role.map((r) => (
              <span key={r}>{r}</span>
            ))}
          </div>
          <p className="eyebrow" style={{ marginTop: "3rem", color: "var(--ink-faint)" }}>
            Full case study on request — {CONTACT.email}
          </p>
        </motion.div>
      </motion.article>
    </div>
  );
}

/* ---------------- ABOUT ---------------- */

export function AboutPanel({ onBack }: { onBack: () => void }) {
  return (
    <div className="panel-pad" style={{ display: "flex", flexDirection: "column", padding: "0 clamp(1.2rem,5vw,4.5rem)" }}>
      <div style={{ position: "sticky", top: 0, paddingTop: "1.2rem", zIndex: 5, display: "flex", justifyContent: "flex-end" }}>
        <BackBar label="Environment" onBack={onBack} />
      </div>
      <motion.div initial="hidden" animate="show" style={{ maxWidth: "44rem", paddingTop: "4vh", paddingBottom: "6vh" }}>
        <motion.p className="eyebrow" variants={rise} custom={0}>
          About the studio
        </motion.p>
        <motion.h1 className="h-serif display-l" variants={rise} custom={1} style={{ margin: "0.8rem 0 2rem" }}>
          A small studio<br />with <em style={{ fontWeight: 300 }}>deep reach</em>
        </motion.h1>
        <motion.p className="body-copy" variants={rise} custom={2} style={{ fontSize: "1.2rem", color: "var(--ink)" }}>
          {ABOUT.lead}
        </motion.p>
        {ABOUT.paragraphs.map((p, i) => (
          <motion.p key={i} className="body-copy" variants={rise} custom={3 + i} style={{ marginTop: "1.4rem" }}>
            {p}
          </motion.p>
        ))}
        <motion.div variants={rise} custom={6} className="role-tags">
          {ABOUT.capabilities.map((c) => (
            <span key={c}>{c}</span>
          ))}
        </motion.div>
      </motion.div>
    </div>
  );
}

/* ---------------- EXPERIMENTS ---------------- */

export function ExperimentsPanel({ onBack }: { onBack: () => void }) {
  return (
    <div className="panel-pad" style={{ display: "flex", flexDirection: "column", padding: "0 clamp(1.2rem,5vw,4.5rem)" }}>
      <div style={{ position: "sticky", top: 0, paddingTop: "1.2rem", zIndex: 5, display: "flex", justifyContent: "flex-end" }}>
        <BackBar label="Environment" onBack={onBack} />
      </div>
      <motion.header variants={rise} initial="hidden" animate="show" custom={0} style={{ paddingTop: "4vh", paddingBottom: "3vh" }}>
        <p className="eyebrow">Experiments</p>
        <h1 className="h-serif display-l" style={{ marginTop: "0.6rem" }}>
          Fragments that<br />
          <em style={{ fontWeight: 300 }}>misbehave well</em>
        </h1>
      </motion.header>
      <div>
        {EXPERIMENTS.map((e, i) => (
          <motion.div key={e.id} className="row-item" variants={rise} initial="hidden" animate="show" custom={i + 1} style={{ cursor: "default" }}>
            <span className="num">{e.index}</span>
            <span>
              <span className="title" style={{ display: "block" }}>
                {e.title}
              </span>
              <span className="body-copy" style={{ display: "block", marginTop: "0.4rem", fontSize: "0.92rem" }}>
                {e.note}
              </span>
            </span>
            <span className="meta">In progress</span>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

/* ---------------- LAB ---------------- */

export function LabPanel({ onBack }: { onBack: () => void }) {
  return (
    <div className="panel-pad" style={{ display: "flex", flexDirection: "column", padding: "0 clamp(1.2rem,5vw,4.5rem)" }}>
      <div style={{ position: "sticky", top: 0, paddingTop: "1.2rem", zIndex: 5, display: "flex", justifyContent: "flex-end" }}>
        <BackBar label="Environment" onBack={onBack} />
      </div>
      <motion.header variants={rise} initial="hidden" animate="show" custom={0} style={{ paddingTop: "4vh", paddingBottom: "3vh" }}>
        <p className="eyebrow">Lab</p>
        <h1 className="h-serif display-l" style={{ marginTop: "0.6rem" }}>
          Notes from the<br />
          <em style={{ fontWeight: 300 }}>instrument bench</em>
        </h1>
      </motion.header>
      <div style={{ maxWidth: "46rem" }}>
        {LAB_NOTES.map((n, i) => (
          <motion.article key={n.id} variants={rise} initial="hidden" animate="show" custom={i + 1} style={{ padding: "1.6rem 0", borderBottom: "1px solid var(--hairline)" }}>
            <h2 className="h-serif display-m" style={{ marginBottom: "0.7rem" }}>
              {n.title}
            </h2>
            <p className="body-copy">{n.body}</p>
          </motion.article>
        ))}
      </div>
    </div>
  );
}

/* ---------------- CONTACT ---------------- */

export function ContactPanel({ onBack }: { onBack: () => void }) {
  return (
    <div className="panel-pad" style={{ display: "flex", flexDirection: "column", padding: "0 clamp(1.2rem,5vw,4.5rem)" }}>
      <div style={{ position: "sticky", top: 0, paddingTop: "1.2rem", zIndex: 5, display: "flex", justifyContent: "flex-end" }}>
        <BackBar label="Environment" onBack={onBack} />
      </div>
      <motion.div initial="hidden" animate="show" style={{ maxWidth: "44rem", paddingTop: "4vh", paddingBottom: "6vh" }}>
        <motion.p className="eyebrow" variants={rise} custom={0}>
          Contact
        </motion.p>
        <motion.h1 className="h-serif display-l" variants={rise} custom={1} style={{ margin: "0.8rem 0 2rem" }}>
          Let&apos;s build<br />
          <em style={{ fontWeight: 300 }}>a place</em>
        </motion.h1>
        <motion.p variants={rise} custom={2} className="body-copy" style={{ fontSize: "1.15rem", color: "var(--ink)" }}>
          Tell us what should exist. We answer within two working days.
        </motion.p>
        <motion.div variants={rise} custom={3} style={{ marginTop: "2.5rem" }}>
          <p className="eyebrow" style={{ marginBottom: "0.6rem" }}>New business</p>
          <a className="h-serif display-m" href={`mailto:${CONTACT.email}`} style={{ display: "inline-block" }}>
            {CONTACT.email}
          </a>
        </motion.div>
        <motion.div variants={rise} custom={4} className="spec-grid" style={{ marginTop: "2.5rem" }}>
          {CONTACT.offices.map((o) => (
            <div key={o.city}>
              <dt>{o.city}</dt>
              <dd>{o.address}</dd>
            </div>
          ))}
          <div>
            <dt>Social</dt>
            {CONTACT.socials.map((s) => (
              <dd key={s.label} style={{ opacity: 0.75 }}>
                {s.label} — {s.handle}
              </dd>
            ))}
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}
