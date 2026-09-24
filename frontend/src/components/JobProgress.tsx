import { useEffect, useState } from "react";
import { api, Job } from "../api";

/**
 * Polls + subscribes to the SSE stream of a running analysis job and
 * renders a progress bar. Auto-stops on done/failed.
 */
export default function JobProgress({ jobId, onDone }: { jobId: string; onDone?: (status: string) => void }) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let es: EventSource | null = null;
    let poll: number | null = null;
    let finished = false;
    const update = (j: Job) => {
      setJob(j);
      if (j.status === "done" || j.status === "failed") {
        finished = true;
        es?.close();
        if (poll !== null) window.clearInterval(poll);
        onDone?.(j.status);
      }
    };

    api
      .job(jobId)
      .then(update)
      .catch((e) => setError(e.message));

    try {
      es = new EventSource(api.jobStreamUrl(jobId));
      es.onmessage = (ev) => {
        try {
          update(JSON.parse(ev.data) as Job);
        } catch {
          /* malformed event */
        }
      };
      es.onerror = () => {
        // EventSource reconnects; fall back to polling to stay responsive.
      };
    } catch {
      /* no SSE support */
    }

    poll = window.setInterval(async () => {
      if (finished) return;
      try {
        const j = await api.job(jobId);
        update(j);
      } catch {
        /* transient */
      }
    }, 1200);

    return () => {
      es?.close();
      if (poll !== null) window.clearInterval(poll);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  if (error) return <div className="text-sm text-rose-300">{error}</div>;
  if (!job) return null;

  const done = job.status === "done" || job.status === "failed";
  const pct = Math.round(job.progress ?? 0);
  return (
    <div
      className={`card card-pad animate-slide-up ${
        done && job.status === "done" ? "border-emerald-500/40 shadow-[0_0_18px_rgba(52,211,153,0.12)]" : ""
      }`}
    >
      <div className="flex items-center justify-between text-sm">
        <span className="inline-flex items-center gap-2 font-medium text-slate-200">
          {/* §4 wording — mirrors the real job status, never invented. */}
          <span
            className={`text-[10px] font-semibold uppercase tracking-widest ${
              done
                ? job.status === "done"
                  ? "text-emerald-400"
                  : "text-rose-400"
                : "text-sonar-400"
            }`}
          >
            {done ? (job.status === "done" ? "Analysis complete" : "Analysis failed") : "Analyzing sonar data"}
          </span>
          <span className="font-normal text-slate-400">{job.message || job.stage}</span>
        </span>
        <span className="mono text-slate-400">{pct}%</span>
      </div>
      <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-700/60">
        <div
          className={`h-full rounded-full transition-all duration-300 ${done && job.status === "done" ? "bg-emerald-400" : job.status === "failed" ? "bg-rose-500" : "bg-sonar-400"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="mt-2 flex items-center justify-between text-xs text-slate-500">
        <span>
          stage <span className="mono text-slate-400">{job.stage}</span>
        </span>
        {/* labels mirror the REAL backend job state — never invented stages */}
        {job.status === "running" && !done && <span className="animate-pulse text-sonar-400">analyzing…</span>}
        {done && job.status === "done" && <span className="font-medium text-emerald-400">✓ complete</span>}
        {job.status === "failed" && <span className="text-rose-400">{job.error ?? "failed"}</span>}
      </div>
    </div>
  );
}
