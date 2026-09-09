import React from 'react';

const STATE_CONFIG = {
  quiet: {
    label: 'QUIET',
    desc: 'Baseline solar corona; no active flare precursors detected.',
    color: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
    dot: 'bg-emerald-400 shadow-[0_0_8px_#10b981]',
  },
  pre_flare: {
    label: 'PRE-FLARE',
    desc: 'Impulsive hard X-ray rise / thermal pre-heating detected.',
    color: 'text-solar-400 bg-solar-500/15 border-solar-500/40 shadow-glow-solar',
    dot: 'bg-solar-400 animate-ping',
  },
  flare: {
    label: 'FLARE IN PROGRESS',
    desc: 'X-ray flux actively peaking across SoLEXS & HEL1OS.',
    color: 'text-rose-400 bg-rose-500/20 border-rose-500/50 shadow-glow-danger',
    dot: 'bg-rose-500 animate-pulse',
  },
  decay: {
    label: 'DECAY PHASE',
    desc: 'Thermal plasma cooling and flux descending towards baseline.',
    color: 'text-sky-400 bg-sky-500/10 border-sky-500/30',
    dot: 'bg-sky-400',
  },
};

export default function StatusIndicator({ state = 'quiet', confidence = null, showDetails = false }) {
  const norm = (state || 'quiet').toLowerCase().replace(/\s+/g, '_');
  const cfg = STATE_CONFIG[norm] || STATE_CONFIG.quiet;

  return (
    <div className="inline-flex flex-col">
      <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border font-mono text-xs font-semibold uppercase tracking-wider ${cfg.color}`}>
        <span className="relative flex h-2 w-2">
          <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${cfg.dot}`} />
          <span className={`relative inline-flex rounded-full h-2 w-2 ${cfg.dot}`} />
        </span>
        <span>{cfg.label}</span>
        {confidence !== null && (
          <span className="text-slate-400 font-normal pl-1 border-l border-slate-700">
            {(confidence * 100).toFixed(0)}% CONF
          </span>
        )}
      </div>
      {showDetails && (
        <p className="text-[11px] text-slate-400 mt-1.5 max-w-xs">{cfg.desc}</p>
      )}
    </div>
  );
}
