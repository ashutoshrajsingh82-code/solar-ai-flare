import React from 'react';

const RISK_STYLES = {
  LOW: {
    bg: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    dot: 'bg-emerald-400 shadow-[0_0_8px_#10b981]',
  },
  WATCH: {
    bg: 'bg-sky-500/10 text-sky-400 border-sky-500/30',
    dot: 'bg-sky-400 shadow-[0_0_8px_#38bdf8]',
  },
  MODERATE: {
    bg: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
    dot: 'bg-amber-400 shadow-[0_0_8px_#f59e0b]',
  },
  HIGH: {
    bg: 'bg-orange-500/15 text-orange-300 border-orange-500/40',
    dot: 'bg-orange-500 shadow-[0_0_10px_#f97316]',
  },
  CRITICAL: {
    bg: 'bg-rose-500/20 text-rose-300 border-rose-500/50 shadow-glow-danger animate-pulse',
    dot: 'bg-rose-500 shadow-[0_0_12px_#f43f5e]',
  },
};

export default function RiskBadge({ level = 'LOW', size = 'md' }) {
  const normLevel = (level || 'LOW').toUpperCase();
  const style = RISK_STYLES[normLevel] || RISK_STYLES.LOW;

  const sizeClasses = {
    sm: 'text-[10px] px-2 py-0.5',
    md: 'text-xs px-2.5 py-1',
    lg: 'text-sm px-3.5 py-1.5 font-bold',
  }[size] || 'text-xs px-2.5 py-1';

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border font-mono font-semibold uppercase tracking-wider ${style.bg} ${sizeClasses}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />
      {normLevel} RISK
    </span>
  );
}
