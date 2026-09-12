import React from 'react';

export default function MetricCard({
  title,
  value,
  unit = '',
  subtitle = null,
  icon: Icon = null,
  badge = null,
  accent = 'solar', // 'solar' | 'helios' | 'danger' | 'emerald' | 'slate'
  className = '',
}) {
  const accentBorder = {
    solar: 'border-solar-500/30 hover:border-solar-500/60',
    helios: 'border-sky-500/30 hover:border-sky-500/60',
    danger: 'border-rose-500/30 hover:border-rose-500/60',
    emerald: 'border-emerald-500/30 hover:border-emerald-500/60',
    slate: 'border-space-750 hover:border-space-600',
  }[accent] || 'border-space-750';

  const iconColor = {
    solar: 'text-solar-400 bg-solar-500/10',
    helios: 'text-sky-400 bg-sky-500/10',
    danger: 'text-rose-400 bg-rose-500/10',
    emerald: 'text-emerald-400 bg-emerald-500/10',
    slate: 'text-slate-400 bg-space-800',
  }[accent] || 'text-slate-400 bg-space-800';

  return (
    <div className={`panel p-4 transition-all duration-200 ${accentBorder} ${className}`}>
      <div className="flex items-start justify-between">
        <div>
          <span className="text-xs uppercase tracking-wider text-slate-400 font-medium">
            {title}
          </span>
          <div className="flex items-baseline gap-1.5 mt-1.5">
            <span className="text-2xl font-bold font-mono tracking-tight text-white">
              {value !== undefined && value !== null ? value : '--'}
            </span>
            {unit && (
              <span className="text-xs font-mono text-slate-400">{unit}</span>
            )}
          </div>
        </div>

        <div className="flex flex-col items-end gap-2">
          {Icon && (
            <div className={`p-2 rounded-lg ${iconColor}`}>
              <Icon size={18} />
            </div>
          )}
          {badge}
        </div>
      </div>

      {subtitle && (
        <div className="mt-2 text-xs text-slate-400 flex items-center gap-1.5 border-t border-space-800/80 pt-2">
          {subtitle}
        </div>
      )}
    </div>
  );
}
