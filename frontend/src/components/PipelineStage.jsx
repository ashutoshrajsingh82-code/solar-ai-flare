import React from 'react';
import { CheckCircle2, Clock, AlertTriangle, ArrowRight, Layers, FileCode, Check } from 'lucide-react';

export default function PipelineStage({
  stageKey,
  title,
  description,
  status = 'pending', // 'complete' | 'pending' | 'active'
  inputRows = null,
  outputRows = null,
  durationSec = null,
  warnings = [],
  missingValues = null,
  isSelected = false,
  onClick = () => {},
}) {
  const isComplete = status === 'complete';
  const hasWarnings = warnings && warnings.length > 0;

  return (
    <div
      onClick={onClick}
      className={`panel cursor-pointer transition-all duration-200 text-left p-4 relative overflow-hidden ${
        isSelected
          ? 'border-solar-500 shadow-glow-solar ring-1 ring-solar-500/50 bg-space-850'
          : isComplete
          ? 'border-space-750 hover:border-space-600 bg-space-900/90'
          : 'border-space-850 opacity-60 bg-space-950/60'
      }`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <div className={`p-1.5 rounded-md ${
            isComplete
              ? 'bg-emerald-500/10 text-emerald-400'
              : 'bg-space-800 text-slate-500'
          }`}>
            {isComplete ? <CheckCircle2 size={16} /> : <Clock size={16} />}
          </div>
          <div>
            <h4 className="text-sm font-semibold text-white tracking-tight">
              {title}
            </h4>
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
              {stageKey}
            </span>
          </div>
        </div>

        <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full uppercase tracking-wider font-semibold ${
          isComplete
            ? hasWarnings
              ? 'bg-amber-500/10 text-amber-300 border border-amber-500/30'
              : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
            : 'bg-space-800 text-slate-500'
        }`}>
          {isComplete ? (hasWarnings ? 'Warning' : 'Complete') : 'Pending'}
        </span>
      </div>

      <p className="text-xs text-slate-400 mt-2.5 line-clamp-2 leading-relaxed">
        {description}
      </p>

      {isComplete && (
        <div className="mt-3 pt-3 border-t border-space-800/80 grid grid-cols-3 gap-2 text-[11px] font-mono">
          <div>
            <span className="text-slate-500 block text-[10px]">IN ROWS</span>
            <span className="text-slate-200 font-medium">
              {inputRows !== null ? Number(inputRows).toLocaleString() : '--'}
            </span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px]">OUT ROWS</span>
            <span className="text-solar-400 font-medium">
              {outputRows !== null ? Number(outputRows).toLocaleString() : '--'}
            </span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px]">LATENCY</span>
            <span className="text-slate-200 font-medium">
              {durationSec !== null ? `${durationSec}s` : '--'}
            </span>
          </div>
        </div>
      )}

      {hasWarnings && (
        <div className="mt-2 text-[10px] text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2 py-1 rounded flex items-center gap-1.5">
          <AlertTriangle size={12} />
          <span>{warnings.length} warning(s) logged</span>
        </div>
      )}
    </div>
  );
}
