import React from 'react';
import { CheckCircle2, AlertTriangle, Activity, Database } from 'lucide-react';

export default function DataQualityCard({
  qualityScore = 0.98,
  cadence = '60s',
  totalRows = 14400,
  missingCount = 12,
  channels = ['SoLEXS (1-30 keV)', 'HEL1OS (10-150 keV)'],
}) {
  const percentScore = (qualityScore * 100).toFixed(1);
  const isHealthy = qualityScore >= 0.90;

  return (
    <div className="panel p-4 border-space-750">
      <div className="flex items-center justify-between border-b border-space-800 pb-3 mb-3">
        <div className="flex items-center gap-2">
          <Activity size={16} className="text-solar-400" />
          <h4 className="text-xs uppercase tracking-wider font-semibold text-slate-300">
            Telemetry Quality & Integrity
          </h4>
        </div>
        <span className={`inline-flex items-center gap-1 text-xs font-mono font-semibold px-2 py-0.5 rounded-full ${
          isHealthy ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30' : 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
        }`}>
          {isHealthy ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}
          {percentScore}% HEALTH
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
        <div className="bg-space-950/60 p-2.5 rounded-lg border border-space-800">
          <span className="text-slate-400 block text-[11px]">Sampling Cadence</span>
          <span className="font-mono text-white font-medium text-sm mt-0.5 block">{cadence}</span>
        </div>
        <div className="bg-space-950/60 p-2.5 rounded-lg border border-space-800">
          <span className="text-slate-400 block text-[11px]">Total Timestamps</span>
          <span className="font-mono text-white font-medium text-sm mt-0.5 block">
            {Number(totalRows || 0).toLocaleString()}
          </span>
        </div>
        <div className="bg-space-950/60 p-2.5 rounded-lg border border-space-800">
          <span className="text-slate-400 block text-[11px]">Missing / Gaps</span>
          <span className="font-mono text-white font-medium text-sm mt-0.5 block">
            {missingCount} pts ({((missingCount / Math.max(1, totalRows)) * 100).toFixed(2)}%)
          </span>
        </div>
        <div className="bg-space-950/60 p-2.5 rounded-lg border border-space-800">
          <span className="text-slate-400 block text-[11px]">Dual Channels</span>
          <span className="font-mono text-solar-400 font-medium text-xs mt-0.5 block truncate">
            SoLEXS + HEL1OS
          </span>
        </div>
      </div>
    </div>
  );
}
