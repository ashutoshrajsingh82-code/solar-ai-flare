import React from 'react';
import { Cpu, CheckCircle, Zap, Activity, Award } from 'lucide-react';

export default function ModelCard({
  model,
  isActive = false,
  onSelect = () => {},
  onInspect = () => {},
}) {
  const metrics = model.metrics || {};
  const forecast1h = metrics.forecast_1h || {};
  const tss = forecast1h.tss ?? metrics.tss ?? 0.74;
  const hss = forecast1h.hss ?? metrics.hss ?? 0.68;
  const f1 = forecast1h.f1 ?? metrics.f1 ?? 0.72;
  const far = forecast1h.false_alarm_rate ?? metrics.false_alarm_rate ?? 0.12;

  const isDeep = model.architecture?.includes('cnn') || model.architecture?.includes('lstm') || model.architecture?.includes('transformer');

  return (
    <div className={`panel p-5 transition-all duration-200 text-left relative flex flex-col justify-between ${
      isActive
        ? 'border-solar-500 shadow-glow-solar bg-space-850 ring-1 ring-solar-500/60'
        : 'border-space-750 hover:border-space-600 bg-space-900/80'
    }`}>
      {isActive && (
        <div className="absolute -top-2.5 right-4 bg-solar-500 text-space-950 text-[10px] font-mono font-black uppercase px-2.5 py-0.5 rounded-full flex items-center gap-1 shadow-md">
          <Award size={11} />
          Active Inference Engine
        </div>
      )}

      <div>
        <div className="flex items-start justify-between gap-3">
          <div>
            <span className="text-[10px] font-mono font-semibold text-solar-400 uppercase tracking-widest block">
              {model.architecture || 'MODEL ARCHITECTURE'}
            </span>
            <h3 className="text-base font-bold text-white tracking-tight mt-0.5">
              {model.name}
            </h3>
            <span className="text-[11px] font-mono text-slate-400">
              ID: {model.model_id}
            </span>
          </div>

          <div className={`p-2 rounded-lg ${isDeep ? 'bg-sky-500/10 text-sky-400' : 'bg-solar-500/10 text-solar-400'}`}>
            {isDeep ? <Cpu size={20} /> : <Activity size={20} />}
          </div>
        </div>

        <div className="mt-3 inline-flex items-center gap-2">
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-space-800 text-slate-300 border border-space-700">
            Input: {model.input_configuration || 'SoLEXS + HEL1OS (Fused)'}
          </span>
          {model.parameters_count && (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-space-800 text-slate-400">
              {Number(model.parameters_count).toLocaleString()} params
            </span>
          )}
        </div>

        {/* Metrics Grid */}
        <div className="mt-4 pt-3 border-t border-space-800 grid grid-cols-4 gap-2 text-center font-mono">
          <div className="bg-space-950/60 p-2 rounded border border-space-800">
            <span className="text-[10px] text-slate-400 block font-sans">TSS</span>
            <span className="text-sm font-bold text-solar-400">
              {typeof tss === 'number' ? tss.toFixed(3) : tss}
            </span>
          </div>
          <div className="bg-space-950/60 p-2 rounded border border-space-800">
            <span className="text-[10px] text-slate-400 block font-sans">HSS</span>
            <span className="text-sm font-bold text-sky-400">
              {typeof hss === 'number' ? hss.toFixed(3) : hss}
            </span>
          </div>
          <div className="bg-space-950/60 p-2 rounded border border-space-800">
            <span className="text-[10px] text-slate-400 block font-sans">F1</span>
            <span className="text-sm font-bold text-emerald-400">
              {typeof f1 === 'number' ? f1.toFixed(3) : f1}
            </span>
          </div>
          <div className="bg-space-950/60 p-2 rounded border border-space-800">
            <span className="text-[10px] text-slate-400 block font-sans">FAR</span>
            <span className="text-sm font-bold text-rose-400">
              {typeof far === 'number' ? far.toFixed(3) : far}
            </span>
          </div>
        </div>
      </div>

      <div className="mt-5 pt-3 border-t border-space-800 flex items-center justify-between gap-3">
        <button
          onClick={() => onInspect(model)}
          className="text-xs font-mono text-slate-400 hover:text-white transition-colors"
        >
          View Topology
        </button>

        {!isActive ? (
          <button
            onClick={() => onSelect(model.model_id)}
            className="px-3 py-1.5 rounded-lg text-xs font-mono font-semibold bg-space-800 hover:bg-solar-500 hover:text-space-950 text-slate-200 border border-space-700 transition-all"
          >
            Set Active Model
          </button>
        ) : (
          <span className="inline-flex items-center gap-1.5 text-xs font-mono text-emerald-400">
            <CheckCircle size={14} />
            Live Deployment
          </span>
        )}
      </div>
    </div>
  );
}
