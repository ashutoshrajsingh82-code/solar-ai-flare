import React from 'react';
import { Link } from 'react-router-dom';
import { Cpu, CheckCircle, Zap, Activity, Award } from 'lucide-react';

// Where each model sits on the baseline ladder (used for the badge and for grouping on the Models page).
const TIERS = {
  climatology: { label: 'No-Skill Reference', order: 0, cls: 'bg-slate-500/15 text-slate-300 border-slate-500/30' },
  flux_rule: { label: 'Physics Rule', order: 1, cls: 'bg-amber-500/10 text-amber-300 border-amber-500/30' },
  hard_soft_ratio_rule: { label: 'Physics Rule', order: 2, cls: 'bg-amber-500/10 text-amber-300 border-amber-500/30' },
  two_feature_logreg: { label: 'Operational-Style', order: 3, cls: 'bg-teal-500/10 text-teal-300 border-teal-500/30' },
  logistic_regression: { label: 'Classical ML', order: 4, cls: 'bg-sky-500/10 text-sky-300 border-sky-500/30' },
  random_forest: { label: 'Classical ML', order: 5, cls: 'bg-sky-500/10 text-sky-300 border-sky-500/30' },
  svm: { label: 'Classical ML', order: 6, cls: 'bg-sky-500/10 text-sky-300 border-sky-500/30' },
  gradient_boosting: { label: 'Classical ML', order: 7, cls: 'bg-sky-500/10 text-sky-300 border-sky-500/30' },
};

export function modelTier(model) {
  return (
    TIERS[model?.architecture] || {
      label: 'Deep Sequence',
      order: 100,
      cls: 'bg-solar-500/10 text-solar-300 border-solar-500/30',
    }
  );
}

export default function ModelCard({
  model,
  isActive = false,
  onSelect = () => {},
  onInspect = () => {},
}) {
  const metrics = model.status === 'untrained' ? {} : (model.metrics || {});
  const forecast1h = metrics.forecast_1h || {};
  // No made-up fallbacks: a metric that was not computed shows as an em dash.
  const tss = forecast1h.tss ?? metrics.tss ?? null;
  const hss = forecast1h.hss ?? metrics.hss ?? null;
  const f1 = forecast1h.f1 ?? metrics.f1 ?? null;
  const far = forecast1h.false_alarm_rate ?? metrics.false_alarm_rate ?? null;
  const fmt = (v) => (typeof v === 'number' ? v.toFixed(3) : '\u2014');
  const tier = modelTier(model);
  const referenceOnly = Boolean(model.params?.reference_only);

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

        <div className="mt-3 flex flex-wrap items-center gap-2">
          <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${tier.cls}`}>
            {tier.label}
          </span>
          {referenceOnly && (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-space-800 text-slate-400 border border-space-700">
              Benchmark Reference
            </span>
          )}
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-space-800 text-slate-300 border border-space-700">
            Input: {model.input_configuration || 'SoLEXS + HEL1OS (Fused)'}
          </span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-space-800 text-slate-300 border border-space-700">
            {model.status || 'untrained'}
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
              {fmt(tss)}
            </span>
          </div>
          <div className="bg-space-950/60 p-2 rounded border border-space-800">
            <span className="text-[10px] text-slate-400 block font-sans">HSS</span>
            <span className="text-sm font-bold text-sky-400">
              {fmt(hss)}
            </span>
          </div>
          <div className="bg-space-950/60 p-2 rounded border border-space-800">
            <span className="text-[10px] text-slate-400 block font-sans">F1</span>
            <span className="text-sm font-bold text-emerald-400">
              {fmt(f1)}
            </span>
          </div>
          <div className="bg-space-950/60 p-2 rounded border border-space-800">
            <span className="text-[10px] text-slate-400 block font-sans">FAR</span>
            <span className="text-sm font-bold text-rose-400">
              {fmt(far)}
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

        {referenceOnly ? (
          <Link
            to={`/training?model=${model.architecture}`}
            className="px-3 py-1.5 rounded-lg text-xs font-mono font-semibold bg-space-800 hover:bg-solar-500 hover:text-space-950 text-slate-200 border border-space-700 transition-all"
          >
            Train to Deploy
          </Link>
        ) : !isActive && model.status !== 'untrained' ? (
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
