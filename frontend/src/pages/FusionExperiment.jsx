import React, { useEffect, useState } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import api from '../services/api';

export default function FusionExperiment() {
  const [readiness, setReadiness] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const refresh = () => {
    setLoading(true); setError('');
    api.getFusionExperiment().then(setReadiness).catch((e) => setError(e.response?.data?.detail || 'Unable to assess data readiness.')).finally(() => setLoading(false));
  };
  useEffect(refresh, []);
  return <div className="space-y-6">
    <div className="panel p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div><span className="text-[10px] font-mono uppercase tracking-widest text-solar-400">Scientific readiness assessment</span>
        <h2 className="text-2xl font-bold text-white mt-1">Multi-instrument fusion</h2>
        <p className="text-sm text-slate-400 mt-2">A paired comparison requires calibrated SoLEXS and HEL1OS channels with independently supported flare labels and identical evaluation examples.</p></div>
      <button onClick={refresh} disabled={loading} className="px-4 py-2 rounded-lg bg-solar-500 text-space-950 font-semibold text-sm flex items-center gap-2 disabled:opacity-50"><RefreshCw size={15} />Recheck readiness</button>
    </div>
    {loading && <div className="panel p-5 text-slate-400">Checking current data readiness…</div>}
    {error && <div className="panel p-5 text-amber-300">{error}</div>}
    {readiness && !loading && <>
      <div className="panel p-5 border-amber-500/30"><div className="flex items-start gap-3"><AlertTriangle className="text-amber-400 shrink-0" />
        <div><h3 className="font-bold text-amber-300">Experiment blocked · no scientific result</h3><p className="text-sm text-slate-300 mt-1">Conclusion: {readiness.scientific_conclusion?.replaceAll('_', ' ')}. No configurations were trained or compared.</p></div></div>
        <ul className="mt-4 space-y-2 text-sm text-slate-300 list-disc pl-5">{readiness.blockers?.map((blocker) => <li key={blocker}>{blocker}</li>)}</ul>
      </div>
      <div className="panel p-5"><h3 className="font-semibold text-white">Planned comparison</h3>
        <p className="text-xs text-slate-400 mt-1">Five paired random seeds · shared chronological splits · common known-label test cohort</p>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mt-4">{(readiness.configurations || []).map((name) => <div key={name} className="rounded border border-space-700 p-3 text-center text-xs text-slate-300">{name.replaceAll('_', ' ')}</div>)}</div>
        <p className="text-xs text-slate-500 mt-4">Seed runs completed: {readiness.completed_seed_count} / {readiness.requested_seed_count}. No scores, confidence intervals, or winner are reported.</p>
      </div>
    </>}
  </div>;
}
