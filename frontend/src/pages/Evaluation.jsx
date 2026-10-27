import React, { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, Info } from 'lucide-react';
import MetricCard from '../components/MetricCard';
import ReliabilityChart from '../charts/ReliabilityChart';
import api from '../services/api';

const horizons = ['1h', '3h', '6h', '12h', '24h'];
const show = (v) => Number.isFinite(v) ? v.toFixed(3) : 'N/A';

export default function Evaluation() {
  const [models, setModels] = useState([]);
  const [modelId, setModelId] = useState('');
  const [horizon, setHorizon] = useState('1h');
  const [threshold, setThreshold] = useState(0.5);
  const [thresholdTouched, setThresholdTouched] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    api.listModels().then((list) => {
      const eligible = (list || []).filter((m) => m.model_ready && ['trained', 'demo_only'].includes(m.status));
      setModels(eligible);
      const selected = eligible.find((m) => m.active) || eligible[0];
      if (selected) setModelId(selected.model_id);
      else setLoading(false);
    }).catch((e) => { setError(e.response?.data?.detail || 'Unable to load models.'); setLoading(false); });
  }, []);

  useEffect(() => {
    if (!modelId) return;
    let cancelled = false;
    setLoading(true); setError('');
    const timer = setTimeout(() => api.getModelEvaluation(modelId, horizon, thresholdTouched ? threshold : undefined)
      .then((data) => { if (!cancelled) { setResult(data); if (!thresholdTouched) setThreshold(data.threshold_used_for_test); } })
      .catch((e) => { if (!cancelled) { setResult(null); setError(e.response?.data?.detail || 'Evaluation is unavailable.'); } })
      .finally(() => { if (!cancelled) setLoading(false); }), thresholdTouched ? 180 : 0);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [modelId, horizon, threshold, thresholdTouched]);

  const m = result?.metrics;
  const confusion = useMemo(() => m ? [[m.tn, m.fp], [m.fn, m.tp]] : null, [m]);
  const provenance = result?.threshold_selection_provenance;

  return <div className="space-y-6">
    <div className="panel p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div><span className="text-[10px] font-mono uppercase tracking-widest text-emerald-400">Out-of-sample evaluation</span>
        <h2 className="text-xl font-bold text-white flex items-center gap-2"><BarChart3 size={22} className="text-solar-400" />Model Evaluation</h2>
        <p className="text-xs text-slate-400">Metrics are recomputed from the model’s stored chronological test labels and probabilities.</p></div>
      <div className="flex gap-2">
        <select aria-label="Model" value={modelId} onChange={(e) => { setModelId(e.target.value); setThresholdTouched(false); }} className="bg-space-900 text-solar-400 border border-space-750 rounded px-2 py-1">
          {models.map((model) => <option key={model.model_id} value={model.model_id}>{model.name}</option>)}
        </select>
        <select aria-label="Forecast horizon" value={horizon} onChange={(e) => setHorizon(e.target.value)} className="bg-space-900 text-solar-400 border border-space-750 rounded px-2 py-1">
          {horizons.map((h) => <option key={h}>{h}</option>)}
        </select>
      </div>
    </div>
    {!models.length && !loading && <div className="panel p-5 text-sm text-slate-400">No trained model artifact is available. Train a model to create an evaluation record.</div>}
    {loading && <div className="panel p-5 text-sm text-slate-400">Loading holdout evaluation…</div>}
    {error && !loading && <div className="panel p-5 text-sm text-amber-300 flex gap-2"><AlertTriangle size={16} />{error}</div>}
    {result && !loading && <>
      <div className="panel p-4 space-y-3">
        <div className="flex justify-between text-xs text-slate-300"><span>Decision threshold: <b className="text-white">{threshold.toFixed(2)}</b></span><span>{result.threshold_exploratory ? 'Exploratory test-set threshold' : 'Validation-selected threshold'}</span></div>
        <input aria-label="Decision threshold" type="range" min="0" max="1" step="0.01" value={threshold} onChange={(e) => { setThreshold(Number(e.target.value)); setThresholdTouched(true); }} className="w-full accent-amber-400" />
        <p className="text-xs text-slate-400">Validation-selected threshold: {result.validation_selected_threshold.toFixed(2)} · Threshold used for these test metrics: {result.threshold_used_for_test.toFixed(2)}. Selection provenance: {typeof provenance === 'object' ? (provenance?.method || JSON.stringify(provenance)) : (provenance || 'recorded validation procedure')}. {result.threshold_exploratory ? 'This alternate test threshold is exploratory and must not be reported as a validation-selected operating point.' : ''}</p>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3">
        {[["TSS", m.tss], ["HSS", m.hss], ["POD", m.pod], ["POFD", m.pofd], ["FAR", m.far], ["Precision", m.precision], ["Brier", m.brier_score], ["Accuracy", m.accuracy]].map(([title, value]) => <MetricCard key={title} title={title} value={show(value)} subtitle={`${horizon} test forecast`} accent="solar" />)}
      </div>
      <div className="panel p-4 text-xs text-slate-400 flex gap-2"><Info size={16} className="shrink-0" />{result.model_status === 'demo_only' ? 'These are synthetic/demo holdout metrics, not observational solar-flare validation.' : 'Test labels use the recorded reference-data provenance. Unknown coverage labels are excluded and reported separately.'}</div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="panel p-5"><h3 className="text-sm text-slate-200 mb-3">Binary confusion matrix · rows actual, columns predicted</h3>
          <table className="w-full text-center text-sm"><thead><tr><th></th><th>Predicted 0</th><th>Predicted 1</th></tr></thead><tbody>{confusion.map((row, i) => <tr key={i}><th className="py-2">Actual {i}</th>{row.map((count, j) => <td key={j} className="p-2 font-mono text-white">{count}</td>)}</tr>)}</tbody></table>
          <p className="mt-3 text-xs text-slate-400">{result.known_label_count} known labels ({m.positive_count} positive, {m.negative_count} negative); {result.unknown_label_count} unknown labels excluded.</p>
        </div>
        <div className="panel p-5"><ReliabilityChart curveData={result.reliability} brierScore={m.brier_score} /></div>
      </div>
      <div className="text-xs text-slate-500">{result.model_name} ({result.model_id}, version {result.version || 'unknown'}) · {result.evaluation_split} · {result.period_start_utc} to {result.period_end_utc} · {result.dataset?.filename || result.dataset?.source || 'dataset provenance unavailable'}</div>
      {m.undefined_metric_reasons && Object.keys(m.undefined_metric_reasons).length > 0 && <div className="panel p-4 text-xs text-amber-300">Undefined metrics: {Object.entries(m.undefined_metric_reasons).map(([key, value]) => `${key}: ${value}`).join(' · ')}</div>}
    </>}
  </div>;
}
