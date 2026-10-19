import React, { useEffect, useState } from 'react';
import { CheckCircle, Info } from 'lucide-react';
import MetricCard from '../components/MetricCard';
import ConfusionMatrix from '../charts/ConfusionMatrix';
import ReliabilityChart from '../charts/ReliabilityChart';
import api from '../services/api';

export default function Evaluation() {
  const [modelId, setModelId] = useState('');
  const [modelMetrics, setModelMetrics] = useState(null);
  const [models, setModels] = useState([]);
  useEffect(() => {
    api.listModels().then(async (list) => {
      setModels(list);
      const selected = list.find((m) => m.active) || list[0];
      if (selected) {
        setModelId(selected.model_id);
        setModelMetrics(await api.getModelMetrics(selected.model_id));
      }
    }).catch(console.error);
  }, []);
  const changeModel = async (id) => {
    setModelId(id);
    setModelMetrics(await api.getModelMetrics(id));
  };
  const eligible = modelMetrics?.status === 'trained' || modelMetrics?.status === 'demo_only';
  const oneHour = eligible ? (modelMetrics?.metrics?.forecast_per_horizon?.['1h'] || modelMetrics?.forecast_per_horizon?.['1h'] || {}) : {};
  const now = eligible ? (modelMetrics?.metrics || modelMetrics || {}) : {};
  const fmt = (value) => Number.isFinite(value) ? value.toFixed(3) : 'N/A';
  return <div className="space-y-6">
    <div className="panel p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div><span className="text-[10px] font-mono uppercase tracking-widest text-emerald-400">Evaluation results</span>
        <h2 className="text-xl font-bold text-white flex items-center gap-2"><CheckCircle size={22} className="text-solar-400" />Model Evaluation</h2>
        <p className="text-xs text-slate-400">Stored metrics from the model's recorded evaluation. Threshold changes require backend recomputation.</p></div>
      {models.length > 0 && <select value={modelId} onChange={(e) => changeModel(e.target.value)} className="bg-space-900 text-solar-400 border border-space-750 rounded px-2 py-1">
        {models.map((m) => <option key={m.model_id} value={m.model_id}>{m.name}</option>)}
      </select>}
    </div>
    {!modelMetrics && <div className="panel p-5 text-sm text-slate-400">No model evaluation is available yet. Train a model to create an evaluation record.</div>}
    {modelMetrics && !eligible && <div className="panel p-5 text-sm text-slate-400">This record is untrained or has uncertain legacy provenance. Evaluation metrics are unavailable until a new training run completes.</div>}
    {modelMetrics && eligible && <>
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
        {[["TSS", oneHour.tss], ["HSS", oneHour.hss], ["POD / Recall", oneHour.recall], ["Precision", oneHour.precision], ["FAR", oneHour.false_alarm_rate], ["F1-Score", oneHour.f1], ["Brier Score", oneHour.brier_score]].map(([title, value]) => <MetricCard key={title} title={title} value={fmt(value)} subtitle="1h forecast evaluation" accent="solar" />)}
      </div>
      <div className="panel p-4 text-xs text-slate-400 flex gap-2"><Info size={16} className="shrink-0" />Metrics represent the recorded test split only when metadata confirms that split. Synthetic/demo data results are not real-world solar-flare validation.</div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="panel p-5"><ConfusionMatrix matrix={now.nowcast_confusion_matrix} /></div>
        <div className="panel p-5"><ReliabilityChart curveData={oneHour.calibration_bins || []} brierScore={oneHour.brier_score} /></div>
      </div>
      <p className="text-xs text-slate-500">Evaluation split: {now.evaluation_split || 'not recorded'} · Samples: {now.evaluation_samples ?? 'not recorded'} · Status: {modelMetrics.status || 'legacy / unknown'}</p>
    </>}
  </div>;
}
