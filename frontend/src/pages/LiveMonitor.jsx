import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Play, Pause, RotateCcw, Radio, ShieldAlert, Square } from 'lucide-react';
import StatusIndicator from '../components/StatusIndicator';
import RiskBadge from '../components/RiskBadge';
import DataQualityCard from '../components/DataQualityCard';
import XRayChart from '../charts/XRayChart';
import HardSoftRatioChart from '../charts/HardSoftRatioChart';
import ForecastChart from '../charts/ForecastChart';
import api from '../services/api';

const dateLabel = (value) => value ? new Date(value).toISOString().replace('T', ' ').replace('Z', ' UTC') : 'Unavailable';

export default function LiveMonitor() {
  const [models, setModels] = useState([]);
  const [modelId, setModelId] = useState('');
  const [readiness, setReadiness] = useState(null);
  const [simState, setSimState] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [speed, setSpeed] = useState(5);
  const [loading, setLoading] = useState(true);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    api.listModels().then((list) => {
      const trained = (list || []).filter((item) => item.status === 'trained');
      setModels(trained);
      const chosen = trained.find((item) => item.active && item.model_ready) || trained.find((item) => item.model_ready) || trained[0];
      if (chosen) setModelId(chosen.model_id);
    }).catch((err) => setError(err.response?.data?.detail || 'Unable to load model registry.'))
      .finally(() => setLoading(false));
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [state, currentAlerts] = await Promise.all([api.getSimulationState(), api.getAlerts(null, 100, true)]);
      setSimState(state); setAlerts(currentAlerts || []);
    } catch (err) { setError(err.response?.data?.detail || 'Unable to refresh replay status.'); }
  }, []);

  useEffect(() => {
    if (!modelId) { setReadiness(null); return; }
    let cancelled = false;
    api.getSimulationReadiness(modelId).then((value) => { if (!cancelled) setReadiness(value); })
      .catch((err) => { if (!cancelled) setReadiness({ ready: false, blockers: [err.response?.data?.detail || 'Readiness could not be assessed.'] }); });
    return () => { cancelled = true; };
  }, [modelId]);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 1500);
    return () => clearInterval(timer);
  }, [refresh]);

  const runAction = async (action) => {
    setWorking(true); setError('');
    try { await action(); await refresh(); }
    catch (err) { setError(err.response?.data?.detail?.message || err.response?.data?.detail || err.message || 'Replay action failed.'); }
    finally { setWorking(false); }
  };
  const start = () => runAction(() => api.startSimulation({ model_id: modelId, dataset_id: readiness?.dataset_id, speed }));
  const handleSpeed = (value) => { setSpeed(value); return runAction(() => api.setSimulationSpeed(value)); };

  const latest = simState?.latest || {};
  const history = simState?.recent_history || [];
  const isRunning = simState?.status === 'running';
  const isPaused = simState?.status === 'paused';
  const activeModel = models.find((item) => item.model_id === modelId);
  const warnings = latest.data_quality?.warnings || [];
  const completeReadiness = readiness?.ready === true;
  const quality = latest.data_quality?.valid_measurement_count === 2 ? 1 : latest.data_quality ? 0.5 : 0;
  const latestAlert = useMemo(() => alerts[0] || null, [alerts]);
  const horizon = Object.keys(latest.forecast || {})[0] || '1h';

  return <div className="space-y-6">
    <div className="panel p-5 border-amber-500/40 bg-amber-950/20">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="flex gap-3"><AlertTriangle className="text-amber-400 shrink-0 mt-1" size={20} /><div>
          <h2 className="text-lg font-bold text-white">Historical Monitoring Replay</h2>
          <p className="text-sm text-amber-200">Historical held-out test replay only. No real-time satellite feed is connected; replay timestamps are never presented as live telemetry.</p>
          <p className="mt-1 text-xs text-slate-400">Source: {simState?.source_label || 'Real-time satellite feed unavailable'} · Replay status: {simState?.status || 'idle'}</p>
        </div></div>
        <div className="flex flex-wrap items-center gap-2">
          <select aria-label="Verified trained model" value={modelId} onChange={(event) => setModelId(event.target.value)} className="bg-space-950 text-white border border-space-700 rounded px-2 py-2 text-sm" disabled={!models.length}>
            {models.length ? models.map((item) => <option key={item.model_id} value={item.model_id}>{item.name} · {item.version}</option>) : <option value="">No trained model</option>}
          </select>
          {!isRunning && !isPaused && <button onClick={start} disabled={working || !completeReadiness} className="px-3 py-2 rounded bg-solar-500 text-space-950 font-bold text-xs flex gap-1 items-center disabled:opacity-40"><Play size={14} />Start test replay</button>}
          {isRunning && <button onClick={() => runAction(api.pauseSimulation)} disabled={working} className="px-3 py-2 rounded border border-amber-400 text-amber-200 text-xs flex gap-1 items-center"><Pause size={14} />Pause</button>}
          {isPaused && <button onClick={() => runAction(api.resumeSimulation)} disabled={working} className="px-3 py-2 rounded border border-emerald-400 text-emerald-200 text-xs flex gap-1 items-center"><Play size={14} />Resume</button>}
          {(isRunning || isPaused) && <button onClick={() => runAction(api.stopSimulation)} disabled={working} className="px-3 py-2 rounded border border-rose-400 text-rose-200 text-xs flex gap-1 items-center"><Square size={14} />Stop</button>}
          <button onClick={() => runAction(api.resetSimulation)} disabled={working} className="px-3 py-2 rounded border border-space-700 text-slate-300 text-xs flex gap-1 items-center"><RotateCcw size={14} />Clear view</button>
          <label className="text-xs text-slate-400">Replay rate <select value={speed} onChange={(event) => handleSpeed(Number(event.target.value))} className="bg-space-950 text-slate-200 border border-space-700 rounded p-1">{[1, 5, 10, 50].map((item) => <option key={item} value={item}>{item}×</option>)}</select></label>
        </div>
      </div>
    </div>

    {error && <div role="alert" className="panel p-3 border-rose-500/40 text-rose-200 text-sm">{typeof error === 'string' ? error : JSON.stringify(error)}</div>}
    {loading && <div className="panel p-4 text-sm text-slate-400">Loading model and replay status…</div>}
    {!loading && readiness && !completeReadiness && <div className="panel p-4 border-amber-500/30"><h3 className="font-semibold text-amber-200">Replay unavailable</h3><p className="text-xs text-slate-400 mt-1">A verified trained checkpoint and a matching, provenance-verified held-out test partition are required.</p><ul className="list-disc pl-5 mt-2 text-xs text-slate-300 space-y-1">{readiness.blockers?.map((reason) => <li key={reason}>{reason}</li>)}</ul></div>}

    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      <div className="panel p-3"><span className="text-[10px] uppercase text-slate-400">Latest observation UTC</span><p className="text-xs text-white mt-1">{dateLabel(latest.observation_timestamp_utc)}</p></div>
      <div className="panel p-3"><span className="text-[10px] uppercase text-slate-400">Forecast issue UTC</span><p className="text-xs text-white mt-1">{dateLabel(latest.forecast_issue_timestamp_utc)}</p></div>
      <div className="panel p-3"><span className="text-[10px] uppercase text-slate-400">Model verification</span><p className="text-xs text-white mt-1">{activeModel ? `${activeModel.name} · ${activeModel.version}` : 'Unavailable'} · {completeReadiness ? 'Verified trained artifact' : 'Not verified'}</p></div>
      <div className="panel p-3"><span className="text-[10px] uppercase text-slate-400">Replay dataset / period</span><p className="text-xs text-white mt-1">{simState?.provenance?.dataset_name || readiness?.dataset_id || 'Unavailable'} · {simState?.provenance?.period_start_utc || readiness?.period?.start_utc || '—'} to {simState?.provenance?.period_end_utc || readiness?.period?.end_utc || '—'}</p></div>
    </div>

    {simState?.status === 'idle' && <div className="panel p-8 text-center text-slate-400">No held-out replay results are loaded. Starting a replay requires the readiness checks above.</div>}
    {latest && latest.observation_timestamp_utc && <>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="panel p-3"><span className="text-[10px] text-slate-400 uppercase">Prediction score · {horizon}</span><p className="text-xl text-white font-mono">{Number.isFinite(latest.forecast?.[horizon]) ? latest.forecast[horizon].toFixed(3) : 'Unavailable'}</p><p className="text-[10px] text-slate-500">Uncalibrated model score, not event probability</p></div>
        <div className="panel p-3"><span className="text-[10px] text-slate-400 uppercase">Risk band</span><div className="mt-2"><RiskBadge level={latest.risk || 'LOW'} size="sm" /></div><p className="text-[10px] text-slate-500 mt-1">Provisional operational category</p></div>
        <div className="panel p-3"><span className="text-[10px] text-slate-400 uppercase">Nowcast</span><div className="mt-2"><StatusIndicator state={latest.nowcast?.state || 'unknown'} confidence={latest.nowcast?.confidence} /></div></div>
        <div className="panel p-3"><span className="text-[10px] text-slate-400 uppercase">Alert state</span><p className="text-sm text-white mt-2">{latestAlert ? `${latestAlert.state} · ${latestAlert.severity} · ${Number(latestAlert.model_score).toFixed(3)}` : 'No active alert'}</p><p className="text-[10px] text-slate-500">{latestAlert?.source_classification === 'historical_test_replay' ? 'Historical replay alert' : latestAlert?.source_classification || 'No alert'}</p></div>
      </div>
      {!!warnings.length && <div className="panel p-3 text-xs text-amber-200">Data-quality warnings: {warnings.join(' · ')}</div>}
      {latest.error && <div className="panel p-3 text-xs text-rose-200">Prediction failed: {latest.error}</div>}
      <div className="panel p-4"><h3 className="text-sm text-slate-200 mb-3">Historical observations through forecast issue time</h3><XRayChart data={history} height={300} /></div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5"><div className="lg:col-span-2 panel p-4"><HardSoftRatioChart data={history} height={220} /></div><div className="panel p-4"><ForecastChart forecast={latest.forecast || {}} height={220} /></div></div>
      <DataQualityCard qualityScore={quality} cadence={`${latest.data_quality?.cadence_seconds_expected || '—'}s`} totalRows={simState?.total_rows || 0} missingCount={warnings.length} />
      <div className="panel p-4"><h3 className="font-semibold text-white flex items-center gap-2"><ShieldAlert size={16} />Outcome maturity</h3><p className="text-xs text-slate-400 mt-1">Forecast labels are attached only after each forecast window has elapsed in replay time. Current prediction outcome status: {JSON.stringify(latest.outcomes || { status: 'pending' })}</p></div>
    </>}
  </div>;
}
