import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } from 'recharts';

export default function ReliabilityChart({ curveData, brierScore, height = 240 }) {
  const reliability = curveData || {};
  const data = (reliability.bins || []).map((bin) => ({
    predicted: bin.mean_predicted_probability,
    observed: bin.observed_event_frequency,
    count: bin.sample_count,
  }));
  return <div className="w-full">
    <div className="flex justify-between mb-2"><h4 className="text-xs uppercase tracking-wider font-semibold text-slate-300">Probability Calibration</h4>
      <span className="text-xs font-mono text-slate-300">Brier: {Number.isFinite(brierScore) ? brierScore.toFixed(4) : 'N/A'}</span></div>
    {reliability.status !== 'available' ? <div style={{ height }} className="flex items-center justify-center text-sm text-slate-400 text-center px-6">{reliability.reason || 'Calibration data is unavailable. Stored test predictions are required.'}</div> :
      <div style={{ width: '100%', height }}><ResponsiveContainer><LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="predicted" domain={[0, 1]} type="number" label={{ value: 'Mean predicted probability', position: 'insideBottom', offset: -4 }} />
        <YAxis domain={[0, 1]} label={{ value: 'Observed event frequency', angle: -90, position: 'insideLeft' }} /><Tooltip content={({ active, payload }) => active && payload?.length ? <div className="rounded border border-space-700 bg-space-950 p-2 text-xs text-slate-200"><div>Mean predicted: {Number(payload[0].payload.predicted).toFixed(3)}</div><div>Observed frequency: {Number(payload[0].payload.observed).toFixed(3)}</div><div>Samples: {payload[0].payload.count}</div></div> : null} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#64748b" strokeDasharray="5 4" />
        <Line dataKey="observed" stroke="#38bdf8" dot name="Observed frequency" />
      </LineChart></ResponsiveContainer></div>}
    {data.length > 0 && <p className="text-[10px] text-slate-500 mt-1">Fixed-width probability bins · point tooltip includes sample count</p>}
  </div>;
}
