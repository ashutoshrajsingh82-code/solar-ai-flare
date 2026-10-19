import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export default function ReliabilityChart({ curveData = [], brierScore, height = 240 }) {
  const data = (curveData || []).filter((d) => Number.isFinite(d.predicted_mean) && Number.isFinite(d.observed_frequency));
  return <div className="w-full">
    <div className="flex justify-between mb-2"><h4 className="text-xs uppercase tracking-wider font-semibold text-slate-300">Probability Calibration</h4>
      <span className="text-xs font-mono text-slate-300">Brier Score: {Number.isFinite(brierScore) ? brierScore.toFixed(4) : 'N/A'}</span></div>
    {data.length === 0 ? <div style={{ height }} className="flex items-center justify-center text-sm text-slate-400 text-center px-6">Calibration data is unavailable. A reliability curve requires stored predictions and known labels.</div> :
      <div style={{ width: '100%', height }}><ResponsiveContainer><LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="predicted_mean" domain={[0, 1]} type="number" />
        <YAxis domain={[0, 1]} /><Tooltip /><Line dataKey="observed_frequency" stroke="#38bdf8" dot name="Observed frequency" />
      </LineChart></ResponsiveContainer></div>}
  </div>;
}
