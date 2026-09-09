import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';

export default function ReliabilityChart({
  curveData = [
    { pred: 0.05, obs: 0.04, count: 420 },
    { pred: 0.15, obs: 0.12, count: 210 },
    { pred: 0.25, obs: 0.24, count: 150 },
    { pred: 0.35, obs: 0.38, count: 90 },
    { pred: 0.45, obs: 0.47, count: 65 },
    { pred: 0.55, obs: 0.54, count: 50 },
    { pred: 0.65, obs: 0.68, count: 42 },
    { pred: 0.75, obs: 0.76, count: 38 },
    { pred: 0.85, obs: 0.89, count: 25 },
    { pred: 0.95, obs: 0.94, count: 18 },
  ],
  brierScore = 0.082,
  height = 240,
}) {
  const chartData = curveData.map((d) => ({
    pred: (d.pred * 100).toFixed(0),
    obs: Number((d.obs * 100).toFixed(1)),
    ideal: Number((d.pred * 100).toFixed(1)),
    count: d.count,
  }));

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const p = payload[0].payload;
      return (
        <div className="bg-space-900 border border-space-700 p-2.5 rounded-lg shadow-xl text-xs font-mono">
          <div className="text-slate-400 text-[10px] mb-1">Bin: {p.pred}% Predicted</div>
          <div className="text-sky-400 flex justify-between gap-3">
            <span>Observed Frequency:</span>
            <span className="font-bold text-white">{p.obs}%</span>
          </div>
          <div className="text-slate-400 text-[10px] mt-1">
            Samples in bin: {p.count}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-xs uppercase tracking-wider font-semibold text-slate-300 font-mono">
          Probability Calibration Curve
        </h4>
        <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/30">
          Brier Score: {brierScore.toFixed(4)}
        </span>
      </div>

      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <LineChart data={chartData} margin={{ top: 10, right: 15, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.5} />
            <XAxis
              dataKey="pred"
              stroke="#64748b"
              tick={{ fontSize: 10, fill: '#64748b', fontFamily: 'JetBrains Mono' }}
              label={{ value: 'Mean Predicted P(%)', position: 'insideBottom', offset: -4, fill: '#64748b', fontSize: 10 }}
            />
            <YAxis
              stroke="#64748b"
              tick={{ fontSize: 10, fill: '#64748b', fontFamily: 'JetBrains Mono' }}
              domain={[0, 100]}
              tickFormatter={(v) => `${v}%`}
              label={{ value: 'Observed (%)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Perfectly Calibrated Diagonal */}
            <Line
              type="monotone"
              dataKey="ideal"
              stroke="#475569"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              dot={false}
              name="Perfect Calibration"
            />

            {/* Empirical Model Calibration */}
            <Line
              type="monotone"
              dataKey="obs"
              stroke="#38bdf8"
              strokeWidth={2.5}
              dot={{ fill: '#0284c7', r: 3 }}
              name="Model Probability"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center justify-between mt-2 pt-2 border-t border-space-800 text-[10px] font-mono text-slate-400">
        <span>Brier Score &rarr; 0 denotes perfect probabilistic calibration</span>
        <span className="text-sky-400">Cyan: Empirical Reliability</span>
      </div>
    </div>
  );
}
