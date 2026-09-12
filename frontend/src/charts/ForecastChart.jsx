import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  ReferenceLine,
} from 'recharts';
import RiskBadge from '../components/RiskBadge';

export default function ForecastChart({
  forecast = { '1h': 0.74, '3h': 0.68, '6h': 0.55, '12h': 0.39, '24h': 0.27 },
  height = 200,
}) {
  const horizons = ['1h', '3h', '6h', '12h', '24h'];

  const chartData = horizons.map((h) => {
    const prob = forecast && forecast[h] !== undefined ? Number(forecast[h]) : 0;
    const pct = prob * 100;

    let color = '#10b981'; // LOW
    let risk = 'LOW';
    if (pct >= 85) {
      color = '#f43f5e';
      risk = 'CRITICAL';
    } else if (pct >= 70) {
      color = '#f97316';
      risk = 'HIGH';
    } else if (pct >= 50) {
      color = '#f59e0b';
      risk = 'MODERATE';
    } else if (pct >= 30) {
      color = '#38bdf8';
      risk = 'WATCH';
    }

    return {
      horizon: `${h.toUpperCase()}`,
      probability: pct,
      rawProb: prob,
      color,
      risk,
    };
  });

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const p = payload[0].payload;
      return (
        <div className="bg-space-900 border border-space-700 p-2.5 rounded-lg shadow-xl text-xs font-mono">
          <div className="text-slate-400 text-[10px] mb-1">Horizon: {p.horizon} Ahead</div>
          <div className="flex items-center justify-between gap-4">
            <span className="text-slate-300">P(Flare):</span>
            <span className="font-bold" style={{ color: p.color }}>{p.probability.toFixed(1)}%</span>
          </div>
          <div className="mt-1">
            <RiskBadge level={p.risk} size="sm" />
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
          M/X Flare Forecast Horizons
        </h4>
        <span className="text-[11px] font-mono text-slate-400">
          Target: M1.0+ Event Onset
        </span>
      </div>

      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <XAxis
              dataKey="horizon"
              stroke="#64748b"
              tick={{ fontSize: 11, fill: '#cbd5e1', fontFamily: 'JetBrains Mono', fontWeight: 600 }}
            />
            <YAxis
              stroke="#64748b"
              tick={{ fontSize: 10, fill: '#64748b', fontFamily: 'JetBrains Mono' }}
              domain={[0, 100]}
              tickFormatter={(v) => `${v}%`}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={50} stroke="#f59e0b" strokeDasharray="3 3" />
            <ReferenceLine y={85} stroke="#f43f5e" strokeDasharray="3 3" />
            <Bar dataKey="probability" radius={[4, 4, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center justify-between mt-2 pt-2 border-t border-space-800 text-[11px] font-mono text-slate-400">
        <span>Low &lt;30%</span>
        <span>Watch 30-50%</span>
        <span>Mod 50-70%</span>
        <span>High 70-85%</span>
        <span className="text-rose-400">Crit &gt;85%</span>
      </div>
    </div>
  );
}
