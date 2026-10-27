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

export default function ForecastChart({ forecast = {}, height = 200 }) {
  const horizons = ['1h', '3h', '6h', '12h', '24h'];
  const available = horizons.some((h) => forecast && forecast[h] !== null && forecast[h] !== undefined && Number.isFinite(Number(forecast[h])));

  if (!available) return <div className="w-full flex items-center justify-center text-sm text-slate-400" style={{ height }}>Forecast unavailable until a compatible trained model produces a prediction.</div>;

  const chartData = horizons.map((h) => {
    const score = forecast && forecast[h] !== null && forecast[h] !== undefined ? Number(forecast[h]) : null;

    let color = '#10b981'; // LOW
    let risk = 'LOW';
    if (score >= 0.85) {
      color = '#f43f5e';
      risk = 'CRITICAL';
    } else if (score >= 0.70) {
      color = '#f97316';
      risk = 'HIGH';
    } else if (score >= 0.50) {
      color = '#f59e0b';
      risk = 'MODERATE';
    } else if (score >= 0.30) {
      color = '#38bdf8';
      risk = 'WATCH';
    }

    return {
      horizon: `${h.toUpperCase()}`,
      score,
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
            <span className="text-slate-300">Uncalibrated score:</span>
            <span className="font-bold" style={{ color: p.color }}>{p.score.toFixed(3)}</span>
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
          Target: M/X-class onset
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
              domain={[0, 1]}
              tickFormatter={(v) => Number(v).toFixed(1)}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={50} stroke="#f59e0b" strokeDasharray="3 3" />
            <ReferenceLine y={85} stroke="#f43f5e" strokeDasharray="3 3" />
            <Bar dataKey="score" radius={[4, 4, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center justify-between mt-2 pt-2 border-t border-space-800 text-[11px] font-mono text-slate-400">
        <span>Low &lt;0.30</span>
        <span>Watch 0.30-0.50</span>
        <span>Mod 0.50-0.70</span>
        <span>High 0.70-0.85</span>
        <span className="text-rose-400">Crit &gt;0.85</span>
      </div>
    </div>
  );
}
