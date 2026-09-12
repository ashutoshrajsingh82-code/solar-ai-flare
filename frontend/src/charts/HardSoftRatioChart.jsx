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
import { Flame, Info } from 'lucide-react';

export default function HardSoftRatioChart({
  data = [],
  height = 180,
  syncId = 'xray-monitor',
}) {
  const chartData = data.map((d, i) => {
    let ratio = d.hard_soft_ratio;
    if (ratio === undefined || ratio === null) {
      const sFlux = Math.max(1e-9, Number(d.solexs_flux || 1e-8));
      const hFlux = Math.max(1e-9, Number(d.helios_flux || 1e-9));
      ratio = hFlux / (sFlux + 1e-12);
    }

    return {
      time: d.timestamp ? new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : `#${i}`,
      ratio: Number(ratio || 0),
      state: d.state || 'quiet',
    };
  });

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const p = payload[0].payload;
      return (
        <div className="bg-space-900 border border-space-700 p-2.5 rounded-lg shadow-xl text-xs font-mono">
          <div className="text-slate-400 text-[10px] mb-1">Time: {p.time}</div>
          <div className="text-plasma-cyan flex items-center justify-between gap-4">
            <span>Hard/Soft Ratio:</span>
            <span className="font-bold text-white">{p.ratio.toFixed(4)}</span>
          </div>
          <div className="text-[10px] text-slate-400 mt-1">
            State: <span className="uppercase text-solar-400">{p.state}</span>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-xs uppercase tracking-wider font-semibold text-slate-300 font-mono flex items-center gap-1.5">
            <Flame size={14} className="text-plasma-cyan" />
            Hard-to-Soft Ratio (HEL1OS / SoLEXS)
          </span>
          <span className="text-[10px] font-mono text-slate-400 bg-space-800 px-1.5 py-0.5 rounded border border-space-700">
            Neupert Diagnostic
          </span>
        </div>
        <span className="text-[11px] font-mono text-slate-400">
          Spike indicates impulsive non-thermal energy injection
        </span>
      </div>

      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <LineChart data={chartData} syncId={syncId} margin={{ top: 5, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.5} />
            <XAxis
              dataKey="time"
              stroke="#64748b"
              tick={{ fontSize: 10, fill: '#64748b', fontFamily: 'JetBrains Mono' }}
              minTickGap={35}
            />
            <YAxis
              stroke="#64748b"
              tick={{ fontSize: 10, fill: '#64748b', fontFamily: 'JetBrains Mono' }}
              domain={['auto', 'auto']}
              label={{
                value: 'Ratio',
                angle: -90,
                position: 'insideLeft',
                fill: '#64748b',
                fontSize: 10,
                fontFamily: 'JetBrains Mono',
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="ratio"
              stroke="#06b6d4"
              strokeWidth={1.8}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
