import React, { useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceArea,
  ReferenceLine,
} from 'recharts';
import { Activity, Eye, EyeOff } from 'lucide-react';

export default function XRayChart({
  data = [],
  height = 320,
  showAnnotations = true,
  syncId = 'xray-monitor',
}) {
  const [useLogScale, setUseLogScale] = useState(true);
  const [showSolexs, setShowSolexs] = useState(true);
  const [showHelios, setShowHelios] = useState(true);

  // Format data for display: calculate log10 if selected
  const chartData = data.map((d, i) => {
    const sFlux = Math.max(1e-9, Number(d.solexs_flux || 1e-8));
    const hFlux = Math.max(1e-9, Number(d.helios_flux || 1e-9));

    return {
      time: d.timestamp ? new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : `#${i}`,
      rawTimestamp: d.timestamp,
      solexs: useLogScale ? Math.log10(sFlux) : sFlux,
      helios: useLogScale ? Math.log10(hFlux) : hFlux,
      rawSolexs: sFlux,
      rawHelios: hFlux,
      state: d.state || (d.nowcast ? d.nowcast.state : 'quiet'),
      flare_class: d.flare_class || '',
    };
  });

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const p = payload[0].payload;
      return (
        <div className="bg-space-900 border border-space-700 p-3 rounded-lg shadow-xl text-xs font-mono">
          <div className="text-slate-400 text-[11px] mb-1.5 border-b border-space-800 pb-1">
            Timestamp: <span className="text-white font-medium">{p.time}</span>
            {p.state && (
              <span className="ml-2 px-1.5 py-0.5 rounded bg-space-800 text-solar-400 uppercase text-[10px]">
                {p.state}
              </span>
            )}
          </div>
          <div className="space-y-1">
            <div className="text-solar-400 flex items-center justify-between gap-4">
              <span>SoLEXS (1-30 keV):</span>
              <span className="font-bold">{p.rawSolexs.toExponential(3)}</span>
            </div>
            <div className="text-sky-400 flex items-center justify-between gap-4">
              <span>HEL1OS (10-150 keV):</span>
              <span className="font-bold">{p.rawHelios.toExponential(3)}</span>
            </div>
          </div>
          {p.flare_class && p.flare_class !== 'none' && (
            <div className="mt-1.5 pt-1 border-t border-space-800 text-amber-400 text-[10px]">
              Active Flare: {p.flare_class}
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-3 text-xs">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowSolexs(!showSolexs)}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded font-mono transition-all ${
              showSolexs ? 'bg-solar-500/20 text-solar-300 border border-solar-500/40' : 'bg-space-800 text-slate-500 border border-space-700'
            }`}
          >
            {showSolexs ? <Eye size={12} /> : <EyeOff size={12} />}
            SoLEXS (1-30 keV Soft)
          </button>

          <button
            onClick={() => setShowHelios(!showHelios)}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded font-mono transition-all ${
              showHelios ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40' : 'bg-space-800 text-slate-500 border border-space-700'
            }`}
          >
            {showHelios ? <Eye size={12} /> : <EyeOff size={12} />}
            HEL1OS (10-150 keV Hard)
          </button>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-slate-400 font-mono text-[11px]">Scale:</span>
          <button
            onClick={() => setUseLogScale(!useLogScale)}
            className="px-2 py-0.5 rounded text-[11px] font-mono bg-space-800 hover:bg-space-700 text-slate-300 border border-space-700"
          >
            {useLogScale ? 'Log10 [W/m²]' : 'Linear [W/m²]'}
          </button>
        </div>
      </div>

      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <LineChart data={chartData} syncId={syncId} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
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
              domain={useLogScale ? [-9, -2] : ['auto', 'auto']}
              label={{
                value: useLogScale ? 'log₁₀(Flux)' : 'Flux',
                angle: -90,
                position: 'insideLeft',
                fill: '#64748b',
                fontSize: 10,
                fontFamily: 'JetBrains Mono',
              }}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Threshold Reference Line for M-Class Flare (~1e-5 W/m^2 = -5 in log) */}
            <ReferenceLine
              y={useLogScale ? -5 : 1e-5}
              stroke="#ef4444"
              strokeDasharray="4 4"
              label={{
                value: 'M-Class Threshold (10⁻⁵)',
                position: 'insideTopRight',
                fill: '#ef4444',
                fontSize: 10,
                fontFamily: 'JetBrains Mono',
              }}
            />

            {showSolexs && (
              <Line
                type="monotone"
                dataKey="solexs"
                name="SoLEXS (Soft X-Ray)"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            )}

            {showHelios && (
              <Line
                type="monotone"
                dataKey="helios"
                name="HEL1OS (Hard X-Ray)"
                stroke="#0ea5e9"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
