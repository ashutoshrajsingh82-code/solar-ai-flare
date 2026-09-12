import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';

export default function FusionComparisonChart({
  results = {},
  height = 320,
}) {
  // results expected format:
  // {
  //   solexs_only: { tss, hss, precision, recall, f1, false_alarm_rate, pr_auc },
  //   helios_only: { ... },
  //   fused: { ... }
  // }
  const metricsKeys = [
    { key: 'tss', label: 'TSS' },
    { key: 'hss', label: 'HSS' },
    { key: 'f1', label: 'F1 Score' },
    { key: 'recall', label: 'Recall (POD)' },
    { key: 'precision', label: 'Precision' },
    { key: 'pr_auc', label: 'PR-AUC' },
    { key: 'false_alarm_rate', label: 'FAR (Lower=Better)' },
  ];

  const solexs = results.solexs_only || {};
  const helios = results.helios_only || {};
  const fused = results.fused || {};

  const chartData = metricsKeys.map(({ key, label }) => ({
    metric: label,
    solexs: Number(solexs[key] ?? 0),
    helios: Number(helios[key] ?? 0),
    fused: Number(fused[key] ?? 0),
  }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-space-900 border border-space-700 p-3 rounded-lg shadow-xl text-xs font-mono">
          <div className="text-slate-300 font-bold border-b border-space-800 pb-1 mb-2">
            Metric: {label}
          </div>
          <div className="space-y-1.5">
            <div className="text-solar-400 flex items-center justify-between gap-4">
              <span>SoLEXS Only (Soft):</span>
              <span className="font-bold">{payload[0]?.value?.toFixed(3)}</span>
            </div>
            <div className="text-sky-400 flex items-center justify-between gap-4">
              <span>HEL1OS Only (Hard):</span>
              <span className="font-bold">{payload[1]?.value?.toFixed(3)}</span>
            </div>
            <div className="text-emerald-400 flex items-center justify-between gap-4">
              <span>Fused (SoLEXS + HEL1OS):</span>
              <span className="font-bold text-white bg-emerald-500/20 px-1 rounded">
                {payload[2]?.value?.toFixed(3)}
              </span>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full">
      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <BarChart data={chartData} margin={{ top: 15, right: 15, left: -15, bottom: 25 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.5} />
            <XAxis
              dataKey="metric"
              stroke="#64748b"
              tick={{ fontSize: 10, fill: '#cbd5e1', fontFamily: 'JetBrains Mono' }}
              interval={0}
              angle={-20}
              textAnchor="end"
            />
            <YAxis
              stroke="#64748b"
              tick={{ fontSize: 10, fill: '#64748b', fontFamily: 'JetBrains Mono' }}
              domain={[0, 1.0]}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              verticalAlign="top"
              wrapperStyle={{ paddingBottom: '10px', fontSize: '11px', fontFamily: 'JetBrains Mono' }}
            />
            <Bar dataKey="solexs" name="SoLEXS Only (1-30 keV)" fill="#f59e0b" radius={[3, 3, 0, 0]} />
            <Bar dataKey="helios" name="HEL1OS Only (10-150 keV)" fill="#0ea5e9" radius={[3, 3, 0, 0]} />
            <Bar dataKey="fused" name="SoLEXS + HEL1OS Fused" fill="#10b981" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
