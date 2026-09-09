import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';

export default function TrainingChart({
  history = [],
  height = 260,
}) {
  if (!history || history.length === 0) {
    return (
      <div className="h-60 flex flex-col items-center justify-center text-slate-500 font-mono text-xs border border-dashed border-space-800 rounded-lg">
        <span>No training history available.</span>
        <span className="text-[11px] text-slate-600 mt-1">Configure hyperparameters and click "START TRAINING".</span>
      </div>
    );
  }

  const chartData = history.map((item, idx) => ({
    epoch: item.epoch ?? idx + 1,
    train_loss: Number(item.train_loss || 0),
    val_loss: Number(item.val_loss || 0),
    tss: item.val_tss_mean ? Number(item.val_tss_mean) : null,
    hss: item.val_hss_mean ? Number(item.val_hss_mean) : null,
    duration: item.duration_sec,
  }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const d = payload[0].payload;
      return (
        <div className="bg-space-900 border border-space-700 p-3 rounded-lg shadow-xl text-xs font-mono">
          <div className="text-slate-300 font-bold border-b border-space-800 pb-1 mb-2">
            Epoch {label} {d.duration ? `(${d.duration}s)` : ''}
          </div>
          <div className="space-y-1">
            <div className="text-solar-400 flex justify-between gap-4">
              <span>Train Loss:</span>
              <span className="font-bold">{d.train_loss.toFixed(4)}</span>
            </div>
            <div className="text-sky-400 flex justify-between gap-4">
              <span>Val Loss:</span>
              <span className="font-bold">{d.val_loss.toFixed(4)}</span>
            </div>
            {d.tss !== null && (
              <div className="text-emerald-400 flex justify-between gap-4">
                <span>Val TSS:</span>
                <span className="font-bold">{d.tss.toFixed(3)}</span>
              </div>
            )}
            {d.hss !== null && (
              <div className="text-plasma-cyan flex justify-between gap-4">
                <span>Val HSS:</span>
                <span className="font-bold">{d.hss.toFixed(3)}</span>
              </div>
            )}
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
          <LineChart data={chartData} margin={{ top: 10, right: 15, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.5} />
            <XAxis
              dataKey="epoch"
              stroke="#64748b"
              tick={{ fontSize: 10, fill: '#64748b', fontFamily: 'JetBrains Mono' }}
              label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 10 }}
            />
            <YAxis
              stroke="#64748b"
              tick={{ fontSize: 10, fill: '#64748b', fontFamily: 'JetBrains Mono' }}
              domain={['auto', 'auto']}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              verticalAlign="top"
              wrapperStyle={{ paddingBottom: '8px', fontSize: '11px', fontFamily: 'JetBrains Mono' }}
            />
            <Line
              type="monotone"
              dataKey="train_loss"
              name="Train Loss"
              stroke="#f59e0b"
              strokeWidth={2}
              dot={{ r: 2 }}
            />
            <Line
              type="monotone"
              dataKey="val_loss"
              name="Val Loss"
              stroke="#0ea5e9"
              strokeWidth={2}
              dot={{ r: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
