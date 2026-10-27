import React from 'react';
import { AlertTriangle, Bell, Check, CheckCheck, Clock, ShieldAlert } from 'lucide-react';
import RiskBadge from './RiskBadge';

export default function AlertPanel({
  alerts = [],
  onAcknowledge = () => {},
  maxItems = 5,
  compact = false,
}) {
  const displayed = alerts.slice(0, maxItems);

  if (!displayed.length) {
    return (
      <div className="panel p-6 text-center border-space-750">
        <Bell size={24} className="mx-auto text-slate-600 mb-2" />
        <p className="text-xs text-slate-400">No active solar flare alerts logged.</p>
      </div>
    );
  }

  return (
    <div className="space-y-2.5">
      {displayed.map((alert) => {
        const isCritical = alert.severity === 'CRITICAL';
        const isHigh = alert.severity === 'HIGH';

        return (
          <div
            key={alert.id}
            className={`p-3 rounded-lg border transition-all ${
              alert.acknowledged
                ? 'bg-space-950/40 border-space-800 opacity-60'
                : isCritical
                ? 'bg-rose-950/30 border-rose-500/50 shadow-glow-danger'
                : isHigh
                ? 'bg-orange-950/20 border-orange-500/40'
                : 'bg-space-900 border-space-750'
            }`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-start gap-2.5">
                <div className={`p-1.5 rounded-md mt-0.5 ${
                  isCritical ? 'bg-rose-500/20 text-rose-400' : 'bg-solar-500/20 text-solar-400'
                }`}>
                  <ShieldAlert size={16} />
                </div>
                <div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <RiskBadge level={alert.severity} size="sm" />
                    <span className="text-[11px] font-mono text-slate-400 flex items-center gap-1">
                      <Clock size={11} />
                      {new Date(alert.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </span>
                    {alert.forecast_horizon && (
                      <span className="text-[10px] font-mono bg-space-800 text-sky-300 px-1.5 py-0.5 rounded">
                        Window: {alert.forecast_horizon}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-200 mt-1.5 font-medium leading-relaxed">
                    {alert.message}
                  </p>
                  {alert.probability !== null && alert.probability !== undefined && (
                    <div className="mt-1 text-[11px] font-mono text-slate-400">
                      Peak Probability: <span className="text-solar-400 font-bold">{(alert.probability * 100).toFixed(1)}%</span>
                      {alert.model && <span className="ml-2 text-slate-500">({alert.model})</span>}
                    </div>
                  )}
                </div>
              </div>

              {!alert.acknowledged ? (
                <button
                  onClick={() => onAcknowledge(alert.id)}
                  title="Acknowledge Alert"
                  className="px-2 py-1 text-[11px] font-mono bg-space-800 hover:bg-space-700 text-slate-300 hover:text-white border border-space-700 rounded transition-colors flex items-center gap-1 whitespace-nowrap"
                >
                  <Check size={12} />
                  Ack
                </button>
              ) : (
                <span className="text-[10px] font-mono text-emerald-400 flex items-center gap-1">
                  <CheckCheck size={12} />
                  Acked
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
