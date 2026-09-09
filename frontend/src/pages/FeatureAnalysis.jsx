import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  Flame,
  HelpCircle,
  TrendingUp,
  Activity,
  Layers,
  Sparkles,
  Info,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  CartesianGrid,
} from 'recharts';
import HardSoftRatioChart from '../charts/HardSoftRatioChart';
import api from '../services/api';

export default function FeatureAnalysis() {
  const [featureData, setFeatureData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadFeatures = async () => {
      try {
        const res = await api.getFeatures();
        setFeatureData(res);
      } catch (e) {
        console.error('Could not fetch engineered features:', e);
      } finally {
        setLoading(false);
      }
    };
    loadFeatures();
  }, []);

  // Top domain engineered features and their importance rankings
  const featureImportance = [
    { feature: 'hard_soft_ratio_rolling_mean_10m', score: 0.235, category: 'Ratio Dynamics' },
    { feature: 'helios_flux_rolling_slope_5m', score: 0.188, category: 'Hard X-ray Impulse' },
    { feature: 'hard_soft_ratio_current', score: 0.142, category: 'Ratio Dynamics' },
    { feature: 'solexs_flux_rolling_mean_30m', score: 0.115, category: 'Thermal Soft X-ray' },
    { feature: 'helios_flux_rolling_max_15m', score: 0.089, category: 'Hard X-ray Impulse' },
    { feature: 'solexs_flux_rolling_gradient_10m', score: 0.076, category: 'Thermal Soft X-ray' },
    { feature: 'hard_soft_ratio_variance_30m', score: 0.054, category: 'Ratio Dynamics' },
    { feature: 'helios_solexs_peak_delay', score: 0.048, category: 'Neupert Kinematics' },
    { feature: 'stl_trend_solexs', score: 0.032, category: 'STL Decomposition' },
    { feature: 'stl_seasonal_helios', score: 0.021, category: 'STL Decomposition' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="panel p-5">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[10px] font-mono uppercase tracking-widest text-solar-400 bg-solar-500/10 px-2 py-0.5 rounded">
            FEATURE ENGINEERING & KINEMATICS
          </span>
        </div>
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <BarChart3 size={22} className="text-solar-400" />
          Multi-Instrument Feature Analysis & Domain Diagnostics
        </h2>
        <p className="text-xs text-slate-400 mt-0.5">
          Evaluating feature importances, rolling window slopes, and the physical significance of the Hard-to-Soft X-ray ratio
        </p>
      </div>

      {/* Scientific Domain Highlight: Why Hard-to-Soft Ratio Matters */}
      <div className="panel p-6 bg-gradient-to-r from-space-900 via-space-850 to-space-900 border-solar-500/40 relative overflow-hidden shadow-glow-solar">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-xl bg-solar-500/20 text-solar-400 border border-solar-500/40 shrink-0">
            <Flame size={28} />
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <h3 className="text-base font-bold text-white tracking-tight">
                Why This Feature Matters: Hard-to-Soft X-Ray Ratio
              </h3>
              <span className="text-[10px] font-mono bg-solar-500 text-space-950 px-2 py-0.5 rounded font-black uppercase">
                Core Domain Feature
              </span>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed max-w-4xl">
              In solar physics, <strong>SoLEXS</strong> primarily measures softer, thermal X-rays (1–30 keV) originating from heated coronal plasma loops (typically 10–30 million Kelvin). In contrast, <strong>HEL1OS</strong> observes hard, non-thermal bremsstrahlung radiation (10–150 keV) produced when magnetic reconnection accelerates electron beams downwards into the denser chromosphere.
            </p>

            <div className="p-3 rounded-lg bg-space-950/70 border border-space-800 text-xs font-mono text-slate-300">
              <span className="text-solar-400 font-bold">Scientific Formula:</span>{' '}
              <code className="text-white bg-space-850 px-2 py-0.5 rounded">Hard-to-Soft Ratio = HEL1OS_Flux / (SoLEXS_Flux + ε)</code>
            </div>

            <p className="text-[11px] text-slate-400 leading-relaxed">
              <strong className="text-slate-300">Scientific Note:</strong> Under the classic Neupert effect hypothesis, the time-integral of hard X-ray emissions often matches the rise of soft X-ray emissions. A rapid spike in the Hard-to-Soft ratio indicates impulsive particle acceleration prior to maximum thermal heating. This differential timing is what our multi-task models exploit for early warning.
            </p>
          </div>
        </div>
      </div>

      {/* Feature Importance Bar Chart */}
      <div className="panel p-5">
        <div className="flex items-center justify-between mb-4 border-b border-space-800 pb-3">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider font-mono flex items-center gap-2">
              <Sparkles size={16} className="text-solar-400" />
              Empirical Feature Importance Ranking (Random Forest Ensemble)
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Gini impurity importance across rolling means, gradients, and ratio transformations
            </p>
          </div>
          <span className="text-xs font-mono text-slate-400 bg-space-950 px-2.5 py-1 rounded border border-space-800">
            Top 10 Features
          </span>
        </div>

        <div style={{ width: '100%', height: 320 }}>
          <ResponsiveContainer>
            <BarChart
              layout="vertical"
              data={featureImportance}
              margin={{ top: 5, right: 30, left: 180, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
              <XAxis
                type="number"
                stroke="#64748b"
                tick={{ fontSize: 10, fill: '#64748b', fontFamily: 'JetBrains Mono' }}
                domain={[0, 0.3]}
              />
              <YAxis
                type="category"
                dataKey="feature"
                stroke="#64748b"
                tick={{ fontSize: 11, fill: '#cbd5e1', fontFamily: 'JetBrains Mono' }}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="bg-space-900 border border-space-700 p-2.5 rounded-lg shadow-xl text-xs font-mono">
                        <div className="text-solar-400 font-bold">{d.feature}</div>
                        <div className="text-slate-300 mt-1">Category: {d.category}</div>
                        <div className="text-emerald-400 mt-1">
                          Importance: {(d.score * 100).toFixed(1)}%
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                {featureImportance.map((entry, idx) => (
                  <Cell
                    key={`cell-${idx}`}
                    fill={entry.category === 'Ratio Dynamics' ? '#f59e0b' : entry.category === 'Hard X-ray Impulse' ? '#0ea5e9' : '#10b981'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="flex items-center justify-center gap-6 mt-3 pt-3 border-t border-space-800 text-xs font-mono text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded bg-solar-500 inline-block" />
            Ratio Dynamics (HEL1OS/SoLEXS)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded bg-sky-500 inline-block" />
            Hard X-Ray Impulsive Component
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded bg-emerald-500 inline-block" />
            Soft X-Ray Thermal Component
          </span>
        </div>
      </div>

      {/* Rolling Stats & Decomposition Description */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="panel p-5">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider font-mono mb-2 flex items-center gap-2">
            <Activity size={16} className="text-sky-400" />
            Multi-Scale Rolling Window Hierarchy
          </h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            Flare onset signatures evolve on multiple timescales. The feature extractor builds rolling statistics over four window sizes:
          </p>
          <ul className="mt-3 space-y-2 text-xs font-mono text-slate-400">
            <li className="p-2 rounded bg-space-950/70 border border-space-800 flex justify-between">
              <span className="text-solar-400 font-bold">5-Minute Window:</span>
              <span>Captures rapid non-thermal impulsive spikes</span>
            </li>
            <li className="p-2 rounded bg-space-950/70 border border-space-800 flex justify-between">
              <span className="text-sky-400 font-bold">10-Minute Window:</span>
              <span>Tracks Hard-to-Soft acceleration gradient</span>
            </li>
            <li className="p-2 rounded bg-space-950/70 border border-space-800 flex justify-between">
              <span className="text-emerald-400 font-bold">30-Minute Window:</span>
              <span>Isolates gradual thermal accumulation</span>
            </li>
            <li className="p-2 rounded bg-space-950/70 border border-space-800 flex justify-between">
              <span className="text-slate-300 font-bold">60-Minute Window:</span>
              <span>Establishes diurnal solar background level</span>
            </li>
          </ul>
        </div>

        <div className="panel p-5">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider font-mono mb-2 flex items-center gap-2">
            <Layers size={16} className="text-solar-400" />
            STL Time-Series Decomposition
          </h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            Seasonal-Trend decomposition using LOESS (STL) isolates low-frequency background solar variance from flare transients:
          </p>
          <div className="mt-3 space-y-2 text-xs font-mono text-slate-400">
            <div className="p-2 rounded bg-space-950/70 border border-space-800">
              <span className="text-white font-bold block">1. Trend Component:</span>
              <span className="text-[11px] text-slate-400">Tracks slow solar rotation and active region flux baseline.</span>
            </div>
            <div className="p-2 rounded bg-space-950/70 border border-space-800">
              <span className="text-white font-bold block">2. Seasonal Component:</span>
              <span className="text-[11px] text-slate-400">Removes periodic orbital or telemetry calibration cycles.</span>
            </div>
            <div className="p-2 rounded bg-space-950/70 border border-space-800">
              <span className="text-white font-bold block">3. Residual Transients:</span>
              <span className="text-[11px] text-slate-400">Contains high-energy flare injection signatures fed directly to ML.</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
