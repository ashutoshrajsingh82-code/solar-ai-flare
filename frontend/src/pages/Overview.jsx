import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Sun,
  ShieldAlert,
  Flame,
  Cpu,
  Radio,
  Clock,
  ArrowRight,
  TrendingUp,
  Workflow,
  AlertTriangle,
  Play,
  RotateCcw,
} from 'lucide-react';
import MetricCard from '../components/MetricCard';
import StatusIndicator from '../components/StatusIndicator';
import RiskBadge from '../components/RiskBadge';
import AlertPanel from '../components/AlertPanel';
import XRayChart from '../charts/XRayChart';
import HardSoftRatioChart from '../charts/HardSoftRatioChart';
import ForecastChart from '../charts/ForecastChart';
import api from '../services/api';

export default function Overview() {
  const [summary, setSummary] = useState(null);
  const [recentTelemetry, setRecentTelemetry] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchDashboardData = async () => {
    try {
      const [sum, sim] = await Promise.all([
        api.getDashboardSummary(),
        api.getSimulationState(),
      ]);
      setSummary(sum);
      if (sim && sim.recent_history && sim.recent_history.length > 0) {
        setRecentTelemetry(sim.recent_history);
      } else {
        // Fallback to data preview if simulator hasn't started
        const prev = await api.getDataPreview(60);
        if (prev && prev.rows) {
          setRecentTelemetry(prev.rows);
        }
      }
    } catch (err) {
      console.error('Failed to load dashboard overview:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    const timer = setInterval(fetchDashboardData, 3000);
    return () => clearInterval(timer);
  }, []);

  const handleAcknowledgeAlert = async (id) => {
    try {
      await api.acknowledgeAlert(id, true);
      fetchDashboardData();
    } catch (e) {
      console.error(e);
    }
  };

  const currentState = summary?.current_state || 'pre_flare';
  const riskLevel = summary?.risk_level || 'HIGH';
  const forecast1h = summary?.forecast_1h !== null && summary?.forecast_1h !== undefined
    ? `${(summary.forecast_1h * 100).toFixed(0)}%`
    : '78%';
  const activeModel = summary?.active_model || 'CNN-LSTM Fusion (Dual-Channel)';
  const systemStatus = summary?.system_status || 'MONITORING';
  const latestInference = summary?.latest_inference || {};
  const forecastHorizons = latestInference.forecast || {
    '1h': 0.78,
    '3h': 0.71,
    '6h': 0.59,
    '12h': 0.41,
    '24h': 0.30,
  };

  return (
    <div className="space-y-6">
      {/* Hero Header Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-space-900 via-space-850 to-space-900 border border-space-750 p-6 shadow-2xl">
        <div className="absolute top-0 right-0 w-96 h-96 bg-solar-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 relative z-10">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-[11px] font-mono uppercase tracking-widest font-black text-solar-400 bg-solar-500/10 border border-solar-500/30 px-2.5 py-0.5 rounded-full">
                ADITYA-L1 SOLAR FLARE INTELLIGENCE
              </span>
              <span className="text-[11px] font-mono text-slate-400">
                SoLEXS • HEL1OS • ML/DL Fusion
              </span>
            </div>
            <h2 className="text-2xl lg:text-3xl font-extrabold text-white tracking-tight">
              Dual-Instrument X-Ray Early Warning System
            </h2>
            <p className="text-sm text-slate-300 mt-1 max-w-2xl leading-relaxed">
              Real-time soft (1–30 keV) & hard (10–150 keV) X-ray spectroscopy fusion from ISRO Aditya-L1 for sub-hour solar flare forecasting and rapid precursor detection.
            </p>
          </div>

          <div className="flex items-center gap-3 bg-space-950/80 p-3 rounded-xl border border-space-750 font-mono shrink-0">
            <div>
              <span className="text-[10px] text-slate-400 block uppercase">1-Hour Flare Risk</span>
              <span className="text-2xl font-bold text-solar-400">{forecast1h}</span>
            </div>
            <div className="h-8 w-px bg-space-800" />
            <div>
              <span className="text-[10px] text-slate-400 block uppercase">Solar State</span>
              <StatusIndicator state={currentState} />
            </div>
            <div className="h-8 w-px bg-space-800" />
            <Link
              to="/live"
              className="px-3.5 py-2 rounded-lg bg-solar-500 hover:bg-solar-400 text-space-950 font-bold text-xs flex items-center gap-1.5 transition-all shadow-glow-solar"
            >
              <Radio size={14} className="animate-pulse" />
              Live Monitor
            </Link>
          </div>
        </div>
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <MetricCard
          title="Current Solar State"
          value={<StatusIndicator state={currentState} confidence={latestInference?.nowcast?.confidence} />}
          subtitle="Real-time Nowcast condition"
          icon={Sun}
          accent="solar"
        />

        <MetricCard
          title="M/X Flare Risk"
          value={<RiskBadge level={riskLevel} size="lg" />}
          subtitle="Derived from forecast probabilities"
          icon={ShieldAlert}
          accent={riskLevel === 'CRITICAL' || riskLevel === 'HIGH' ? 'danger' : 'solar'}
        />

        <MetricCard
          title="1-Hour Probability"
          value={forecast1h}
          unit="P(M/X)"
          subtitle="Onset within next 60 minutes"
          icon={Clock}
          accent="solar"
        />

        <MetricCard
          title="Active Model"
          value={activeModel.split(' ')[0]}
          unit={activeModel.includes('Fusion') ? 'Fusion' : ''}
          subtitle="Dual-head neural classifier"
          icon={Cpu}
          accent="helios"
        />

        <MetricCard
          title="Telemetry Status"
          value={systemStatus}
          subtitle="Continuous Aditya-L1 sync"
          icon={Radio}
          accent={systemStatus === 'MONITORING' ? 'emerald' : 'slate'}
        />
      </div>

      {/* Main Real-Time X-Ray Dual Channel Monitor */}
      <div className="panel p-5">
        <div className="flex items-center justify-between mb-4 border-b border-space-800 pb-3">
          <div>
            <h3 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
              <TrendingUp size={18} className="text-solar-400" />
              Real-Time Dual-Channel X-Ray Monitor
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              SoLEXS (1–30 keV Soft Thermal) & HEL1OS (10–150 keV Hard Non-thermal) aligned telemetry
            </p>
          </div>
          <span className="text-xs font-mono text-slate-400 bg-space-950 px-2.5 py-1 rounded border border-space-800">
            Cadence: 60s
          </span>
        </div>

        <XRayChart data={recentTelemetry} height={340} />
      </div>

      {/* Domain Feature & Forecast Horizons Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Hard / Soft Ratio Diagnostic */}
        <div className="lg:col-span-2 panel p-5">
          <HardSoftRatioChart data={recentTelemetry} height={220} />
          <div className="mt-3 bg-space-950/60 p-3 rounded-lg border border-space-800 text-xs text-slate-300 flex items-start gap-2">
            <Flame size={16} className="text-solar-400 shrink-0 mt-0.5" />
            <p className="leading-relaxed">
              <strong className="text-solar-300">Neupert Effect Early Warning:</strong> Hard X-rays from HEL1OS trace non-thermal electron beam acceleration in the solar chromosphere, which routinely precedes thermal plasma accumulation observed in SoLEXS soft X-rays. A rising Hard/Soft ratio indicates impending thermal flare eruption.
            </p>
          </div>
        </div>

        {/* Forecast Horizons Probabilities */}
        <div className="panel p-5">
          <ForecastChart forecast={forecastHorizons} height={220} />
        </div>
      </div>

      {/* Alerts Feed & Pipeline Health Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Alerts Feed */}
        <div className="lg:col-span-2 panel p-5">
          <div className="flex items-center justify-between mb-4 border-b border-space-800 pb-3">
            <div className="flex items-center gap-2">
              <ShieldAlert size={18} className="text-rose-400" />
              <h3 className="text-sm font-bold text-white tracking-tight uppercase">
                Active Solar Flare Warnings
              </h3>
            </div>
            <Link to="/alerts" className="text-xs font-mono text-solar-400 hover:text-solar-300 flex items-center gap-1">
              View All Alerts <ArrowRight size={12} />
            </Link>
          </div>

          <AlertPanel
            alerts={summary?.recent_alerts || []}
            onAcknowledge={handleAcknowledgeAlert}
            maxItems={4}
          />
        </div>

        {/* Pipeline & Scientific Health */}
        <div className="panel p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 border-b border-space-800 pb-3 mb-4">
              <Workflow size={18} className="text-sky-400" />
              <h3 className="text-sm font-bold text-white tracking-tight uppercase">
                Pipeline Health
              </h3>
            </div>

            <div className="space-y-3 text-xs font-mono">
              <div className="flex justify-between items-center p-2.5 bg-space-950/60 rounded border border-space-800">
                <span className="text-slate-400">Stages Complete</span>
                <span className="text-emerald-400 font-bold">
                  {summary?.pipeline_health?.stages_complete || 6} / {summary?.pipeline_health?.total_stages || 6}
                </span>
              </div>
              <div className="flex justify-between items-center p-2.5 bg-space-950/60 rounded border border-space-800">
                <span className="text-slate-400">Training Readiness</span>
                <span className="text-solar-400 font-bold">READY</span>
              </div>
              <div className="flex justify-between items-center p-2.5 bg-space-950/60 rounded border border-space-800">
                <span className="text-slate-400">Data Mode</span>
                <span className="text-slate-300 font-bold">SYNTHETIC DEMO</span>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-space-800">
            <Link
              to="/pipeline"
              className="w-full py-2 px-3 rounded-lg bg-space-800 hover:bg-space-700 text-slate-200 text-xs font-mono font-medium flex items-center justify-center gap-2 border border-space-700 transition-colors"
            >
              Inspect Data Pipeline <ArrowRight size={13} />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
