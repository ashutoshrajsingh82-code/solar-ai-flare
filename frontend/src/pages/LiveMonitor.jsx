import React, { useState, useEffect } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  Gauge,
  Activity,
  Radio,
  Clock,
  Zap,
  ShieldAlert,
  Sliders,
  CheckCircle2,
} from 'lucide-react';
import StatusIndicator from '../components/StatusIndicator';
import RiskBadge from '../components/RiskBadge';
import DataQualityCard from '../components/DataQualityCard';
import XRayChart from '../charts/XRayChart';
import HardSoftRatioChart from '../charts/HardSoftRatioChart';
import ForecastChart from '../charts/ForecastChart';
import api from '../services/api';

export default function LiveMonitor() {
  const [simState, setSimState] = useState(null);
  const [speed, setSpeed] = useState(5);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState('');

  const fetchState = async () => {
    try {
      const state = await api.getSimulationState();
      setSimState(state);
      if (state.speed) setSpeed(state.speed);
    } catch (err) {
      console.error('Failed to get simulation state:', err);
    }
  };

  useEffect(() => {
    fetchState();
    const interval = setInterval(fetchState, 1500);
    return () => clearInterval(interval);
  }, []);

  const handleStart = async () => {
    setLoading(true);
    setStatusMsg('');
    try {
      await api.startSimulation({ speed, sequence_length: 30 });
      setStatusMsg('Telemetry simulation stream active.');
      fetchState();
    } catch (e) {
      setStatusMsg(`Error starting simulation: ${e.response?.data?.detail || e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handlePause = async () => {
    try {
      await api.pauseSimulation();
      fetchState();
    } catch (e) {
      console.error(e);
    }
  };

  const handleResume = async () => {
    try {
      await api.resumeSimulation();
      fetchState();
    } catch (e) {
      console.error(e);
    }
  };

  const handleReset = async () => {
    try {
      await api.resetSimulation();
      fetchState();
    } catch (e) {
      console.error(e);
    }
  };

  const handleSpeedChange = async (newSpeed) => {
    setSpeed(newSpeed);
    try {
      await api.setSimulationSpeed(newSpeed);
      fetchState();
    } catch (e) {
      console.error(e);
    }
  };

  const isRunning = simState?.running || false;
  const isPaused = simState?.paused || false;
  const latest = simState?.latest || {};
  const history = simState?.recent_history || [];

  const nowcastState = latest.nowcast?.state || 'quiet';
  const nowcastConf = latest.nowcast?.confidence;
  const riskLevel = latest.risk || 'LOW';
  const forecastHorizons = latest.forecast || {
    '1h': 0.15,
    '3h': 0.12,
    '6h': 0.08,
    '12h': 0.04,
    '24h': 0.02,
  };

  return (
    <div className="space-y-6">
      {/* Top Mission Control Ribbon */}
      <div className="panel p-4 bg-space-900 border-space-750 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className={`p-2.5 rounded-xl ${isRunning ? 'bg-emerald-500/20 text-emerald-400 animate-pulse' : 'bg-space-800 text-slate-400'}`}>
            <Radio size={20} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-bold text-white tracking-tight">
                ADITYA-L1 MISSION CONTROL TELEMETRY CONSOLE
              </h2>
              <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full font-bold uppercase ${
                isRunning ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40' : 'bg-slate-800 text-slate-400'
              }`}>
                {isRunning ? 'STREAM ACTIVE' : isPaused ? 'PAUSED' : 'STANDBY'}
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono">
              Live sequential playback of SoLEXS + HEL1OS observations via Aditya-L1 data bus
            </p>
          </div>
        </div>

        {/* Simulator Controls */}
        <div className="flex items-center gap-2 font-mono text-xs">
          {!isRunning && !isPaused ? (
            <button
              onClick={handleStart}
              disabled={loading}
              className="px-4 py-2 rounded-lg bg-solar-500 hover:bg-solar-400 text-space-950 font-bold flex items-center gap-1.5 transition-all shadow-glow-solar"
            >
              <Play size={14} fill="currentColor" />
              START LIVE SIMULATION
            </button>
          ) : isRunning ? (
            <button
              onClick={handlePause}
              className="px-3.5 py-2 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 font-semibold flex items-center gap-1.5 transition-all"
            >
              <Pause size={14} fill="currentColor" />
              Pause
            </button>
          ) : (
            <button
              onClick={handleResume}
              className="px-3.5 py-2 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/40 font-semibold flex items-center gap-1.5 transition-all"
            >
              <Play size={14} fill="currentColor" />
              Resume
            </button>
          )}

          <button
            onClick={handleReset}
            className="px-3 py-2 rounded-lg bg-space-800 hover:bg-space-700 text-slate-300 border border-space-700 flex items-center gap-1.5 transition-all"
            title="Reset to beginning of stream"
          >
            <RotateCcw size={14} />
            Restart
          </button>

          {/* Speed Selector */}
          <div className="flex items-center gap-1 ml-2 bg-space-950 p-1 rounded-lg border border-space-800">
            <span className="text-[10px] text-slate-400 px-1.5 flex items-center gap-1">
              <Gauge size={12} />
              Rate:
            </span>
            {[1, 5, 10, 50].map((s) => (
              <button
                key={s}
                onClick={() => handleSpeedChange(s)}
                className={`px-2 py-0.5 rounded text-[11px] font-bold transition-all ${
                  speed === s
                    ? 'bg-solar-500 text-space-950 shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {s}×
              </button>
            ))}
          </div>
        </div>
      </div>

      {statusMsg && (
        <div className="bg-solar-500/10 border border-solar-500/30 p-2.5 rounded-lg text-xs font-mono text-solar-300">
          {statusMsg}
        </div>
      )}

      {/* Live Inference Telemetry Bar */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 font-mono">
        <div className="panel p-3 bg-space-900 border-space-750">
          <span className="text-[10px] text-slate-400 block uppercase">Timestamp (UTC)</span>
          <span className="text-xs font-bold text-white block mt-0.5 truncate">
            {latest.timestamp ? new Date(latest.timestamp).toUTCString().slice(17, 25) : '--:--:--'}
          </span>
        </div>

        <div className="panel p-3 bg-space-900 border-space-750">
          <span className="text-[10px] text-solar-400 block uppercase">SoLEXS (1-30 keV)</span>
          <span className="text-sm font-bold text-solar-300 block mt-0.5">
            {latest.input?.solexs_flux !== undefined ? Number(latest.input.solexs_flux).toExponential(3) : '--'}
          </span>
        </div>

        <div className="panel p-3 bg-space-900 border-space-750">
          <span className="text-[10px] text-sky-400 block uppercase">HEL1OS (10-150 keV)</span>
          <span className="text-sm font-bold text-sky-300 block mt-0.5">
            {latest.input?.helios_flux !== undefined ? Number(latest.input.helios_flux).toExponential(3) : '--'}
          </span>
        </div>

        <div className="panel p-3 bg-space-900 border-space-750">
          <span className="text-[10px] text-plasma-cyan block uppercase">Hard/Soft Ratio</span>
          <span className="text-sm font-bold text-white block mt-0.5">
            {latest.input?.hard_soft_ratio !== undefined ? Number(latest.input.hard_soft_ratio).toFixed(4) : '--'}
          </span>
        </div>

        <div className="panel p-3 bg-space-900 border-space-750 flex flex-col justify-between">
          <span className="text-[10px] text-slate-400 block uppercase">Nowcast / Risk</span>
          <div className="flex items-center gap-1.5 mt-0.5">
            <StatusIndicator state={nowcastState} confidence={nowcastConf} />
            <RiskBadge level={riskLevel} size="sm" />
          </div>
        </div>
      </div>

      {/* Main Dual X-Ray Live Trace */}
      <div className="panel p-5">
        <div className="flex items-center justify-between mb-4 border-b border-space-800 pb-3">
          <div>
            <h3 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
              <Activity size={18} className="text-solar-400" />
              Live Streaming Dual-Channel X-Ray Flux
            </h3>
            <p className="text-xs text-slate-400 mt-0.5 font-mono">
              Synchronized SoLEXS (thermal soft) & HEL1OS (impulsive hard) sliding sequence buffer
            </p>
          </div>
          <div className="flex items-center gap-2 font-mono text-xs">
            <span className="text-slate-400">Step:</span>
            <span className="text-solar-400 font-bold">{simState?.current_index || 0}</span>
            <span className="text-slate-500">/ {simState?.total_rows || 0}</span>
          </div>
        </div>

        <XRayChart data={history} height={320} />
      </div>

      {/* Hard/Soft Ratio & Forecast Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 panel p-5">
          <HardSoftRatioChart data={history} height={220} />
        </div>

        <div className="panel p-5">
          <ForecastChart forecast={forecastHorizons} height={220} />
        </div>
      </div>

      {/* Telemetry & Quality Summary */}
      <DataQualityCard
        qualityScore={latest.data_quality || 0.98}
        cadence="60s"
        totalRows={simState?.total_rows || 14400}
        missingCount={12}
      />
    </div>
  );
}
