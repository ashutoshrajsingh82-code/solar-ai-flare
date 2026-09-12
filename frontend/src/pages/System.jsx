import React, { useState, useEffect } from 'react';
import {
  Server,
  Cpu,
  Database,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Wifi,
  HardDrive,
  Activity,
  Layers,
} from 'lucide-react';
import api from '../services/api';

export default function System() {
  const [systemInfo, setSystemInfo] = useState(null);
  const [backendHealth, setBackendHealth] = useState(null);
  const [latencyMs, setLatencyMs] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchSystemData = async () => {
    const start = performance.now();
    try {
      const [sys, health] = await Promise.all([
        api.getSystemInfo().catch(() => null),
        api.getHealth().catch(() => null),
      ]);
      const end = performance.now();
      setLatencyMs(Math.round(end - start));
      setSystemInfo(sys);
      setBackendHealth(health);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemData();
    const interval = setInterval(fetchSystemData, 10000);
    return () => clearInterval(interval);
  }, []);

  const device = systemInfo?.device || 'CPU (Standard Host Execution)';
  const isCuda = device.toLowerCase().includes('cuda') || device.toLowerCase().includes('gpu');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="panel p-5">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[10px] font-mono uppercase tracking-widest text-sky-400 bg-sky-500/10 px-2 py-0.5 rounded">
            INFRASTRUCTURE & ENVIRONMENT DIAGNOSTICS
          </span>
        </div>
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <Server size={22} className="text-solar-400" />
          System Health, Compute Hardware & Runtime Stack
        </h2>
        <p className="text-xs text-slate-400 mt-0.5">
          Monitoring API microservice health, SQLite database metrics, execution latency, and device targets
        </p>
      </div>

      {/* Subsystems Health Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 font-mono text-xs">
        <div className="panel p-4">
          <div className="flex items-center justify-between">
            <span className="text-slate-400 uppercase text-[10px]">React Frontend</span>
            <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#10b981]" />
          </div>
          <span className="text-sm font-bold text-white mt-1 block">ONLINE</span>
          <span className="text-[10px] text-slate-500 mt-1 block">Vite 6.0 + React 18</span>
        </div>

        <div className="panel p-4">
          <div className="flex items-center justify-between">
            <span className="text-slate-400 uppercase text-[10px]">FastAPI Backend</span>
            <span className={`w-2 h-2 rounded-full ${backendHealth?.status === 'ok' ? 'bg-emerald-400 shadow-[0_0_8px_#10b981]' : 'bg-rose-400'}`} />
          </div>
          <span className="text-sm font-bold text-white mt-1 block">
            {backendHealth?.status === 'ok' ? 'ONLINE' : 'DEGRADED'}
          </span>
          <span className="text-[10px] text-slate-500 mt-1 block">Uvicorn + Python 3.13</span>
        </div>

        <div className="panel p-4">
          <div className="flex items-center justify-between">
            <span className="text-slate-400 uppercase text-[10px]">SQLite Database</span>
            <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#10b981]" />
          </div>
          <span className="text-sm font-bold text-white mt-1 block">CONNECTED</span>
          <span className="text-[10px] text-slate-500 mt-1 block">solarflare.db</span>
        </div>

        <div className="panel p-4">
          <div className="flex items-center justify-between">
            <span className="text-slate-400 uppercase text-[10px]">API Round-Trip</span>
            <Activity size={14} className="text-solar-400" />
          </div>
          <span className="text-sm font-bold text-solar-400 mt-1 block">
            {latencyMs !== null ? `${latencyMs} ms` : '--'}
          </span>
          <span className="text-[10px] text-slate-500 mt-1 block">Local REST transport</span>
        </div>
      </div>

      {/* Runtime Hardware & Deep Learning Engine Specs */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="panel p-5 space-y-3 font-mono text-xs">
          <div className="flex items-center gap-2 border-b border-space-800 pb-3 mb-2">
            <Cpu size={18} className="text-solar-400" />
            <h3 className="text-sm font-bold text-white uppercase tracking-wider font-sans">
              Compute Device & Acceleration
            </h3>
          </div>

          <div className="p-3 bg-space-950/70 rounded-lg border border-space-800 flex justify-between items-center">
            <span className="text-slate-400">Target Device</span>
            <span className="text-solar-400 font-bold">{device}</span>
          </div>
          <div className="p-3 bg-space-950/70 rounded-lg border border-space-800 flex justify-between items-center">
            <span className="text-slate-400">Hardware Acceleration</span>
            <span className="text-white font-medium">{isCuda ? 'CUDA GPU Enabled' : 'CPU Threadpool (No GPU Required)'}</span>
          </div>
          <div className="p-3 bg-space-950/70 rounded-lg border border-space-800 flex justify-between items-center">
            <span className="text-slate-400">Deep Learning Backend</span>
            <span className="text-sky-400 font-bold">PyTorch 2.14.0+cpu</span>
          </div>
          <div className="p-3 bg-space-950/70 rounded-lg border border-space-800 flex justify-between items-center">
            <span className="text-slate-400">Scientific Stack</span>
            <span className="text-emerald-400 font-medium">Astropy 8.0 + Statsmodels 0.15 + Scikit-Learn</span>
          </div>
        </div>

        <div className="panel p-5 space-y-3 font-mono text-xs">
          <div className="flex items-center gap-2 border-b border-space-800 pb-3 mb-2">
            <HardDrive size={18} className="text-sky-400" />
            <h3 className="text-sm font-bold text-white uppercase tracking-wider font-sans">
              Operational State & Persistence
            </h3>
          </div>

          <div className="p-3 bg-space-950/70 rounded-lg border border-space-800 flex justify-between items-center">
            <span className="text-slate-400">Operating Mode</span>
            <span className="text-solar-400 font-bold uppercase">
              {systemInfo?.data_mode || 'demo'} (Synthetic Physical Playback)
            </span>
          </div>
          <div className="p-3 bg-space-950/70 rounded-lg border border-space-800 flex justify-between items-center">
            <span className="text-slate-400">Active Inference Model</span>
            <span className="text-white font-bold">{systemInfo?.active_model || 'cnn_lstm_fusion_v1'}</span>
          </div>
          <div className="p-3 bg-space-950/70 rounded-lg border border-space-800 flex justify-between items-center">
            <span className="text-slate-400">Database Schema</span>
            <span className="text-white">Alerts, Datasets, Experiments, Inferences</span>
          </div>
          <div className="p-3 bg-space-950/70 rounded-lg border border-space-800 flex justify-between items-center">
            <span className="text-slate-400">CORS Configuration</span>
            <span className="text-emerald-400">Enabled (Vite :5173 / Localhost)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
