import React, { useState, useEffect } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Radio,
  Workflow,
  Database,
  BarChart3,
  Cpu,
  Flame,
  CheckCircle,
  GitCompare,
  Bell,
  Server,
  Sun,
  ChevronRight,
  Wifi,
  Activity,
} from 'lucide-react';
import api from '../services/api';

const NAV_ITEMS = [
  { path: '/', label: 'Overview', icon: LayoutDashboard },
  { path: '/live', label: 'Live Monitor', icon: Radio, badge: 'LIVE' },
  { path: '/pipeline', label: 'Data Pipeline', icon: Workflow },
  { path: '/data', label: 'Data Explorer', icon: Database },
  { path: '/features', label: 'Feature Analysis', icon: BarChart3 },
  { path: '/models', label: 'Models', icon: Cpu },
  { path: '/training', label: 'Training', icon: Flame },
  { path: '/evaluation', label: 'Evaluation', icon: CheckCircle },
  { path: '/fusion', label: 'Fusion Experiment', icon: GitCompare, highlight: true },
  { path: '/alerts', label: 'Alerts', icon: Bell },
  { path: '/system', label: 'System', icon: Server },
];

export default function AppLayout() {
  const [backendStatus, setBackendStatus] = useState('checking');
  const [unreadAlerts, setUnreadAlerts] = useState(0);
  const location = useLocation();

  useEffect(() => {
    let isMounted = true;
    const checkHealth = async () => {
      try {
        const res = await api.getHealth();
        if (isMounted) {
          setBackendStatus(res.status === 'ok' ? 'online' : 'degraded');
        }
      } catch (err) {
        if (isMounted) setBackendStatus('offline');
      }
    };

    const checkAlerts = async () => {
      try {
        const alerts = await api.getAlerts();
        if (isMounted) {
          const unacked = alerts.filter((a) => !a.acknowledged).length;
          setUnreadAlerts(unacked);
        }
      } catch (e) {
        // quiet ignore in polling
      }
    };

    checkHealth();
    checkAlerts();
    const interval = setInterval(() => {
      checkHealth();
      checkAlerts();
    }, 8000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="flex h-screen bg-space-950 text-slate-100 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-space-900/95 border-r border-space-750 flex flex-col justify-between z-20 select-none">
        <div>
          {/* Header Branding */}
          <div className="p-4 border-b border-space-750">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-solar-500/20 text-solar-400 border border-solar-500/30 shadow-glow-solar">
                <Sun size={22} className="animate-spin-slow" />
              </div>
              <div>
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] font-mono tracking-widest font-black text-solar-400 uppercase">
                    ADITYA-L1
                  </span>
                  <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-space-800 text-sky-400 border border-sky-500/20">
                    ISRO
                  </span>
                </div>
                <h1 className="text-sm font-extrabold text-white tracking-tight leading-tight">
                  SOLAR FLARE INTELLIGENCE
                </h1>
              </div>
            </div>

            <div className="mt-2.5 flex items-center justify-between text-[11px] font-mono bg-space-950/70 px-2.5 py-1.5 rounded border border-space-800">
              <span className="text-slate-400">System Identifier:</span>
              <span className="text-solar-400 font-semibold tracking-tight">SoLEXS + HEL1OS Fusion</span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="p-2 space-y-0.5 overflow-y-auto max-h-[calc(100vh-170px)]">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) =>
                    `flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                      isActive
                        ? 'bg-solar-500 text-space-950 font-bold shadow-glow-solar'
                        : item.highlight
                        ? 'text-solar-300 hover:bg-space-800/80 hover:text-solar-200 border border-solar-500/20'
                        : 'text-slate-300 hover:bg-space-800 hover:text-white'
                    }`
                  }
                >
                  <div className="flex items-center gap-2.5">
                    <Icon size={16} />
                    <span>{item.label}</span>
                  </div>

                  <div className="flex items-center gap-1.5">
                    {item.badge && (
                      <span className="text-[9px] font-mono px-1.5 py-0.5 rounded-full bg-rose-500 text-white font-bold animate-pulse">
                        {item.badge}
                      </span>
                    )}
                    {item.path === '/alerts' && unreadAlerts > 0 && (
                      <span className="text-[10px] font-mono px-1.5 py-0.2 rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/40 font-bold">
                        {unreadAlerts}
                      </span>
                    )}
                  </div>
                </NavLink>
              );
            })}
          </nav>
        </div>

        {/* Footer Telemetry Status */}
        <div className="p-3 border-t border-space-750 bg-space-950/50">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-slate-400 flex items-center gap-1.5">
              <Wifi size={13} className={backendStatus === 'online' ? 'text-emerald-400' : 'text-rose-400'} />
              Backend API
            </span>
            <span
              className={`text-[10px] font-semibold uppercase px-2 py-0.5 rounded-full ${
                backendStatus === 'online'
                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                  : 'bg-rose-500/10 text-rose-400 border border-rose-500/30'
              }`}
            >
              {backendStatus}
            </span>
          </div>
          <div className="mt-2 text-[10px] font-mono text-slate-500 flex justify-between">
            <span>DATA_MODE=demo</span>
            <span>v1.0.0</span>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        <header className="h-12 border-b border-space-750/70 bg-space-900/60 backdrop-blur-md px-6 flex items-center justify-between text-xs font-mono shrink-0">
          <div className="flex items-center gap-2 text-slate-400">
            <span>Aditya-L1 Mission Payload</span>
            <ChevronRight size={12} />
            <span className="text-solar-400 font-semibold uppercase">
              {NAV_ITEMS.find((n) => n.path === location.pathname)?.label || 'Monitoring'}
            </span>
          </div>

          <div className="flex items-center gap-4 text-[11px]">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-solar-400 shadow-glow-solar" />
              <span className="text-slate-300">SoLEXS: 1–30 keV</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-sky-400 shadow-glow-helios" />
              <span className="text-slate-300">HEL1OS: 10–150 keV</span>
            </div>
            <span className="text-slate-600">|</span>
            <span className="text-slate-400 font-mono">Lagrangian Point L1</span>
          </div>
        </header>

        <div className="p-6 flex-1">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
