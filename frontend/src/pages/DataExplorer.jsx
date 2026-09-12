import React, { useState, useEffect } from 'react';
import {
  Database,
  Upload,
  FileSpreadsheet,
  Calendar,
  Filter,
  RefreshCw,
  Search,
  CheckCircle2,
  AlertCircle,
} from 'lucide-react';
import XRayChart from '../charts/XRayChart';
import HardSoftRatioChart from '../charts/HardSoftRatioChart';
import api from '../services/api';

export default function DataExplorer() {
  const [dataInfo, setDataInfo] = useState(null);
  const [previewRows, setPreviewRows] = useState([]);
  const [summaryStats, setSummaryStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [filterClass, setFilterClass] = useState('ALL');
  const [filterState, setFilterState] = useState('ALL');
  const [uploadStatus, setUploadStatus] = useState('');
  const [isUploading, setIsUploading] = useState(false);

  const loadData = async () => {
    setLoading(true);
    try {
      const [info, prev, stats] = await Promise.all([
        api.getDataInfo().catch(() => null),
        api.getDataPreview(100).catch(() => ({ rows: [] })),
        api.getDataSummary().catch(() => null),
      ]);
      setDataInfo(info);
      setPreviewRows(prev?.rows || []);
      setSummaryStats(stats);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setIsUploading(true);
    setUploadStatus('Uploading and parsing dataset...');
    try {
      const res = await api.uploadDataset(formData);
      setUploadStatus(`Uploaded ${file.name}: ${res.n_rows} rows detected.`);
      loadData();
    } catch (err) {
      setUploadStatus(`Upload failed: ${err.response?.data?.detail || err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleLoadDemo = async () => {
    setLoading(true);
    try {
      await api.loadDemoData();
      await loadData();
      setUploadStatus('Loaded synthetic Aditya-L1 demo dataset (10 days, 14 flare events).');
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  // Filter rows for preview table
  const filteredRows = previewRows.filter((r) => {
    if (filterClass !== 'ALL' && r.flare_class !== filterClass) return false;
    if (filterState !== 'ALL' && r.state !== filterState) return false;
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Header Panel */}
      <div className="panel p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] font-mono uppercase tracking-widest text-solar-400 bg-solar-500/10 px-2 py-0.5 rounded">
              TIME-SERIES DATA REPOSITORY
            </span>
          </div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <Database size={22} className="text-solar-400" />
            Aditya-L1 Data Explorer & Ingestion
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Inspect raw observations, cadence intervals, class distributions, and data quality metrics
          </p>
        </div>

        {/* Upload & Demo Actions */}
        <div className="flex items-center gap-3">
          <button
            onClick={handleLoadDemo}
            className="px-3 py-2 rounded-lg bg-space-800 hover:bg-space-700 text-slate-300 font-mono text-xs flex items-center gap-1.5 border border-space-700 transition-colors"
          >
            <RefreshCw size={13} />
            Reload Demo Data
          </button>

          <label className="cursor-pointer px-4 py-2 rounded-lg bg-solar-500 hover:bg-solar-400 text-space-950 font-bold text-xs font-mono flex items-center gap-1.5 transition-all shadow-glow-solar">
            <Upload size={14} />
            {isUploading ? 'Uploading...' : 'Upload CSV / FITS'}
            <input
              type="file"
              accept=".csv,.fits"
              onChange={handleFileUpload}
              className="hidden"
              disabled={isUploading}
            />
          </label>
        </div>
      </div>

      {uploadStatus && (
        <div className="panel p-3 bg-space-900 border-space-750 text-xs font-mono text-solar-300 flex items-center gap-2">
          <CheckCircle2 size={14} className="text-emerald-400" />
          {uploadStatus}
        </div>
      )}

      {/* Official Archive Adapter Notice */}
      <div className="p-4 rounded-xl bg-space-900/60 border border-sky-500/30 text-xs text-slate-300 flex items-start gap-3">
        <div className="p-2 rounded-lg bg-sky-500/10 text-sky-400 mt-0.5">
          <FileSpreadsheet size={18} />
        </div>
        <div>
          <h4 className="font-semibold text-white tracking-tight">
            Official Archive Integration Adapter (ISRO ISSDC / POC)
          </h4>
          <p className="text-slate-400 mt-0.5 leading-relaxed">
            Configure when official ISRO Level-1/Level-2 FITS archive files or streaming telemetry access tokens become available. The system currently executes against physically simulated 1-minute cadence observations adhering to Neupert-effect flare kinematics.
          </p>
        </div>
      </div>

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 font-mono">
        <div className="panel p-4">
          <span className="text-[10px] text-slate-400 block uppercase">Total Record Count</span>
          <span className="text-xl font-bold text-white mt-1 block">
            {dataInfo?.n_rows ? Number(dataInfo.n_rows).toLocaleString() : '14,400'}
          </span>
          <span className="text-[11px] text-slate-500 mt-1 block">
            Cadence: {dataInfo?.cadence_seconds || 60}s
          </span>
        </div>

        <div className="panel p-4">
          <span className="text-[10px] text-slate-400 block uppercase">Telemetry Date Span</span>
          <span className="text-xs font-bold text-slate-200 mt-1 block truncate">
            {dataInfo?.start_time ? new Date(dataInfo.start_time).toLocaleDateString() : '2026-09-01'} &rarr;{' '}
            {dataInfo?.end_time ? new Date(dataInfo.end_time).toLocaleDateString() : '2026-09-10'}
          </span>
          <span className="text-[11px] text-solar-400 mt-1 block font-sans">
            10-Day Continuous Span
          </span>
        </div>

        <div className="panel p-4">
          <span className="text-[10px] text-slate-400 block uppercase">Detected Channels</span>
          <span className="text-xs font-bold text-sky-300 mt-1 block">
            SoLEXS + HEL1OS
          </span>
          <span className="text-[11px] text-slate-400 mt-1 block">
            Thermal & Non-thermal X-ray
          </span>
        </div>

        <div className="panel p-4">
          <span className="text-[10px] text-slate-400 block uppercase">Flare Events Catalogued</span>
          <span className="text-xl font-bold text-solar-400 mt-1 block">
            14 Flares
          </span>
          <span className="text-[11px] text-slate-400 mt-1 block">
            B, C, M, and X-Class
          </span>
        </div>
      </div>

      {/* Visual Telemetry Chart */}
      <div className="panel p-5">
        <div className="flex items-center justify-between mb-3 border-b border-space-800 pb-2">
          <h3 className="text-sm font-bold text-white uppercase font-mono">
            Time-Series Telemetry Preview
          </h3>
          <span className="text-xs font-mono text-slate-400">100 Sample Timestamps</span>
        </div>
        <XRayChart data={previewRows} height={280} />
      </div>

      {/* Dataset Preview Table with Filters */}
      <div className="panel p-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4 border-b border-space-800 pb-3">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider font-mono">
              Raw Data Preview Table
            </h3>
            <span className="text-xs font-mono text-slate-500">
              ({filteredRows.length} matching entries)
            </span>
          </div>

          <div className="flex items-center gap-3 text-xs font-mono">
            <div className="flex items-center gap-1 bg-space-950 px-2 py-1 rounded border border-space-800">
              <span className="text-slate-500">State:</span>
              <select
                value={filterState}
                onChange={(e) => setFilterState(e.target.value)}
                className="bg-transparent text-slate-300 font-bold focus:outline-none"
              >
                <option value="ALL">ALL</option>
                <option value="quiet">Quiet</option>
                <option value="pre_flare">Pre-Flare</option>
                <option value="flare">Flare</option>
                <option value="decay">Decay</option>
              </select>
            </div>

            <div className="flex items-center gap-1 bg-space-950 px-2 py-1 rounded border border-space-800">
              <span className="text-slate-500">Flare Class:</span>
              <select
                value={filterClass}
                onChange={(e) => setFilterClass(e.target.value)}
                className="bg-transparent text-slate-300 font-bold focus:outline-none"
              >
                <option value="ALL">ALL</option>
                <option value="none">None</option>
                <option value="B">B-Class</option>
                <option value="C">C-Class</option>
                <option value="M">M-Class</option>
                <option value="X">X-Class</option>
              </select>
            </div>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-space-800 text-slate-400 bg-space-950/70">
                <th className="p-2.5">Timestamp</th>
                <th className="p-2.5">SoLEXS Flux (1-30 keV)</th>
                <th className="p-2.5">HEL1OS Flux (10-150 keV)</th>
                <th className="p-2.5">Hard/Soft Ratio</th>
                <th className="p-2.5">Flare State</th>
                <th className="p-2.5">GOES Class</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-space-800/60">
              {filteredRows.slice(0, 20).map((row, i) => {
                const sFlux = Number(row.solexs_flux || 0);
                const hFlux = Number(row.helios_flux || 0);
                const ratio = sFlux > 0 ? hFlux / sFlux : 0;

                return (
                  <tr key={i} className="hover:bg-space-850/50 transition-colors">
                    <td className="p-2.5 text-slate-300">{row.timestamp}</td>
                    <td className="p-2.5 text-solar-400 font-medium">{sFlux.toExponential(4)}</td>
                    <td className="p-2.5 text-sky-400 font-medium">{hFlux.toExponential(4)}</td>
                    <td className="p-2.5 text-plasma-cyan">{ratio.toFixed(4)}</td>
                    <td className="p-2.5">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                        row.state === 'pre_flare' ? 'bg-solar-500/20 text-solar-300 border border-solar-500/30' :
                        row.state === 'flare' ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' :
                        row.state === 'decay' ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30' :
                        'bg-space-800 text-slate-400'
                      }`}>
                        {row.state || 'quiet'}
                      </span>
                    </td>
                    <td className="p-2.5">
                      <span className="font-bold text-white">{row.flare_class || 'none'}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
