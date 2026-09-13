import React, { useState, useEffect } from 'react';
import {
  Workflow,
  Play,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  ArrowDown,
  Clock,
  Layers,
  Sparkles,
  Info,
  Sliders,
} from 'lucide-react';
import PipelineStage from '../components/PipelineStage';
import api from '../services/api';
import { useJobs } from '../context/JobsContext';

const PIPELINE_DEFINITIONS = [
  {
    key: 'raw',
    title: 'Raw Data Ingestion',
    description: 'Ingestion of dual X-ray telemetry streams from SoLEXS (1-30 keV) & HEL1OS (10-150 keV) + GOES flare catalogue.',
  },
  {
    key: 'cleaned',
    title: 'Quality Control & Cleaning',
    description: 'De-duplication, detection and filtration of non-physical X-ray spikes, and imputation of short telemetry gaps.',
  },
  {
    key: 'aligned',
    title: 'Time Alignment & Resampling',
    description: 'Sub-minute timestamp synchronization of independent SoLEXS and HEL1OS clocks to a uniform 60s cadence grid.',
  },
  {
    key: 'transformed',
    title: 'Log Transformation & Scaling',
    description: 'log10(x + eps) compression across 6 orders of magnitude + StandardScaler fit strictly on training split (no leakage).',
  },
  {
    key: 'featured',
    title: 'Feature Engineering & Neupert Ratio',
    description: 'Derivation of rolling statistics (5m/10m/30m/60m), gradients, Hard/Soft ratio (HEL1OS/SoLEXS), and optional STL components.',
  },
  {
    key: 'labeled',
    title: 'Sliding Window & Dual Labels',
    description: 'Generation of 4-class Nowcast labels + 5 binary forecast horizons (1h/3h/6h/12h/24h) with leak-free chronological splitting.',
  },
];

export default function DataPipeline() {
  const [stages, setStages] = useState([]);
  const [selectedStageKey, setSelectedStageKey] = useState('featured');
  const [cadence, setCadence] = useState(60);
  const [preFlareWindow, setPreFlareWindow] = useState(60);

  // Job state (jobId, status, message) now lives in JobsContext, above the
  // router -- so navigating to another page and back no longer loses
  // progress or looks like the pipeline "stopped". The job itself always
  // keeps running server-side regardless; this just keeps the UI in sync.
  const { pipeline, startPipeline } = useJobs();
  const loading = pipeline.status === 'pending' || pipeline.status === 'running';

  const fetchStatus = async () => {
    try {
      const res = await api.getPipelineStatus();
      if (res && res.stages) {
        setStages(res.stages);
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  // While a pipeline job is running (tracked in context, survives
  // navigation), keep refreshing the per-stage cards on this page so they
  // update live -- this effect re-attaches whenever you land back on this
  // page while a job is still in flight.
  useEffect(() => {
    if (!loading) return undefined;
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, [loading]);

  // Also refresh once right after a job finishes, to catch the final stage.
  useEffect(() => {
    if (pipeline.status === 'done') fetchStatus();
  }, [pipeline.status]);

  const handleRunPipeline = () => {
    startPipeline({
      cadence_seconds: Number(cadence),
      pre_flare_window_minutes: Number(preFlareWindow),
      clean_duplicates: true,
      remove_spikes: true,
      // STL is useful for exploration but considerably slower; keep the
      // normal interactive pipeline responsive by leaving it optional.
      stl_decomposition: false,
    });
  };

  const activeStageDetails = stages.find((s) => s.stage === selectedStageKey) || {};
  const activeDef = PIPELINE_DEFINITIONS.find((d) => d.key === selectedStageKey);

  return (
    <div className="space-y-6">
      {/* Header Panel */}
      <div className="panel p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] font-mono uppercase tracking-widest text-sky-400 bg-sky-500/10 px-2 py-0.5 rounded">
              SCIENTIFIC ETL & FEATURE STORE
            </span>
          </div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <Workflow size={22} className="text-solar-400" />
            Aditya-L1 Data Preprocessing Pipeline
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Transparent execution flow from raw sensor telemetry to feature-engineered multi-task tensors
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs font-mono bg-space-950 px-3 py-1.5 rounded-lg border border-space-800">
            <span className="text-slate-400">Resampling Cadence:</span>
            <select
              value={cadence}
              onChange={(e) => setCadence(e.target.value)}
              className="bg-space-900 text-solar-400 font-bold border border-space-750 rounded px-1.5 py-0.5 text-xs focus:outline-none"
            >
              <option value="10">10s (High-Res Nowcast)</option>
              <option value="60">60s (Standard)</option>
              <option value="300">300s (Long Forecast)</option>
            </select>
          </div>

          <button
            onClick={handleRunPipeline}
            disabled={loading}
            className="px-4 py-2 rounded-lg bg-solar-500 hover:bg-solar-400 text-space-950 font-bold text-xs font-mono flex items-center gap-1.5 transition-all shadow-glow-solar disabled:opacity-50"
          >
            <Play size={14} fill="currentColor" />
            {loading ? 'PROCESSING PIPELINE...' : 'RUN FULL PIPELINE'}
          </button>
        </div>
      </div>

      {pipeline.message && (
        <div className="panel p-3 bg-solar-500/10 border-solar-500/30 text-xs font-mono text-solar-300">
          {pipeline.message}
        </div>
      )}

      {/* Visual Pipeline Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Stages List */}
        <div className="lg:col-span-2 space-y-3">
          {PIPELINE_DEFINITIONS.map((def, idx) => {
            const stageData = stages.find((s) => s.stage === def.key) || {};
            const isSelected = selectedStageKey === def.key;

            return (
              <React.Fragment key={def.key}>
                <PipelineStage
                  stageKey={def.key}
                  title={def.title}
                  description={def.description}
                  status={stageData.status || 'pending'}
                  inputRows={stageData.input_rows}
                  outputRows={stageData.output_rows}
                  durationSec={stageData.duration_sec}
                  warnings={stageData.warnings || []}
                  isSelected={isSelected}
                  onClick={() => setSelectedStageKey(def.key)}
                />
                {idx < PIPELINE_DEFINITIONS.length - 1 && (
                  <div className="flex justify-center py-0.5 text-slate-600">
                    <ArrowDown size={14} />
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Selected Stage Detail Inspector */}
        <div className="panel p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 border-b border-space-800 pb-3 mb-4">
              <Info size={18} className="text-solar-400" />
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                Stage Diagnostic Inspector
              </h3>
            </div>

            <div className="mb-4">
              <span className="text-[10px] font-mono text-solar-400 uppercase tracking-widest font-semibold block">
                Selected Stage
              </span>
              <h4 className="text-base font-bold text-white mt-0.5">
                {activeDef?.title}
              </h4>
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                {activeDef?.description}
              </p>
            </div>

            <div className="space-y-2.5 font-mono text-xs">
              <div className="p-2.5 rounded bg-space-950/70 border border-space-800 flex justify-between">
                <span className="text-slate-400">Execution Status</span>
                <span className="text-emerald-400 font-bold uppercase">
                  {activeStageDetails.status || 'Pending'}
                </span>
              </div>
              <div className="p-2.5 rounded bg-space-950/70 border border-space-800 flex justify-between">
                <span className="text-slate-400">Input Telemetry Points</span>
                <span className="text-white font-bold">
                  {activeStageDetails.input_rows !== undefined && activeStageDetails.input_rows !== null
                    ? Number(activeStageDetails.input_rows).toLocaleString()
                    : '--'}
                </span>
              </div>
              <div className="p-2.5 rounded bg-space-950/70 border border-space-800 flex justify-between">
                <span className="text-slate-400">Output Valid Rows</span>
                <span className="text-solar-400 font-bold">
                  {activeStageDetails.output_rows !== undefined && activeStageDetails.output_rows !== null
                    ? Number(activeStageDetails.output_rows).toLocaleString()
                    : '--'}
                </span>
              </div>
              <div className="p-2.5 rounded bg-space-950/70 border border-space-800 flex justify-between">
                <span className="text-slate-400">Processing Time</span>
                <span className="text-white font-bold">
                  {activeStageDetails.duration_sec !== undefined && activeStageDetails.duration_sec !== null
                    ? `${activeStageDetails.duration_sec}s`
                    : '--'}
                </span>
              </div>
            </div>

            {activeStageDetails.details && Object.keys(activeStageDetails.details).length > 0 && (
              <div className="mt-4 pt-3 border-t border-space-800">
                <span className="text-[11px] font-mono text-slate-400 block mb-2 font-semibold">
                  Metadata & Parameters:
                </span>
                <pre className="p-3 bg-space-950 rounded border border-space-800 text-[11px] font-mono text-slate-300 overflow-x-auto max-h-56">
                  {JSON.stringify(activeStageDetails.details, null, 2)}
                </pre>
              </div>
            )}
          </div>

          <div className="mt-4 pt-3 border-t border-space-800 text-[11px] font-mono text-slate-500">
            Scientifically verified leak-free chronological splitting active.
          </div>
        </div>
      </div>
    </div>
  );
}