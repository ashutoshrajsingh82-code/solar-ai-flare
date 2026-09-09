import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Layers,
  Award,
  CheckCircle2,
  GitFork,
  ArrowRight,
  Sparkles,
  Info,
  Activity,
  Maximize2,
} from 'lucide-react';
import ModelCard from '../components/ModelCard';
import api from '../services/api';

export default function Models() {
  const [models, setModels] = useState([]);
  const [activeModelId, setActiveModelId] = useState('cnn_lstm_fusion_v1');
  const [selectedModel, setSelectedModel] = useState(null);
  const [notification, setNotification] = useState('');
  const [loading, setLoading] = useState(true);

  const fetchModels = async () => {
    try {
      const list = await api.listModels();
      setModels(list);
      const active = list.find((m) => m.active);
      if (active) {
        setActiveModelId(active.model_id);
        setSelectedModel(active);
      } else if (list.length > 0) {
        setSelectedModel(list[0]);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleSelectModel = async (modelId) => {
    try {
      await api.selectModel(modelId);
      setActiveModelId(modelId);
      setNotification(`Active inference engine updated to ${modelId}.`);
      fetchModels();
    } catch (e) {
      setNotification(`Failed to select model: ${e.response?.data?.detail || e.message}`);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="panel p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] font-mono uppercase tracking-widest text-sky-400 bg-sky-500/10 px-2 py-0.5 rounded">
              MODEL SUITE & REGISTRY
            </span>
          </div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <Cpu size={22} className="text-solar-400" />
            Aditya-L1 Model Registry & Architectures
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Compare classical feature baselines and dual-task deep sequence networks (Nowcast + Forecast)
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs bg-space-950 px-3 py-1.5 rounded-lg border border-space-800">
          <span className="text-slate-400">Active Engine:</span>
          <span className="text-solar-400 font-bold">{activeModelId}</span>
        </div>
      </div>

      {notification && (
        <div className="panel p-3 bg-solar-500/10 border-solar-500/30 text-xs font-mono text-solar-300 flex items-center gap-2">
          <CheckCircle2 size={14} className="text-emerald-400" />
          {notification}
        </div>
      )}

      {/* Candidate Deep Architecture Topology Banner */}
      <div className="panel p-6 bg-gradient-to-r from-space-900 via-space-850 to-space-900 border-space-750">
        <div className="flex items-center justify-between mb-4 border-b border-space-800 pb-3">
          <div className="flex items-center gap-2">
            <Layers size={18} className="text-solar-400" />
            <h3 className="text-sm font-bold text-white uppercase tracking-wider font-mono">
              Primary Deep Architecture: CNN-LSTM Dual-Head Multi-Task Network
            </h3>
          </div>
          <span className="text-[10px] font-mono bg-solar-500 text-space-950 px-2.5 py-0.5 rounded font-black uppercase">
            Primary Candidate
          </span>
        </div>

        {/* ASCII / Graphical Architecture Topology Diagram */}
        <div className="overflow-x-auto py-2">
          <div className="min-w-[700px] flex items-center justify-between text-center font-mono text-xs gap-2">
            {/* Stage 1: Input */}
            <div className="p-3 bg-space-950 border border-space-750 rounded-xl flex-1">
              <span className="text-[10px] text-slate-500 uppercase block">Input Tensor</span>
              <span className="font-bold text-solar-400 block mt-1">2-Channel Sequence</span>
              <span className="text-[10px] text-slate-400 block mt-0.5">(Batch, 30, 2)</span>
              <span className="text-[9px] text-slate-500 block">SoLEXS + HEL1OS</span>
            </div>

            <ArrowRight size={14} className="text-slate-600 shrink-0" />

            {/* Stage 2: 1D Conv */}
            <div className="p-3 bg-space-950 border border-space-750 rounded-xl flex-1">
              <span className="text-[10px] text-slate-500 uppercase block">Spatial / Local</span>
              <span className="font-bold text-sky-400 block mt-1">1D-CNN Extractor</span>
              <span className="text-[10px] text-slate-400 block mt-0.5">Conv1D + BatchNorm</span>
              <span className="text-[9px] text-slate-500 block">Kernel=5, Pool=2</span>
            </div>

            <ArrowRight size={14} className="text-slate-600 shrink-0" />

            {/* Stage 3: LSTM */}
            <div className="p-3 bg-space-950 border border-space-750 rounded-xl flex-1">
              <span className="text-[10px] text-slate-500 uppercase block">Temporal Memory</span>
              <span className="font-bold text-emerald-400 block mt-1">LSTM Sequence</span>
              <span className="text-[10px] text-slate-400 block mt-0.5">Hidden=64, Dropout=0.3</span>
              <span className="text-[9px] text-slate-500 block">Long-range flare history</span>
            </div>

            <ArrowRight size={14} className="text-slate-600 shrink-0" />

            {/* Stage 4: Latent */}
            <div className="p-3 bg-space-950 border border-solar-500/30 rounded-xl flex-1 shadow-glow-solar">
              <span className="text-[10px] text-solar-400 uppercase block font-semibold">Shared Latent</span>
              <span className="font-bold text-white block mt-1">Dense 64-d</span>
              <span className="text-[10px] text-slate-400 block mt-0.5">ReLU + Regularization</span>
              <span className="text-[9px] text-slate-500 block">Fused Representation</span>
            </div>

            <ArrowRight size={14} className="text-slate-600 shrink-0" />

            {/* Stage 5: Dual Heads */}
            <div className="space-y-2 flex-1">
              <div className="p-2 bg-space-950 border border-emerald-500/30 rounded-lg">
                <span className="text-[9px] text-emerald-400 uppercase block font-bold">Nowcast Head</span>
                <span className="font-bold text-white text-[11px] block">4-Class Softmax</span>
                <span className="text-[9px] text-slate-400 block">Quiet / Pre / Flare / Decay</span>
              </div>
              <div className="p-2 bg-space-950 border border-sky-500/30 rounded-lg">
                <span className="text-[9px] text-sky-400 uppercase block font-bold">Forecast Head</span>
                <span className="font-bold text-white text-[11px] block">5 Binary Sigmoids</span>
                <span className="text-[9px] text-slate-400 block">1h / 3h / 6h / 12h / 24h</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Model Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {models.map((m) => (
          <ModelCard
            key={m.model_id}
            model={m}
            isActive={m.model_id === activeModelId}
            onSelect={handleSelectModel}
            onInspect={(mod) => setSelectedModel(mod)}
          />
        ))}
      </div>
    </div>
  );
}
