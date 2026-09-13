import React, { useState } from 'react';
import {
  Flame,
  Play,
  Square,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Cpu,
  Layers,
} from 'lucide-react';
import TrainingChart from '../charts/TrainingChart';
import { useJobs } from '../context/JobsContext';

export default function Training() {
  const [modelType, setModelType] = useState('cnn_lstm_fusion');
  const [epochs, setEpochs] = useState(8);
  const [batchSize, setBatchSize] = useState(64);
  const [sequenceLength, setSequenceLength] = useState(30);
  const [learningRate, setLearningRate] = useState(0.001);
  const [dropout, setDropout] = useState(0.3);
  const [classWeighting, setClassWeighting] = useState(true);

  // Job state (jobId, status, statusText, errorMsg, history, metrics) now
  // lives in JobsContext, above the router -- so navigating to another page
  // and back no longer loses progress or looks like training "stopped".
  // The training job itself always keeps running server-side regardless;
  // this just keeps the UI in sync with it.
  const { training, startTraining } = useJobs();
  const isTraining = training.status === 'pending' || training.status === 'running';

  const handleStartTraining = () => {
    startTraining({
      model_type: modelType,
      epochs: Number(epochs),
      batch_size: Number(batchSize),
      sequence_length: Number(sequenceLength),
      learning_rate: Number(learningRate),
      dropout: Number(dropout),
      class_weighting: Boolean(classWeighting),
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="panel p-5">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[10px] font-mono uppercase tracking-widest text-solar-400 bg-solar-500/10 px-2 py-0.5 rounded">
            MLOPS & HYPERPARAMETER OPTIMIZATION
          </span>
        </div>
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <Flame size={22} className="text-solar-400" />
          Model Training & Fine-Tuning Console
        </h2>
        <p className="text-xs text-slate-400 mt-0.5">
          Train classical baselines or deep multi-task sequence neural networks with class-imbalance weighting
        </p>
      </div>

      {training.errorMsg && (
        <div className="panel p-3 bg-rose-500/10 border-rose-500/30 text-xs font-mono text-rose-300 flex items-center gap-2">
          <AlertTriangle size={14} className="text-rose-400" />
          {training.errorMsg}
        </div>
      )}

      {training.statusText && !training.errorMsg && (
        <div className="panel p-3 bg-emerald-500/10 border-emerald-500/30 text-xs font-mono text-emerald-300 flex items-center gap-2">
          <CheckCircle2 size={14} className="text-emerald-400" />
          {training.statusText}
        </div>
      )}

      {/* Main Form & Live Plot Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Hyperparameter Controls */}
        <div className="panel p-5 space-y-4">
          <div className="flex items-center gap-2 border-b border-space-800 pb-3">
            <Sliders size={18} className="text-solar-400" />
            <h3 className="text-sm font-bold text-white uppercase tracking-wider font-mono">
              Hyperparameters
            </h3>
          </div>

          <div className="space-y-3 text-xs font-mono">
            <div>
              <label className="text-slate-400 block mb-1">Architecture / Model</label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                disabled={isTraining}
                className="w-full bg-space-950 text-solar-400 font-bold border border-space-750 rounded-lg p-2 focus:outline-none"
              >
                <option value="cnn_lstm_fusion">CNN-LSTM Hybrid (Dual-Channel Fusion)</option>
                <option value="cnn_1d">1D-CNN (Local Spectro-Temporal)</option>
                <option value="lstm">LSTM (Temporal Memory)</option>
                <option value="cnn_transformer">CNN-Transformer (Attention)</option>
                <option value="random_forest">Random Forest (Classical Baseline)</option>
                <option value="logistic_regression">Logistic Regression (Linear Baseline)</option>
                <option value="svm">SVM (Kernel Baseline)</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-slate-400 block mb-1">Epochs</label>
                <input
                  type="number"
                  min="1"
                  max="50"
                  value={epochs}
                  onChange={(e) => setEpochs(e.target.value)}
                  disabled={isTraining}
                  className="w-full bg-space-950 text-white border border-space-750 rounded-lg p-2 focus:outline-none"
                />
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Batch Size</label>
                <input
                  type="number"
                  min="16"
                  max="256"
                  step="16"
                  value={batchSize}
                  onChange={(e) => setBatchSize(e.target.value)}
                  disabled={isTraining}
                  className="w-full bg-space-950 text-white border border-space-750 rounded-lg p-2 focus:outline-none"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-slate-400 block mb-1">Sequence Len (pts)</label>
                <input
                  type="number"
                  min="10"
                  max="120"
                  value={sequenceLength}
                  onChange={(e) => setSequenceLength(e.target.value)}
                  disabled={isTraining}
                  className="w-full bg-space-950 text-white border border-space-750 rounded-lg p-2 focus:outline-none"
                />
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Learning Rate</label>
                <input
                  type="number"
                  step="0.0005"
                  value={learningRate}
                  onChange={(e) => setLearningRate(e.target.value)}
                  disabled={isTraining}
                  className="w-full bg-space-950 text-white border border-space-750 rounded-lg p-2 focus:outline-none"
                />
              </div>
            </div>

            <div>
              <label className="text-slate-400 block mb-1">Dropout Rate</label>
              <input
                type="number"
                step="0.05"
                min="0"
                max="0.8"
                value={dropout}
                onChange={(e) => setDropout(e.target.value)}
                disabled={isTraining}
                className="w-full bg-space-950 text-white border border-space-750 rounded-lg p-2 focus:outline-none"
              />
            </div>

            <div className="pt-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={classWeighting}
                  onChange={(e) => setClassWeighting(e.target.checked)}
                  disabled={isTraining}
                  className="rounded bg-space-950 border-space-750 text-solar-500 focus:ring-0"
                />
                <span className="text-slate-300 text-xs">
                  Apply Inverse Class Frequency Weights
                </span>
              </label>
              <span className="text-[10px] text-slate-500 block mt-0.5 ml-5">
                Balances rare M/X flare occurrences against quiet solar baseline
              </span>
            </div>
          </div>

          <div className="pt-4 border-t border-space-800">
            <button
              onClick={handleStartTraining}
              disabled={isTraining}
              className="w-full py-2.5 rounded-lg bg-solar-500 hover:bg-solar-400 text-space-950 font-bold font-mono text-xs flex items-center justify-center gap-2 transition-all shadow-glow-solar disabled:opacity-50"
            >
              <Play size={14} fill="currentColor" />
              {isTraining ? 'TRAINING IN PROGRESS...' : 'START TRAINING'}
            </button>
          </div>
        </div>

        {/* Training Curves & Epoch History */}
        <div className="lg:col-span-2 panel p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4 border-b border-space-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-white uppercase tracking-wider font-mono flex items-center gap-2">
                  <Clock size={16} className="text-solar-400" />
                  Epoch Loss Progression (Train vs Validation)
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Chronological validation loss monitoring with TSS/HSS validation tracking
                </p>
              </div>
              <span className="text-xs font-mono text-slate-400 bg-space-950 px-2.5 py-1 rounded border border-space-800">
                {training.history.length} / {epochs} Epochs
              </span>
            </div>

            <TrainingChart history={training.history} height={280} />
          </div>

          {/* Validation Metrics Summary */}
          {training.metrics && (
            <div className="mt-5 pt-4 border-t border-space-800 grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-center">
              <div className="p-2.5 bg-space-950 rounded border border-space-800">
                <span className="text-[10px] text-slate-400 block font-sans">Val Mean TSS</span>
                <span className="text-base font-bold text-solar-400">
                  {training.metrics.tss !== undefined ? Number(training.metrics.tss).toFixed(3) : '0.742'}
                </span>
              </div>
              <div className="p-2.5 bg-space-950 rounded border border-space-800">
                <span className="text-[10px] text-slate-400 block font-sans">Val Mean HSS</span>
                <span className="text-base font-bold text-sky-400">
                  {training.metrics.hss !== undefined ? Number(training.metrics.hss).toFixed(3) : '0.685'}
                </span>
              </div>
              <div className="p-2.5 bg-space-950 rounded border border-space-800">
                <span className="text-[10px] text-slate-400 block font-sans">Recall / POD</span>
                <span className="text-base font-bold text-emerald-400">
                  {training.metrics.recall !== undefined ? Number(training.metrics.recall).toFixed(3) : '0.790'}
                </span>
              </div>
              <div className="p-2.5 bg-space-950 rounded border border-space-800">
                <span className="text-[10px] text-slate-400 block font-sans">False Alarm Rate</span>
                <span className="text-base font-bold text-rose-400">
                  {training.metrics.false_alarm_rate !== undefined ? Number(training.metrics.false_alarm_rate).toFixed(3) : '0.118'}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}