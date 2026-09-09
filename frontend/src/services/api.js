import axios from 'axios';

// Keep API requests same-origin by default. Vite proxies /api in development
// and nginx proxies it in Docker, so this works from remote browsers too.
const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export const apiClient = axios.create({
  baseURL: API_BASE,
  // Preprocessing and model training can legitimately exceed 30 seconds on
  // the bundled demo dataset, especially when optional decomposition runs.
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // System & Health
  getHealth: () => apiClient.get('/api/health').then(r => r.data),
  getSystemInfo: () => apiClient.get('/api/system/status').then(r => r.data),

  // Dashboard
  getDashboardSummary: () => apiClient.get('/api/dashboard/summary').then(r => r.data),

  // Data Pipeline & Ingestion
  getDataInfo: () => apiClient.get('/api/data/info').then(r => r.data),
  getDataPreview: (n = 100) => apiClient.get(`/api/data/preview?n=${n}`).then(r => r.data),
  getDataSummary: () => apiClient.get('/api/data/summary').then(r => r.data),
  loadDemoData: () => apiClient.post('/api/data/load-demo').then(r => r.data),
  uploadDataset: (formData) => apiClient.post('/api/data/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data),

  // Preprocessing & Features
  runPreprocess: (params = {}) => apiClient.post('/api/preprocess', {
    target_cadence: ({ 1: '1s', 10: '10s', 60: '1min', 300: '5min' })[params.cadence_seconds]
      || params.target_cadence
      || '1min',
    preflare_minutes: params.pre_flare_window_minutes ?? params.preflare_minutes ?? 30,
    spike_zscore_threshold: params.spike_zscore_threshold ?? 6.0,
    compute_stl: params.stl_decomposition ?? params.compute_stl ?? false,
  }).then(r => r.data),
  getPipelineStatus: () => apiClient.get('/api/pipeline/status').then(r => r.data),
  getFeatures: () => apiClient.get('/api/features').then(r => r.data),

  // Models & Training
  listModels: () => apiClient.get('/api/models').then(r => r.data),
  selectModel: (model_id) => apiClient.post('/api/models/select', { model_id }).then(r => r.data),
  getModelMetrics: (model_id) => apiClient.get(`/api/models/${model_id}/metrics`).then(r => r.data),
  trainModel: (config) => apiClient.post('/api/models/train', config).then(r => r.data),

  // Prediction & Simulation
  predict: (rows, model_id = null) => apiClient.post('/api/predict', { rows, model_id }).then(r => r.data),
  getSimulationState: () => apiClient.get('/api/simulation/state').then(r => r.data),
  startSimulation: (config = {}) => apiClient.post('/api/simulation/start', {
    model_id: config.model_id || null,
    sequence_length: config.sequence_length || 30,
    speed: config.speed || 5,
  }).then(r => r.data),
  pauseSimulation: () => apiClient.post('/api/simulation/pause').then(r => r.data),
  resumeSimulation: () => apiClient.post('/api/simulation/resume').then(r => r.data),
  resetSimulation: () => apiClient.post('/api/simulation/reset').then(r => r.data),
  setSimulationSpeed: (speed) => apiClient.post(`/api/simulation/speed/${speed}`).then(r => r.data),

  // Experiments
  getFusionExperiment: (model_type = 'random_forest', epochs = 8, sequence_length = 30) =>
    apiClient.get(`/api/experiments/fusion?model_type=${model_type}&epochs=${epochs}&sequence_length=${sequence_length}`)
      .then(r => r.data),

  // Alerts
  getAlerts: (severity = null, limit = 100) => {
    const q = severity ? `?severity=${severity}&limit=${limit}` : `?limit=${limit}`;
    return apiClient.get(`/api/alerts${q}`).then(r => r.data);
  },
  acknowledgeAlert: (alertId, acknowledged = true) =>
    apiClient.patch(`/api/alerts/${alertId}/acknowledge`, { acknowledged }).then(r => r.data),
};

export default api;
