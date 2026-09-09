import React from 'react';
import { Routes, Route } from 'react-router-dom';
import AppLayout from './layouts/AppLayout';

import Overview from './pages/Overview';
import LiveMonitor from './pages/LiveMonitor';
import DataPipeline from './pages/DataPipeline';
import DataExplorer from './pages/DataExplorer';
import FeatureAnalysis from './pages/FeatureAnalysis';
import Models from './pages/Models';
import Training from './pages/Training';
import Evaluation from './pages/Evaluation';
import FusionExperiment from './pages/FusionExperiment';
import Alerts from './pages/Alerts';
import System from './pages/System';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<AppLayout />}>
        <Route index element={<Overview />} />
        <Route path="live" element={<LiveMonitor />} />
        <Route path="pipeline" element={<DataPipeline />} />
        <Route path="data" element={<DataExplorer />} />
        <Route path="features" element={<FeatureAnalysis />} />
        <Route path="models" element={<Models />} />
        <Route path="training" element={<Training />} />
        <Route path="evaluation" element={<Evaluation />} />
        <Route path="fusion" element={<FusionExperiment />} />
        <Route path="alerts" element={<Alerts />} />
        <Route path="system" element={<System />} />
      </Route>
    </Routes>
  );
}
