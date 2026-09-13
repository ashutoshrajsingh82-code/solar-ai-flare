import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import api from '../services/api';

/**
 * Tracks long-running background jobs (training, preprocessing) at a level
 * ABOVE the routed page components, so navigating from e.g. /training to
 * /models does not kill the polling loop or reset progress.
 *
 * Previously each page (Training.jsx, DataPipeline.jsx) held its own job
 * state and polling loop in local useState/closures. React Router unmounts
 * a page component when you navigate away from its route, which killed the
 * loop and reset state on remount -- even though the actual job kept
 * running fine on the server the whole time. Wrapping <Routes> with this
 * provider (see main.jsx) means the provider itself never unmounts during
 * normal navigation, so polling continues and the page just re-reads
 * whatever the current job state is when you come back to it.
 */

const JobsContext = createContext(null);

const initialTrainingState = {
  jobId: null,
  status: 'idle', // idle | pending | running | done | failed
  statusText: '',
  errorMsg: '',
  history: [],
  metrics: null,
};

const initialPipelineState = {
  jobId: null,
  status: 'idle', // idle | pending | running | done | failed
  message: '',
};

const MAX_TRAIN_POLL_MS = 30 * 60 * 1000; // 30 min
const MAX_PIPELINE_POLL_MS = 20 * 60 * 1000; // 20 min
const TRAIN_POLL_INTERVAL_MS = 3000;
const PIPELINE_POLL_INTERVAL_MS = 2000;

export function JobsProvider({ children }) {
  const [training, setTraining] = useState(initialTrainingState);
  const [pipeline, setPipeline] = useState(initialPipelineState);

  const trainingTimerRef = useRef(null);
  const trainingStartedAtRef = useRef(0);
  const pipelineTimerRef = useRef(null);
  const pipelineStartedAtRef = useRef(0);

  const stopTrainingPolling = () => {
    if (trainingTimerRef.current) {
      clearInterval(trainingTimerRef.current);
      trainingTimerRef.current = null;
    }
  };

  const stopPipelinePolling = () => {
    if (pipelineTimerRef.current) {
      clearInterval(pipelineTimerRef.current);
      pipelineTimerRef.current = null;
    }
  };

  const pollTrainingOnce = async (jobId) => {
    try {
      const job = await api.getTrainingJobStatus(jobId);

      if (job.status === 'running') {
        setTraining((prev) => (prev.jobId !== jobId ? prev : {
          ...prev,
          status: 'running',
          statusText: `Training in progress (job ${jobId})...`,
        }));
      } else if (job.status === 'done') {
        const res = job.result || {};
        setTraining((prev) => (prev.jobId !== jobId ? prev : {
          ...prev,
          status: 'done',
          statusText: `Training completed successfully. Model registered: ${res.model_id}`,
          history: res.history || prev.history,
          metrics: res.metrics || prev.metrics,
        }));
        stopTrainingPolling();
      } else if (job.status === 'failed') {
        setTraining((prev) => (prev.jobId !== jobId ? prev : {
          ...prev,
          status: 'failed',
          errorMsg: `Training failed: ${job.error}`,
        }));
        stopTrainingPolling();
      }

      if (Date.now() - trainingStartedAtRef.current > MAX_TRAIN_POLL_MS) {
        setTraining((prev) => (prev.jobId !== jobId ? prev : {
          ...prev,
          errorMsg: 'Training is taking unusually long (30+ min) -- check the backend logs.',
        }));
        stopTrainingPolling();
      }
    } catch (e) {
      // Transient network hiccup on a single poll shouldn't kill tracking --
      // just log it and let the next tick try again.
      console.error('Training poll failed:', e);
    }
  };

  const startTraining = async (config) => {
    stopTrainingPolling();
    setTraining({ ...initialTrainingState, status: 'pending', statusText: 'Initializing training job...' });
    try {
      const { job_id } = await api.trainModel(config);
      trainingStartedAtRef.current = Date.now();
      setTraining((prev) => ({ ...prev, jobId: job_id, statusText: `Training job queued (${job_id})...` }));
      trainingTimerRef.current = setInterval(() => pollTrainingOnce(job_id), TRAIN_POLL_INTERVAL_MS);
    } catch (err) {
      setTraining((prev) => ({
        ...prev,
        status: 'failed',
        errorMsg: `Training failed: ${err.response?.data?.detail || err.message}`,
      }));
    }
  };

  const pollPipelineOnce = async (jobId) => {
    try {
      const job = await api.getPreprocessJobStatus(jobId);

      if (job.status === 'running') {
        setPipeline((prev) => (prev.jobId !== jobId ? prev : {
          ...prev,
          status: 'running',
          message: `Pipeline running (job ${jobId})...`,
        }));
      } else if (job.status === 'done') {
        setPipeline((prev) => (prev.jobId !== jobId ? prev : {
          ...prev,
          status: 'done',
          message: 'Preprocessing pipeline completed successfully.',
        }));
        stopPipelinePolling();
      } else if (job.status === 'failed') {
        setPipeline((prev) => (prev.jobId !== jobId ? prev : {
          ...prev,
          status: 'failed',
          message: `Error running pipeline: ${job.error}`,
        }));
        stopPipelinePolling();
      }

      if (Date.now() - pipelineStartedAtRef.current > MAX_PIPELINE_POLL_MS) {
        setPipeline((prev) => (prev.jobId !== jobId ? prev : {
          ...prev,
          message: 'Pipeline is taking unusually long (20+ min) -- check the backend logs.',
        }));
        stopPipelinePolling();
      }
    } catch (e) {
      console.error('Pipeline poll failed:', e);
    }
  };

  const startPipeline = async (config) => {
    stopPipelinePolling();
    setPipeline({ ...initialPipelineState, status: 'pending', message: 'Initializing pipeline job...' });
    try {
      const { job_id } = await api.runPreprocess(config);
      pipelineStartedAtRef.current = Date.now();
      setPipeline((prev) => ({ ...prev, jobId: job_id, message: `Pipeline job queued (${job_id})...` }));
      pipelineTimerRef.current = setInterval(() => pollPipelineOnce(job_id), PIPELINE_POLL_INTERVAL_MS);
    } catch (err) {
      setPipeline((prev) => ({
        ...prev,
        status: 'failed',
        message: `Error running pipeline: ${err.response?.data?.detail || err.message}`,
      }));
    }
  };

  // This provider wraps <Routes> (see main.jsx) so it only unmounts when the
  // whole app closes -- this cleanup is just good hygiene, not something
  // that fires on ordinary page navigation.
  useEffect(() => {
    return () => {
      stopTrainingPolling();
      stopPipelinePolling();
    };
  }, []);

  return (
    <JobsContext.Provider value={{ training, pipeline, startTraining, startPipeline }}>
      {children}
    </JobsContext.Provider>
  );
}

export function useJobs() {
  const ctx = useContext(JobsContext);
  if (!ctx) {
    throw new Error('useJobs must be used within a JobsProvider');
  }
  return ctx;
}