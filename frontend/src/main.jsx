import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import { JobsProvider } from './context/JobsContext';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <JobsProvider>
        <App />
      </JobsProvider>
    </BrowserRouter>
  </React.StrictMode>,
);