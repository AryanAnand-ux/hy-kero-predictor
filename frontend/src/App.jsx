import { useState, useEffect, useCallback } from 'react';
import './index.css';
import DashboardPage from './pages/DashboardPage';
import HistoryPage   from './pages/HistoryPage';
import ModelsPage    from './pages/ModelsPage';
import PredictPage   from './pages/PredictPage';
import UploadPage    from './pages/UploadPage';
import { fetchHealth, fetchHistory } from './api';
import Chatbot from './components/Chatbot';
import ErrorBoundary from './components/ErrorBoundary';
import { FLASH_POINT_MIN, FLASH_POINT_MAX, TOAST_DISMISS_MS, ALERT_POLL_INTERVAL_MS } from './constants';
import { getFlashPointStatus } from './utils';
import logo from './assets/Indian_Oil_Logo.svg.webp';

const PAGES = [
  { id: 'dashboard', label: 'Dashboard',   icon: '📊' },
  { id: 'predict',   label: 'Predict',     icon: '🔮' },
  { id: 'history',   label: 'History',     icon: '📈' },
  { id: 'models',    label: 'Models',      icon: '🤖' },
  { id: 'upload',    label: 'Upload',      icon: '📂' },
];

export default function App() {
  const [page, setPage] = useState('dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [healthStatus, setHealthStatus] = useState('loading'); // 'loading', 'healthy', 'degraded', 'error'
  const [modelName, setModelName] = useState('');

  // ── Alert & Notification State ──
  const [latestCriticalAlert, setLatestCriticalAlert] = useState(null);
  const [dismissedAlertId, setDismissedAlertId] = useState(() => {
    return localStorage.getItem('acknowledged_alert_id') || null;
  });
  const [toasts, setToasts] = useState([]);

  // Toast Helpers
  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const addToast = useCallback((title, text, type = 'info') => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, title, text, type }]);
    setTimeout(() => {
      removeToast(id);
    }, TOAST_DISMISS_MS);
  }, [removeToast]);

  const showBrowserNotification = (title, body) => {
    if ('Notification' in window && Notification.permission === 'granted') {
      try {
        new Notification(title, { body });
      } catch (err) {
        console.warn("Failed to trigger native notification", err);
      }
    }
  };

  // Check latest prediction from database and trigger alerts if critical
  const checkLatestPrediction = useCallback(() => {
    fetchHistory('', '', '', 1)
      .then(res => {
        if (res.data && res.data.length > 0) {
          const latest = res.data[0];
          const val = latest.predicted;
          const status = getFlashPointStatus(val);
          const isCritical = status.status === 'danger';
          if (isCritical) {
            setLatestCriticalAlert(latest);
          } else {
            setLatestCriticalAlert(null);
          }
        }
      })
      .catch(err => console.error("Failed to fetch latest prediction for alerts", err));
  }, []);

  const handleAcknowledgeAlert = useCallback(() => {
    if (latestCriticalAlert) {
      localStorage.setItem('acknowledged_alert_id', String(latestCriticalAlert.id));
      setDismissedAlertId(String(latestCriticalAlert.id));
      addToast('Alert Acknowledged', `Critical alert for prediction ID ${latestCriticalAlert.id} has been acknowledged.`, 'success');
    }
  }, [latestCriticalAlert, addToast]);

  useEffect(() => {
    fetchHealth()
      .then(res => {
        if (res.status === 'healthy') {
          setHealthStatus('healthy');
          setModelName(res.model_name || '');
        } else {
          setHealthStatus('degraded');
        }
      })
      .catch(() => {
        setHealthStatus('error');
      });

    // Run alert check and set up interval
    checkLatestPrediction();
    const interval = setInterval(checkLatestPrediction, ALERT_POLL_INTERVAL_MS);

    const handleNewPrediction = (e) => {
      const pred = e.detail;
      checkLatestPrediction();

      const status = getFlashPointStatus(pred.predicted);
      const isCritical = status.status === 'danger';
      const isWarning = status.status === 'warning';

      if (isCritical) {
        const title = pred.predicted < FLASH_POINT_MIN 
          ? '🚨 CRITICAL: Flash Point Below Spec' 
          : '🚨 CRITICAL: Flash Point Above Spec';
        const msg = `Predicted flash point is ${pred.predicted}°C (Spec: ${FLASH_POINT_MIN}°C - ${FLASH_POINT_MAX}°C) at ${new Date(pred.sample_ts).toLocaleTimeString('en-IN')}.`;
        addToast(title, msg, 'critical');
        showBrowserNotification(title, msg);
      } else if (isWarning) {
        addToast('⚠️ Warning: High Flash Point', `Predicted flash point is high: ${pred.predicted}°C at ${new Date(pred.sample_ts).toLocaleTimeString('en-IN')}.`, 'warning');
      } else {
        addToast('✅ Prediction Completed', `Predicted flash point is in spec: ${pred.predicted}°C at ${new Date(pred.sample_ts).toLocaleTimeString('en-IN')}.`, 'success');
      }
    };

    const handleBatchPrediction = (e) => {
      const batchResult = e.detail;
      checkLatestPrediction();

      const total = batchResult.rows_processed;
      const criticalCount = batchResult.critical_count || 0;

      if (criticalCount > 0) {
        const title = `🚨 Batch Prediction: Critical Violations!`;
        const msg = `${criticalCount} of ${total} uploaded readings exceed critical limits (< ${FLASH_POINT_MIN}°C or > ${FLASH_POINT_MAX}°C).`;
        addToast(title, msg, 'critical');
        showBrowserNotification(title, msg);
      } else {
        addToast('✅ Batch Upload Completed', `${total} rows processed successfully. All predictions are within spec.`, 'success');
      }
    };

    window.addEventListener('new-prediction', handleNewPrediction);
    window.addEventListener('batch-prediction', handleBatchPrediction);

    // Request browser notification permissions
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }

    return () => {
      clearInterval(interval);
      window.removeEventListener('new-prediction', handleNewPrediction);
      window.removeEventListener('batch-prediction', handleBatchPrediction);
    };
  }, [checkLatestPrediction, addToast]);

  const renderPage = () => {
    switch (page) {
      case 'dashboard': return <DashboardPage />;
      case 'predict':   return <PredictPage />;
      case 'history':   return <HistoryPage />;
      case 'models':    return <ModelsPage />;
      case 'upload':    return <UploadPage />;
      default:          return <DashboardPage />;
    }
  };

  const activePage = PAGES.find(p => p.id === page);

  return (
    <>
    <div className={`app-layout ${sidebarOpen ? 'sidebar-open' : ''}`}>
      {/* ── Sidebar Backdrop for Mobile ── */}
      {sidebarOpen && (
        <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />
      )}

      {/* ── Sidebar ── */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-logo">
          <img src={logo} alt="IndianOil logo" className="sidebar-logo-img" />
          <div className="sidebar-logo-text">
            <h2>HY Kero</h2>
            <p>Flash Point Predictor</p>
          </div>
        </div>
        <nav className="sidebar-nav">
          {PAGES.map(p => (
            <button
              key={p.id}
              className={`nav-item ${page === p.id ? 'active' : ''}`}
              onClick={() => {
                setPage(p.id);
                setSidebarOpen(false); // Close on selection on mobile
              }}
            >
              <span style={{ fontSize: '1.1rem' }}>{p.icon}</span>
              <span>{p.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <p>IOCL — CDU Unit</p>
          <p>v1.0.0 • ML Pipeline</p>
        </div>
      </aside>

      {/* ── Main content ── */}
      <main className="main-content">
        <header className="topbar">
          <div className="topbar-left">
            <button 
              className="sidebar-toggle-btn"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              aria-label="Toggle Sidebar"
            >
              ☰
            </button>
            <div>
              <div className="topbar-title">
                {activePage?.icon}{' '}
                {activePage?.label}
              </div>
              <div className="topbar-sub">
                HY Kero Flash Point Prediction • Crude Distillation Unit
              </div>
            </div>
          </div>
          <div className="status-pill" style={{
            background: healthStatus === 'healthy' ? 'var(--green)' : healthStatus === 'degraded' ? 'var(--orange)' : healthStatus === 'error' ? 'var(--red)' : '#bbb'
          }}>
            <div className="status-dot" />
            {healthStatus === 'healthy' ? `Model Active (${modelName})` : healthStatus === 'degraded' ? 'Model Degraded' : healthStatus === 'error' ? 'API Offline' : 'Connecting...'}
          </div>
        </header>

        {latestCriticalAlert && String(latestCriticalAlert.id) !== dismissedAlertId && (
          <div className="global-alert-banner">
            <div className="global-alert-content">
              <span style={{ fontSize: '1.4rem' }}>🚨</span>
              <span className="global-alert-message">
                <strong>CRITICAL FLASH POINT DETECTED:</strong> {latestCriticalAlert.predicted}°C (Spec Limit: {FLASH_POINT_MIN}°C - {FLASH_POINT_MAX}°C) at {new Date(latestCriticalAlert.sample_ts).toLocaleString('en-IN')} (Shift: {latestCriticalAlert.shift})
              </span>
            </div>
            <div className="global-alert-actions">
              <button className="global-alert-btn ack-btn" onClick={handleAcknowledgeAlert}>
                ACKNOWLEDGE ALERT
              </button>
              <button className="global-alert-btn" onClick={() => setPage('history')}>
                VIEW HISTORY
              </button>
            </div>
          </div>
        )}

        <div className="page">
          <ErrorBoundary>
            {renderPage()}
          </ErrorBoundary>
        </div>
      </main>
    </div>
    <div className="toast-panel">
      {toasts.map(t => {
        let icon = 'ℹ️';
        if (t.type === 'critical') icon = '🚨';
        else if (t.type === 'success') icon = '✅';
        else if (t.type === 'warning') icon = '⚠️';
        return (
          <div key={t.id} className={`toast-item ${t.type}`}>
            <div className="toast-icon">{icon}</div>
            <div className="toast-body">
              <div className="toast-title">{t.title}</div>
              <div className="toast-text">{t.text}</div>
            </div>
            <button className="toast-close" onClick={() => removeToast(t.id)}>×</button>
          </div>
        );
      })}
    </div>
    <Chatbot />
    </>
  );
}
