import { useState, useEffect, memo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer } from 'recharts';
import { fetchModelMetrics, fetchFeatureImportance } from '../api';
import BrutTooltip from '../components/BrutTooltip';
import { BRUT_COLORS_ARR as BRUT_COLORS, FEATURE_IMPORTANCE_TOP_N } from '../constants';

const ModelsPage = memo(function ModelsPage() {
  const [metrics, setMetrics]   = useState(null);
  const [fi, setFi]             = useState([]);
  const [loading, setLoading]   = useState(true);
  const [tab, setTab]           = useState('comparison');

  useEffect(() => {
    Promise.all([fetchModelMetrics(), fetchFeatureImportance(FEATURE_IMPORTANCE_TOP_N)])
      .then(([m, f]) => { setMetrics(m); setFi(f.features || []); })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div style={{ display: 'flex', justifyContent: 'center', padding: 80 }}>
      <div className="spinner" />
    </div>
  );

  const models = metrics?.models || [];
  const maxFi   = fi.length ? Math.max(...fi.map(f => f.importance)) : 1;

  return (
    <div>
      <div className="section-header">
        <div className="section-icon" style={{ background: '#B388FF' }}>🤖</div>
        <div>
          <h2>Model Evaluation & Overfitting</h2>
          <p>Performance metrics and time-aware validation scores across all trained models</p>
        </div>
      </div>

      {metrics?.best_model && (
        <div className="card" style={{ marginBottom: 24, background: '#C6FF00' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <span style={{ fontSize: '2.5rem' }}>🏆</span>
            <div>
              <div style={{ fontSize: 11, color: '#1a1a1a', textTransform: 'uppercase', letterSpacing: '.1em', fontFamily: 'Space Mono, monospace', fontWeight: 700 }}>Selected Production Model</div>
              <div style={{ fontSize: 24, fontWeight: 800, color: '#1a1a1a' }}>{metrics.best_model}</div>
              <div style={{ fontSize: 12, color: '#333', marginTop: 2, fontFamily: 'Space Mono, monospace', fontWeight: 600 }}>
                Test RMSE: <strong style={{ color: '#1a1a1a' }}>{models.find(m => m.is_best)?.test_rmse ?? '--'}</strong>
                &nbsp;|&nbsp;
                CV RMSE (5-Fold): <strong style={{ color: '#1a1a1a' }}>{models.find(m => m.is_best)?.cv_rmse ?? '--'}</strong>
                &nbsp;|&nbsp;
                Test R²: <strong style={{ color: '#1a1a1a' }}>{models.find(m => m.is_best)?.test_r2 ?? '--'}</strong>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="tabs">
        {['comparison', 'chart', 'features'].map(t => (
          <button key={t} className={`tab ${tab === t ? 'active' : ''}`}
            onClick={() => setTab(t)} style={{ cursor: 'pointer', textTransform: 'capitalize' }}>{t}</button>
        ))}
      </div>

      {tab === 'comparison' && (
        <>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Train RMSE</th>
                  <th>Test RMSE</th>
                  <th>CV RMSE (Mean)</th>
                  <th>5-Fold CV Scores</th>
                  <th>Test R²</th>
                  <th>MAPE</th>
                  <th>Overfit?</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m, i) => {
                  const hasCV = m.cv_rmse !== null && m.cv_rmse !== undefined;
                  // If train RMSE is way lower than test, or if R2 test is negative
                  const overfit = m.train_rmse < m.test_rmse * 0.5 || (m.test_r2 !== null && m.test_r2 < 0);
                  return (
                    <tr key={i} className={m.is_best ? 'best-row' : ''}>
                      <td className="highlight" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        {m.is_best && <span>🏆</span>}{m.model}
                      </td>
                      <td>{m.train_rmse ?? '--'}</td>
                      <td className="highlight">{m.test_rmse ?? '--'}</td>
                      <td>{hasCV ? m.cv_rmse : '--'}</td>
                      <td style={{ fontFamily: 'Space Mono, monospace', fontSize: '10px' }}>
                        {m.cv_folds && m.cv_folds.length ? `[${m.cv_folds.join(', ')}]` : 'N/A'}
                      </td>
                      <td className="highlight">{m.test_r2 ?? '--'}</td>
                      <td>{m.test_mape !== null ? `${m.test_mape}%` : '--'}</td>
                      <td style={{ color: overfit ? '#FF5252' : '#00E676', fontWeight: 700 }}>
                        {overfit ? '⚠️ YES' : '✅ NO'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="helper-banner" style={{ marginTop: 18, background: '#fef6dd' }}>
            The current production choice favors simpler regularized models because they generalize more reliably across the latest operating window.
          </div>

          {/* Overfitting Analysis Section */}
          <div className="card" style={{ marginTop: 24, borderLeft: '8px solid var(--orange)' }}>
            <h3 style={{ textTransform: 'uppercase', fontSize: 13, fontFamily: 'Space Mono', marginBottom: 10 }}>💡 Overfitting & Distribution Shift Analysis</h3>
            <div style={{ fontSize: 13, display: 'flex', flexDirection: 'column', gap: 10, lineHeight: 1.6 }}>
              <p>
                <strong>The Overfitting Problem:</strong> Complex tree-based models (like <em>Random Forest</em>, <em>XGBoost</em>, and <em>LightGBM</em>) fit the training set well but performed poorly on test data (negative R² values). This is a classic case of overfitting — models memorize training patterns that don't generalize to the shifted operating regime.
              </p>
              <p>
                <strong>The Root Cause (Distribution Shift):</strong> Statistical tests revealed a significant shift in the target variable. The train period (April 2025 – January 2026) has a mean Flash Point of <strong>76.7°C</strong>, while the test period (January 2026 – March 2026) shifted up to <strong>80.4°C</strong>. Complex non-linear tree models cannot extrapolate outside their training domain, resulting in negative validation scores.
              </p>
              <div>
                <strong>The Solution (Regularization, Feature Pruning & Winsorization):</strong>
                <ul style={{ paddingLeft: 20, marginTop: 6 }}>
                  <li><strong>Huber Regressor (Tuned):</strong> The current production model uses a robust Huber loss to suppress the influence of extreme residuals from shifted operating conditions. It is selected by time-aware CV RMSE, resulting in a stable <strong>Test R² of ~0.59</strong> and <strong>Test RMSE of ~2.46°C</strong>.</li>
                  <li><strong>Target Winsorization:</strong> Training targets are clipped at the 0.5th–99.5th percentiles to prevent extreme lab measurement anomalies from skewing model coefficients.</li>
                  <li><strong>Top-40 Feature Pruning:</strong> Features are ranked by absolute Huber coefficient magnitude and pruned to the top 40 out of 69, reducing noise and collinearity.</li>
                  <li><strong>TimeSeriesSplit CV:</strong> We employ a time-aware 5-fold cross-validation scheme to evaluate models chronologically, preventing leakage from future data and mimicking production performance.</li>
                </ul>
              </div>
            </div>
          </div>
        </>
      )}

      {tab === 'chart' && (
        <div className="chart-container">
          <div className="chart-header">
            <span className="chart-title">Test RMSE by Model (lower is better)</span>
          </div>
          <ResponsiveContainer width="100%" height={380}>
            <BarChart data={models} layout="vertical" margin={{ top: 5, right: 30, left: 120, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(26,26,26,.1)" horizontal={false} />
              <XAxis type="number" tick={{ fill: '#555', fontSize: 11, fontFamily: 'Space Mono' }} />
              <YAxis type="category" dataKey="model" tick={{ fill: '#1a1a1a', fontSize: 11, fontFamily: 'Space Mono', fontWeight: 700 }} width={115} />
              <Tooltip content={<BrutTooltip showLabel={false} unit="" />} />
              <Bar dataKey="test_rmse" name="Test RMSE" radius={0}>
                {models.map((m, i) => (
                  <Cell key={i} fill={m.is_best ? '#00E676' : BRUT_COLORS[i % BRUT_COLORS.length]} stroke="#1a1a1a" strokeWidth={2} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {tab === 'features' && (
        <div className="chart-container">
          <div className="chart-header">
            <span className="chart-title">Top 15 Feature Importances (Selected Model)</span>
          </div>
          {fi.map((f, i) => (
            <div className="fi-row" key={i}>
              <div className="fi-label" style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>{f.feature}</span>
                <span style={{ color: '#1a1a1a', fontWeight: 700 }}>{(f.importance * 100).toFixed(2)}%</span>
              </div>
              <div className="fi-bar-bg">
                <div className="fi-bar-fill" style={{ width: `${(f.importance / maxFi) * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

export default ModelsPage;
