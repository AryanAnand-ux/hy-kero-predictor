import { useState, useEffect, memo, useMemo, useCallback } from 'react';
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ComposedChart, Line, Area, ReferenceLine } from 'recharts';
import { fetchHistory, fetchHistoryStats, fetchModelMetrics } from '../api';
import Kpi from '../components/Kpi';
import { BRUT_COLORS_OBJ as BRUT_COLORS, FLASH_POINT_MIN, FLASH_POINT_MAX } from '../constants';

const DashboardPage = memo(function DashboardPage() {
  const [chartData, setChartData]   = useState([]);
  const [stats, setStats]           = useState(null);
  const [bestModel, setBestModel]   = useState(null);
  const [loading, setLoading]       = useState(true);
  const [error, setError]           = useState('');

  useEffect(() => {
    Promise.all([fetchHistory('', '', '', 300), fetchHistoryStats(), fetchModelMetrics()])
      .then(([h, s, m]) => {
        const bestModelObj = m.models?.find(mo => mo.is_best) || null;
        const rmse = bestModelObj?.test_rmse ?? 2.46;
        
        // Deduplicate records by sample_ts to avoid graph overlapping/straight line issues on duplicate timestamps
        const seenTs = new Set();
        const uniqueRows = [];
        const rawData = h.data || [];
        
        for (const r of rawData) {
          if (!r.sample_ts) continue;
          if (seenTs.has(r.sample_ts)) continue;
          seenTs.add(r.sample_ts);
          
          const pred = Number(r.predicted.toFixed(2));
          const ciLower = Number((r.confidence_lower ?? (pred - 1.96 * rmse)).toFixed(2));
          const ciUpper = Number((r.confidence_upper ?? (pred + 1.96 * rmse)).toFixed(2));
          
          uniqueRows.push({
            ts: r.sample_ts,
            time: new Date(r.sample_ts).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }),
            actual: r.actual !== null ? Number(r.actual.toFixed(2)) : null,
            predicted: pred,
            ci_lower: ciLower,
            ci_upper: ciUpper,
            ci_diff: Number((ciUpper - ciLower).toFixed(2)),
            shift: r.shift,
          });
        }
        
        // Keep the 60 most recent unique predictions, then reverse to chronological order for the chart
        const rows = uniqueRows.slice(0, 60).reverse();
        
        setChartData(rows);
        setStats(s);
        setBestModel(bestModelObj);
      })
      .catch(err => {
        console.error(err);
        setError(`Cannot connect to backend server. ${err.message || 'Make sure the backend API is running and model artifacts exist.'}`);
      })
      .finally(() => setLoading(false));
  }, []);

  // Memoize processed data, computed domains, and X-axis ticks to avoid re-renders
  const processed = useMemo(() => {
    if (!chartData || chartData.length === 0) return { data: [], yMin: 0, yMax: 100, ticks: [] };
    const values = chartData.flatMap(d => [d.actual, d.predicted].filter(v => typeof v === 'number'));
    const dataMin = Math.min(...values);
    const dataMax = Math.max(...values);
    const padding = Math.max((dataMax - dataMin) * 0.08, 1);
    const yMin = Math.max(dataMin - padding, 0);
    const yMax = dataMax + padding;

    // Build tick list from available timestamps: at most 8 ticks evenly spaced
    const maxTicks = 8;
    const len = chartData.length;
    const tickCount = Math.min(maxTicks, len);
    const step = Math.max(1, Math.floor(len / tickCount));
    const rawTicks = [];
    for (let i = 0; i < len; i += step) rawTicks.push(chartData[i].ts);
    const last = chartData[len - 1].ts;
    if (rawTicks.length === 0 || rawTicks[rawTicks.length - 1] !== last) rawTicks.push(last);

    // Deduplicate ticks by date (keep first occurrence per day)
    const seen = new Set();
    const ticks = rawTicks.filter(ts => {
      const key = new Date(ts).toDateString();
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

    return { data: chartData, yMin, yMax, ticks };
  }, [chartData]);

  const renderDot = useCallback((props) => {
    const { cx, cy, payload } = props;
    if (payload == null || payload.actual == null) return null;
    const outOfSpec = payload.actual < FLASH_POINT_MIN || payload.actual > FLASH_POINT_MAX;
    const fill = outOfSpec ? '#FF5252' : '#FFE500';
    return (
      <circle cx={cx} cy={cy} r={4} stroke="#1a1a1a" strokeWidth={1} fill={fill} />
    );
  }, []);

  const CustomTooltip = useCallback(({ active, payload }) => {
    if (!active || !payload || payload.length === 0) return null;
    // payload contains series entries; find data point
    const point = payload[0].payload;
    const residual = point.actual != null ? Number((point.actual - point.predicted).toFixed(2)) : null;
    return (
      <div className="brut-tooltip" style={{ background: '#fff', border: '3px solid #1a1a1a', padding: 12, fontFamily: 'Space Mono, monospace', fontSize: 12, color: '#111' }}>
        <div style={{ fontWeight: 800, marginBottom: 6 }}>{new Date(point.ts).toLocaleString('en-IN')}</div>
        <div style={{ display: 'flex', gap: 10 }}>
          <div style={{ minWidth: 120 }}>
            <div><strong>Shift:</strong> {point.shift}</div>
            <div><strong>Actual:</strong> {point.actual ?? '—'}</div>
            <div><strong>Predicted:</strong> {point.predicted}</div>
            <div><strong>Residual:</strong> {residual ?? '—'}</div>
          </div>
          <div style={{ minWidth: 140 }}>
            <div><strong>95% CI</strong></div>
            <div>{point.ci_lower} — {point.ci_upper}</div>
          </div>
        </div>
      </div>
    );
  }, []);

  return (
    <div>
      <div className="section-header" style={{ marginBottom: 24 }}>
        <div className="section-icon" style={{ background: BRUT_COLORS.yellow }}>🏭</div>
        <div>
          <h2>HY Kero Flash Point Dashboard</h2>
          <p>Real-time monitoring — CDU Heavy Kerosene Quality Prediction</p>
        </div>
      </div>

      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: 60 }}><div className="spinner" /></div>
      ) : error ? (
        <div style={{ padding: 20, background: '#FF5252', border: '3px solid #1a1a1a', boxShadow: '5px 5px 0 #1a1a1a', color: '#fff', fontSize: 13, fontFamily: 'Space Mono, monospace', fontWeight: 700, margin: '24px 0' }}>
          ⚠️ {error}
        </div>
      ) : (
        <>
          <div className="stats-grid">
            <Kpi label="Avg Flash GC (Actual)"   value={stats ? `${stats.actual_mean}°C` : '--'}    color={BRUT_COLORS.green}  icon="🌡️" sub="Mean of all lab results" />
            <Kpi label="Flash GC Range"           value={stats ? `${stats.actual_min}–${stats.actual_max}°C` : '--'} color={BRUT_COLORS.orange} icon="📊" sub="Min to Max" />
            <Kpi label="Total Lab Measurements"   value={stats?.total_predictions ?? '--'}            color={BRUT_COLORS.blue}   icon="🧪" sub="Across all shifts" />
            <Kpi label="Best Model"               value={bestModel?.model ?? '--'}                   color={BRUT_COLORS.purple} icon="🤖" sub={bestModel ? `RMSE: ${bestModel.test_rmse} | R²: ${bestModel.test_r2}` : ''} />
          </div>

          <div className="chart-container">
            <div className="chart-header">
              <div>
                <div className="chart-title">Flash GC — Last 60 Predictions</div>
                <div style={{ fontSize: 11, color: '#555', marginTop: 4, fontFamily: 'Space Mono, monospace' }}>Actual (lab) vs Model Predicted</div>
              </div>
              <div style={{ display: 'flex', gap: 16, fontSize: 11, fontFamily: 'Space Mono, monospace', fontWeight: 700, alignItems: 'center' }}>
                <span style={{ color: '#1a1a1a', display: 'flex', alignItems: 'center', gap: 4 }}>● Actual (Lab)</span>
                <span style={{ color: '#4D7CFF', display: 'flex', alignItems: 'center', gap: 4 }}>┈ Predicted (ML)</span>
                <span style={{ color: 'rgba(77, 124, 255, 0.4)', display: 'flex', alignItems: 'center', gap: 4 }}>■ 95% Confidence Interval</span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={processed.data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="2 2" stroke="rgba(26,26,26,.06)" />
                <XAxis
                  dataKey="ts"
                  ticks={processed.ticks}
                  tickFormatter={(v) => new Date(v).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' })}
                  tick={{ fill: '#555', fontSize: 11, fontFamily: 'Space Mono', transform: 'rotate(-20deg)' }}
                  angle={-20}
                  padding={{ left: 12, right: 12 }}
                />
                <YAxis
                  domain={[processed.yMin, processed.yMax]}
                  tick={{ fill: '#555', fontSize: 11, fontFamily: 'Space Mono' }}
                  width={64}
                />
                <Tooltip content={<CustomTooltip />} />

                {/* Confidence interval rendered as a band: bottom baseline + diff area stacked */}
                <Area dataKey="ci_lower" stroke="none" fillOpacity={0} stackId="ci" />
                <Area dataKey="ci_diff" stroke="none" fill="#4D7CFF" fillOpacity={0.12} stackId="ci" />

                {/* Primary Actual Line (prominent) */}
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#0f172a"
                  strokeWidth={3}
                  dot={renderDot}
                  activeDot={{ r: 6 }}
                  name="Actual (Lab)"
                  isAnimationActive={false}
                  connectNulls
                />

                {/* Predicted dashed line (subtle, not obstructed by CI) */}
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="#4D7CFF"
                  strokeWidth={2.2}
                  strokeDasharray="6 4"
                  dot={false}
                  name="Predicted (ML)"
                  isAnimationActive={false}
                  connectNulls
                />

                {/* Specification / target lines */}
                <ReferenceLine y={FLASH_POINT_MIN} stroke="#FF5252" strokeDasharray="4 4" label={{ value: `Spec Min ${FLASH_POINT_MIN}°C`, position: 'insideBottomLeft', fill: '#FF5252', fontFamily: 'Space Mono', fontSize: 11 }} />
                <ReferenceLine y={FLASH_POINT_MAX} stroke="#FF5252" strokeDasharray="4 4" label={{ value: `Spec Max ${FLASH_POINT_MAX}°C`, position: 'insideTopLeft', fill: '#FF5252', fontFamily: 'Space Mono', fontSize: 11 }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className="responsive-grid-half" style={{ marginTop: 24 }}>
            {bestModel && (
              <div className="card">
                <div className="card-title">Best Model Performance</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                  {[
                    { label: 'Test RMSE', value: bestModel.test_rmse, max: 5,   good: 2 },
                    { label: 'Test MAE',  value: bestModel.test_mae,  max: 4,   good: 1.5 },
                    { label: 'Test R²',   value: bestModel.test_r2,   max: 1,   good: 0.85, invert: true },
                    { label: 'MAPE %',    value: bestModel.test_mape, max: 10,  good: 5 },
                  ].map(m => (
                    <div key={m.label}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6, fontFamily: 'Space Mono, monospace' }}>
                        <span style={{ color: '#555', fontWeight: 700, textTransform: 'uppercase', fontSize: 10 }}>{m.label}</span>
                        <span style={{ fontWeight: 700, color: (m.invert ? m.value >= m.good : m.value <= m.good) ? '#00E676' : '#FF8A50' }}>{m.value}</span>
                      </div>
                      <div className="metric-bar-bg">
                        <div className="metric-bar-fill" style={{ width: `${Math.min((m.value / m.max) * 100, 100)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="card">
              <div className="card-title">About This Project</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: 13, color: '#1a1a1a' }}>
                <InfoRow icon="🏭" text="IOCL Crude Distillation Unit — HY Kero product" />
                <InfoRow icon="📡" text="41 DCS sensor tags sampled every 15 minutes" />
                <InfoRow icon="⏱️" text="±45 min window around lab sample time (6AM / 2PM / 10PM)" />
                <InfoRow icon="🎯" text="Target: Flash GC (°C) — 3× per day via lab analysis" />
                <InfoRow icon="📅" text="Training data: Apr 2025 – Mar 2026 (~730 labeled shifts)" />
                <InfoRow icon="🧬" text="117 engineered features — lags, physics-derived, temporal" />
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
});

export default DashboardPage;

function InfoRow({ icon, text }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontFamily: 'Space Mono, monospace', fontSize: 12 }}>
      <span style={{ fontSize: 16 }}>{icon}</span>
      <span>{text}</span>
    </div>
  );
}
