import { useState, useEffect, useCallback, memo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
  BarChart, Bar, Cell,
} from 'recharts';
import { fetchHistory, fetchHistoryStats } from '../api';
import BrutTooltip from '../components/BrutTooltip';
import { HISTORY_PAGE_LIMIT, RESIDUAL_HIGHLIGHT_THRESHOLD } from '../constants';

const HistoryPage = memo(function HistoryPage() {
  const [data, setData]       = useState([]);
  const [stats, setStats]     = useState(null);
  const [shift, setShift]     = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [tab, setTab]         = useState('chart');
  const [loading, setLoading] = useState(true);

  const [debouncedStartDate, setDebouncedStartDate] = useState(startDate);
  const [debouncedEndDate, setDebouncedEndDate] = useState(endDate);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedStartDate(startDate);
    }, 300);
    return () => clearTimeout(handler);
  }, [startDate]);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedEndDate(endDate);
    }, 300);
    return () => clearTimeout(handler);
  }, [endDate]);

  const load = useCallback(async (s, start, end) => {
    setLoading(true);
    try {
      const [h, st] = await Promise.all([
        fetchHistory(s, start, end, HISTORY_PAGE_LIMIT),
        fetchHistoryStats()
      ]);
      const rows = (h.data || []).reverse().map(r => ({
        ...r,
        time: new Date(r.sample_ts).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }),
        actual:    Number(r.actual?.toFixed(2)),
        predicted: Number(r.predicted?.toFixed(2)),
        residual:  Number(r.residual?.toFixed(2)),
      }));
      setData(rows);
      setStats(st);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load(shift, debouncedStartDate, debouncedEndDate);
  }, [shift, debouncedStartDate, debouncedEndDate, load]);

  const handleExportCSV = () => {
    if (!data.length) return;
    const headers = ['Timestamp', 'Shift', 'Actual', 'Predicted', 'Residual'];
    const rows = data.map(r => [
      r.sample_ts,
      r.shift,
      r.actual,
      r.predicted,
      r.residual
    ]);
    
    // Construct CSV file format content
    const csvRows = [headers.join(','), ...rows.map(e => e.join(','))];
    const csvContent = csvRows.join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `hykero_predictions_history.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="section-header">
        <div className="section-icon" style={{ background: '#00E5FF' }}>📈</div>
        <div>
          <h2>Prediction History</h2>
          <p>Actual vs Predicted Flash GC over time with filtering and export options</p>
        </div>
      </div>

      {stats && (
        <div className="stats-grid" style={{ marginBottom: 20 }}>
          {[
            { label: 'Total Predictions', value: stats.total_predictions, color: '#4D7CFF' },
            { label: 'Actual Mean',        value: `${stats.actual_mean}°C`,  color: '#00E676' },
            { label: 'Actual Range',       value: `${stats.actual_min}–${stats.actual_max}°C`, color: '#FF8A50' },
            { label: 'Avg Residual (Bias)', value: `${stats.residual_mean}°C`, color: stats.residual_mean > 0.5 ? '#FF5252' : '#00E676' },
          ].map(s => (
            <div className="stat-card" key={s.label} style={{ '--accent-color': s.color }}>
              <div className="stat-label">{s.label}</div>
              <div className="stat-value" style={{ fontSize: '1.6rem' }}>{s.value}</div>
            </div>
          ))}
        </div>
      )}

      {/* Filter and Export Bar */}
      <div className="card" style={{ padding: 16, marginBottom: 20 }}>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
          
          {/* Shift selector */}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span style={{ fontSize: 11, fontWeight: 'bold', fontFamily: 'Space Mono' }}>SHIFT:</span>
            <div className="tabs" style={{ margin: 0, padding: 0 }}>
              {[['', 'All'], ['M', 'M'], ['E', 'E'], ['N', 'N']].map(([v, l]) => (
                <button key={v} className={`tab ${shift === v ? 'active' : ''}`}
                  onClick={() => setShift(v)} style={{ cursor: 'pointer', padding: '6px 12px', fontSize: 11 }}>{l}
                </button>
              ))}
            </div>
          </div>

          {/* Date Picker Start */}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span style={{ fontSize: 11, fontWeight: 'bold', fontFamily: 'Space Mono' }}>FROM:</span>
            <input 
              className="form-input" 
              type="date" 
              value={startDate} 
              onChange={e => setStartDate(e.target.value)} 
              style={{ padding: '6px 10px', width: 140, margin: 0 }} 
            />
          </div>

          {/* Date Picker End */}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span style={{ fontSize: 11, fontWeight: 'bold', fontFamily: 'Space Mono' }}>TO:</span>
            <input 
              className="form-input" 
              type="date" 
              value={endDate} 
              onChange={e => setEndDate(e.target.value)} 
              style={{ padding: '6px 10px', width: 140, margin: 0 }} 
            />
          </div>

          {/* Clear Dates */}
          {(startDate || endDate) && (
            <button className="btn" onClick={() => { setStartDate(''); setEndDate(''); }} style={{ padding: '8px 12px', fontSize: 11 }}>
              ❌ CLEAR DATES
            </button>
          )}

          {/* Export CSV */}
          <button className="btn btn-primary" onClick={handleExportCSV} disabled={!data.length} style={{ padding: '8px 16px', fontSize: 11, marginLeft: 'auto' }}>
            📥 EXPORT CSV
          </button>
        </div>
      </div>

      {/* View tabs */}
      <div className="tabs" style={{ marginBottom: 16 }}>
        {['chart', 'residual', 'table'].map(t => (
          <button key={t} className={`tab ${tab === t ? 'active' : ''}`}
            onClick={() => setTab(t)} style={{ cursor: 'pointer', textTransform: 'capitalize' }}>{t}
          </button>
        ))}
      </div>

      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: 60 }}>
          <div className="spinner" />
        </div>
      ) : !data.length ? (
        <div className="card" style={{ textAlign: 'center', padding: 30 }}>
          <div style={{ fontSize: 32, marginBottom: 8 }}>📭</div>
          <div className="card-title" style={{ marginBottom: 6 }}>No history available yet</div>
          <p style={{ fontFamily: 'Space Mono, monospace', fontSize: 12, color: '#5e534d' }}>
            Adjust the shift or date filters, or run a prediction to populate this view.
          </p>
        </div>
      ) : tab === 'chart' ? (
        <div className="chart-container">
          <div className="chart-header">
            <span className="chart-title">Actual vs Predicted Flash GC</span>
            <span style={{ fontSize: 11, color: '#555', fontFamily: 'Space Mono, monospace', fontWeight: 700 }}>{data.length} pts</span>
          </div>
          <ResponsiveContainer width="100%" height={380}>
            <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(26,26,26,.1)" />
              <XAxis dataKey="time" tick={{ fill: '#555', fontSize: 10, fontFamily: 'Space Mono' }} interval="preserveStartEnd" />
              <YAxis domain={['auto', 'auto']} tick={{ fill: '#555', fontSize: 11, fontFamily: 'Space Mono' }}
                label={{ value: 'Flash GC (°C)', angle: -90, position: 'insideLeft', fill: '#555', fontSize: 11 }} />
              <Tooltip content={<BrutTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'Space Mono, monospace', fontWeight: 700 }} />
              <ReferenceLine y={63} stroke="#FF5252" strokeDasharray="4 4" label={{ value: 'Min Spec (63°C)', fill: '#FF5252', fontSize: 10, position: 'top' }} />
              <Line type="monotone" dataKey="actual"    stroke="#1a1a1a" strokeWidth={2.5} dot={false} name="Actual" />
              <Line type="monotone" dataKey="predicted" stroke="#4D7CFF" strokeWidth={2.5} dot={false} strokeDasharray="6 3" name="Predicted" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : tab === 'residual' ? (
        <div className="chart-container">
          <div className="chart-header">
            <span className="chart-title">Prediction Residuals (Actual − Predicted)</span>
          </div>
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={data.slice(-100)} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(26,26,26,.1)" />
              <XAxis dataKey="time" tick={{ fill: '#555', fontSize: 10, fontFamily: 'Space Mono' }} interval={9} />
              <YAxis tick={{ fill: '#555', fontSize: 11, fontFamily: 'Space Mono' }}
                label={{ value: 'Residual (°C)', angle: -90, position: 'insideLeft', fill: '#555', fontSize: 11 }} />
              <Tooltip content={<BrutTooltip />} />
              <ReferenceLine y={0} stroke="#1a1a1a" strokeWidth={2} />
              <Bar dataKey="residual" name="Residual" radius={0}>
                {data.slice(-100).map((entry, i) => (
                  <Cell key={i} fill={entry.residual > 0 ? '#4D7CFF' : '#FF5252'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Date / Time</th>
                <th>Shift</th>
                <th>Actual (°C)</th>
                <th>Predicted (°C)</th>
                <th>Residual</th>
              </tr>
            </thead>
            <tbody>
              {data.slice().reverse().slice(0, 100).map((r, i) => (
                <tr key={i}>
                  <td>{new Date(r.sample_ts).toLocaleString('en-IN')}</td>
                  <td><span className={`shift-badge shift-${r.shift}`}>{r.shift}</span></td>
                  <td className="highlight">{r.actual}</td>
                  <td className="highlight">{r.predicted}</td>
                  <td style={{ color: Math.abs(r.residual) > RESIDUAL_HIGHLIGHT_THRESHOLD ? '#FF5252' : '#00E676', fontWeight: 700 }}>
                    {r.residual > 0 ? '+' : ''}{r.residual}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
});

export default HistoryPage;
