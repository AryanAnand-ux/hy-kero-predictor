import { useState, memo } from 'react';
import { uploadPredictBatch } from '../api';
import { getFlashPointStatus } from '../utils';

const UploadPage = memo(function UploadPage() {
  const [file, setFile]       = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState(null);
  const [error, setError]     = useState('');

  const handleDrop = (e) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true); setError(''); setResult(null);
    try {
      const data = await uploadPredictBatch(file);
      if (data.status === 'success') {
        setResult(data);
        // Dispatch custom event to notify parent App of batch prediction
        window.dispatchEvent(new CustomEvent('batch-prediction', {
          detail: data
        }));
      } else {
        setError(data.detail || 'Upload failed');
      }
    } catch (e) {
      setError(e.message || 'Cannot connect to backend.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setResult(null);
    setError('');
  };

  const handleDownloadCSV = () => {
    if (!result || !result.data || !result.data.length) return;
    
    // Extract headers
    const headers = Object.keys(result.data[0]);
    const csvContent = [
      headers.join(','),
      ...result.data.map(row => 
        headers.map(h => {
          const val = row[h];
          if (typeof val === 'string' && (val.includes(',') || val.includes('\n') || val.includes('"'))) {
            return `"${val.replace(/"/g, '""')}"`;
          }
          return val === null || val === undefined ? '' : val;
        }).join(',')
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    
    // Set appropriate output filename
    const baseName = file ? file.name.replace(/\.[^/.]+$/, "") : "batch";
    link.setAttribute("download", `predicted_${baseName}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const downloadSampleTemplate = (e) => {
    e.preventDefault();
    const headers = [
      'sample_ts', 'shift',
      'MF_Bottom_T_mean', 'SS_11C5_mean', 'CDU_Draw_LN_MF_F_mean', 'feat_HK_LN_draw_ratio',
      'CDU_Draw_RCO_F_mean', 'MF_HK_Draw_T_mean', 'Outlet_temp_11F1_mean', 'feat_HK_strip_eff',
      'Outlet_temp_11F3_mean', 'MF_LGO_CR_Ret_T_mean', 'feat_SS_per_HK_flow', 'lag1_MF_HK_CR_Ret_T_mean',
      'time_index', 'SS_11C4_mean', 'lag1_MF_LGO_CR_Ret_T_mean', 'MF_Reflux_T_mean',
      'CDU_Draw_HK_F_mean', 'month_cos', 'CDU_Draw_LN_Stab_bottom_F_mean', 'lag1_MF_Top_T_mean',
      'lag1_Outlet_temp_11F3_mean', 'MF_LGO_Draw_T_mean', 'MF_Reflux_ATF_CR_F_mean',
      'CDU_Draw_LPG_Stab_top_F_mean', 'lag1_MF_HN_Draw_T_mean', 'SS_11C3_mean',
      'MF_HGO_Draw_T_mean', 'CDU_Draw_LGO_F_mean', 'lag1_MF_ATF_Draw_T_mean', 'MF_ATF_CR_Ret_T_mean'
    ];
    
    const rows = [
      [
        '2026-07-08 06:00:00', 'M',
        '351.84', '2998.46', '94.34', '0.95',
        '501.62', '228.77', '366.03', '-134.84',
        '367.87', '219.99', '33.30', '160.75',
        '0.003', '4583.17', '213.61', '48.58',
        '90.04', '-0.50', '147.34', '150.93',
        '367.19', '294.31', '430.13', '60.79',
        '162.43', '0.71', '348.41', '288.43',
        '185.02', '129.98'
      ],
      [
        '2026-07-08 14:00:00', 'E',
        '352.05', '3001.70', '87.63', '1.03',
        '472.61', '230.07', '366.32', '-135.71',
        '368.20', '223.93', '33.33', '162.98',
        '0.004', '4581.98', '219.99', '56.44',
        '90.07', '-0.50', '154.17', '150.96',
        '367.87', '296.68', '430.28', '60.60',
        '156.62', '0.72', '353.82', '301.10',
        '181.07', '132.53'
      ]
    ];
    
    const csvContent = [
      headers.join(','),
      ...rows.map(r => r.join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", "sample_batch_input.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="section-header">
        <div className="section-icon" style={{ background: '#00E676' }}>📂</div>
        <div>
          <h2>Batch Prediction Upload</h2>
          <p>Upload a processed feature CSV or sensor Excel for batch Flash GC predictions</p>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-title">How it works</div>
        <ol style={{ paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 8, fontSize: 12, color: '#1a1a1a', fontFamily: 'Space Mono, monospace' }}>
          <li>Upload your sensor data or feature CSV (matching features inside <strong>data/processed/X_full.csv</strong>)</li>
          <li>Our FastAPI backend pre-processes the features and fills in any missing sensor tags</li>
          <li>Predictions are computed instantly using the regularized Lasso/ElasticNet production model</li>
          <li>Download the fully annotated CSV including 95% Confidence Interval limits</li>
          <li>Need a template? Download the <a href="#" onClick={downloadSampleTemplate} style={{ color: '#4D7CFF', textDecoration: 'underline', fontWeight: 700 }}>sample CSV template</a> to see the required column structure</li>
        </ol>
      </div>

      {!result && (
        <div
          className={`upload-zone ${dragging ? 'dragging' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input').click()}
          onKeyDown={e => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              document.getElementById('file-input').click();
            }
          }}
          role="button"
          tabIndex={0}
          style={{ marginBottom: 24 }}
        >
          <input id="file-input" type="file" accept=".xlsx,.csv"
            style={{ display: 'none' }}
            onChange={e => setFile(e.target.files[0])} />
          <div className="upload-icon">{file ? '📄' : '☁️'}</div>
          <div className="upload-text">
            {file ? file.name : 'Drop CSV / XLSX here or click to browse'}
          </div>
          <div className="upload-sub">
            {file ? `${(file.size / 1024).toFixed(1)} KB` : 'Supports .csv and .xlsx (Max 10MB)'}
          </div>
        </div>
      )}

      {file && !result && (
        <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
          <button className="btn btn-primary" onClick={handleSubmit} disabled={loading}
            style={{ flex: 2, justifyContent: 'center', padding: 14 }}>
            {loading
              ? <><div className="spinner" style={{ width: 18, height: 18 }} /> PROCESSING…</>
              : '🚀 RUN BATCH PREDICTION'}
          </button>
          <button className="btn" onClick={handleClear} disabled={loading}
            style={{ flex: 1, justifyContent: 'center', padding: 14 }}>
            CLEAR
          </button>
        </div>
      )}

      {error && (
        <div style={{ padding: 14, background: '#FF5252', border: '2px solid #1a1a1a', boxShadow: '3px 3px 0 #1a1a1a', color: '#fff', fontSize: 12, fontFamily: 'Space Mono, monospace', fontWeight: 700, marginBottom: 20 }}>
          ⚠️ {error}
        </div>
      )}

      {result && (
        <div className="card">
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20, flexWrap: 'wrap', gap: 12 }}>
            <div className="card-title" style={{ margin: 0 }}>Batch Prediction Summary</div>
            <div style={{ display: 'flex', gap: 12 }}>
              <button className="btn btn-primary" onClick={handleDownloadCSV} style={{ padding: '8px 16px', fontSize: 11 }}>
                📥 DOWNLOAD FULL CSV
              </button>
              <button className="btn" onClick={handleClear} style={{ padding: '8px 16px', fontSize: 11 }}>
                🔄 UPLOAD ANOTHER
              </button>
            </div>
          </div>

          <div style={{ display: 'flex', gap: 20, marginBottom: 20, flexWrap: 'wrap' }}>
            <div className="stat-card" style={{ '--accent-color': '#00E676', flex: 1, minWidth: '150px' }}>
              <div className="stat-label">Rows Processed</div>
              <div className="stat-value" style={{ fontSize: '1.8rem' }}>{result.rows_processed}</div>
            </div>
            <div className="stat-card" style={{ '--accent-color': '#4D7CFF', flex: 1, minWidth: '150px' }}>
              <div className="stat-label">Production Model</div>
              <div style={{ fontSize: 14, fontWeight: 800, color: '#1a1a1a', marginTop: 8, fontFamily: 'Space Mono, monospace' }}>{result.model_used}</div>
            </div>
          </div>

          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Row #</th>
                  <th>Predicted Flash Point (°C)</th>
                  <th>95% Confidence Interval</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {result.data.slice(0, 50).map((row, i) => {
                  const v = row.predicted_flash_point;
                  const low = row.confidence_lower;
                  const high = row.confidence_upper;
                  const status = getFlashPointStatus(v);
                  return (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td className="highlight">{v !== null && v !== undefined ? `${v}°C` : 'N/A'}</td>
                      <td style={{ fontFamily: 'Space Mono, monospace', fontSize: '11px' }}>
                        {low !== null && high !== null ? `[${low}°C to ${high}°C]` : 'N/A'}
                      </td>
                      <td style={{ color: status.color, fontWeight: 700 }}>
                        {status.status === 'danger' ? (status.label === 'Below Spec' ? '🚨 BELOW SPEC' : '🚨 ABOVE SPEC') : status.status === 'warning' ? '⚠️ HIGH' : '✅ IN SPEC'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {result.data.length > 50 && (
              <p style={{ padding: 12, color: '#555', fontSize: 11, textAlign: 'center', fontFamily: 'Space Mono, monospace', fontWeight: 700, textTransform: 'uppercase', borderTop: '2px solid #1a1a1a' }}>
                Showing first 50 of {result.data.length} rows
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
});

export default UploadPage;
