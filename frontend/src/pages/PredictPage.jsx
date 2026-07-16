import { useState, memo } from 'react';
import { predictFromSensors, predictFromWindow } from '../api';
import FlashBadge from '../components/FlashBadge';
import Detail from '../components/Detail';
import { FLASH_POINT_MIN, FLASH_POINT_MAX, FLASH_POINT_WARNING } from '../constants';

const SENSOR_DEFAULTS = {
  MF_HK_Draw_T:     '224',
  MF_FlashZone_T:   '365',
  MF_Top_T:         '151',
  Outlet_temp_11F1: '368',
  Outlet_temp_11F2: '367',
  Outlet_temp_11F3: '369',
  Outlet_temp_11F4: '368',
  CDU_Draw_HK_F:    '95',
  SS_11C5:          '5.0',
};

const SENSOR_LABELS = {
  MF_HK_Draw_T:     'HK Draw Temperature (°C)',
  MF_FlashZone_T:   'Flash Zone Temperature (°C)',
  MF_Top_T:         'Column Top Temperature (°C)',
  Outlet_temp_11F1: 'Furnace 11F1 Outlet Temp (°C)',
  Outlet_temp_11F2: 'Furnace 11F2 Outlet Temp (°C)',
  Outlet_temp_11F3: 'Furnace 11F3 Outlet Temp (°C)',
  Outlet_temp_11F4: 'Furnace 11F4 Outlet Temp (°C)',
  CDU_Draw_HK_F:    'HK Draw Flow Rate (m³/hr)',
  SS_11C5:          'Stripping Steam 11C5 (TPH)',
};

const PRESET_SCENARIOS = [
  {
    id: 'normal',
    label: 'Normal Run',
    description: 'Stable conditions near expected operating band',
    sensors: { MF_HK_Draw_T: '224', MF_FlashZone_T: '365', MF_Top_T: '151', Outlet_temp_11F1: '368', Outlet_temp_11F2: '367', Outlet_temp_11F3: '369', Outlet_temp_11F4: '368', CDU_Draw_HK_F: '95', SS_11C5: '5.0' },
    lags: [],
  },
  {
    id: 'watch',
    label: 'Watchlist',
    description: 'Mild drift with rising furnace duty',
    sensors: { MF_HK_Draw_T: '229', MF_FlashZone_T: '372', MF_Top_T: '153', Outlet_temp_11F1: '372', Outlet_temp_11F2: '371', Outlet_temp_11F3: '374', Outlet_temp_11F4: '373', CDU_Draw_HK_F: '101', SS_11C5: '5.8' },
    lags: [],
  },
  {
    id: 'risk',
    label: 'High Risk',
    description: 'Potential off-spec trend with elevated severity',
    sensors: { MF_HK_Draw_T: '236', MF_FlashZone_T: '381', MF_Top_T: '157', Outlet_temp_11F1: '378', Outlet_temp_11F2: '377', Outlet_temp_11F3: '380', Outlet_temp_11F4: '379', CDU_Draw_HK_F: '108', SS_11C5: '6.8' },
    lags: [],
  },
];

const PredictPage = memo(function PredictPage() {
  const [sensors, setSensors]   = useState(SENSOR_DEFAULTS);
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState('');
  const [validationMessage, setValidationMessage] = useState('');
  const [mode, setMode]         = useState('manual');
  const [timestamp, setTimestamp] = useState(() => {
    const d = new Date(); d.setMinutes(0, 0, 0);
    return d.toISOString().slice(0, 16);
  });

  const validateInputs = () => {
    if (mode === 'window' && !timestamp) {
      return 'Please choose a timestamp for the window-based prediction.';
    }

    return '';
  };

  const handlePredict = async () => {
    const validation = validateInputs();
    if (validation) {
      setValidationMessage(validation);
      setError('');
      setResult(null);
      return;
    }

    setLoading(true); setError(''); setValidationMessage(''); setResult(null);
    try {
      const parsedSensors = {};
      for (const [k, v] of Object.entries(sensors)) {
        parsedSensors[k] = v === '' ? 0.0 : parseFloat(v);
      }
      
      let res;
      if (mode === 'manual') {
        res = await predictFromSensors(parsedSensors);
      } else {
        res = await predictFromWindow(timestamp);
      }
      if (res.status === 'success' || res.predicted_flash_point) {
        setResult(res);
        // Dispatch custom event to notify parent App of new prediction
        window.dispatchEvent(new CustomEvent('new-prediction', {
          detail: {
            id: res.id,
            predicted: res.predicted_flash_point,
            sample_ts: res.sample_ts || new Date().toISOString(),
            shift: res.shift || 'M'
          }
        }));
      } else {
        setError(res.detail || 'Prediction failed. Make sure the backend is running and model is trained.');
      }
    } catch (e) {
      setError(e.message || 'Cannot connect to backend. Make sure FastAPI is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSensors(SENSOR_DEFAULTS);
    setResult(null);
    setError('');
    setValidationMessage('');
  };

  const applyPreset = (preset) => {
    setSensors(preset.sensors);
    setResult(null);
    setError('');
    setValidationMessage('');
  };

  const predictionStatus = result?.predicted_flash_point == null
    ? null
    : result.predicted_flash_point < FLASH_POINT_MIN
      ? { label: 'Below Spec (Critical)', tone: 'danger', note: 'Immediate operator review recommended.' }
      : result.predicted_flash_point > FLASH_POINT_MAX
        ? { label: 'Above Spec (Critical)', tone: 'danger', note: 'Flash point exceeds maximum safety limits.' }
        : result.predicted_flash_point > FLASH_POINT_WARNING
          ? { label: 'High (Warning)', tone: 'warning', note: 'Process drift may be pushing product out of range.' }
          : { label: 'In Spec', tone: 'normal', note: 'Current conditions look within expected operating bounds.' };

  return (
    <div>
      <div className="section-header">
        <div className="section-icon" style={{ background: '#FF6B9D' }}>🔥</div>
        <div>
          <h2>Flash Point Predictor</h2>
          <p>Enter process conditions to get an instant Flash GC estimate with confidence bounds</p>
        </div>
      </div>

      {/* Mode selector */}
      <div className="tabs" style={{ marginBottom: 24 }}>
        <button className={`tab ${mode === 'manual' ? 'active' : ''}`}
          onClick={() => setMode('manual')} style={{ cursor: 'pointer' }}>
          ✏️ Manual Input
        </button>
        <button className={`tab ${mode === 'window' ? 'active' : ''}`}
          onClick={() => setMode('window')} style={{ cursor: 'pointer' }}>
          🕐 Time Window
        </button>
      </div>

      <div className="helper-banner" style={{ marginBottom: 16 }}>
        <strong>Try a scenario:</strong> use the quick presets below to explore normal, watch, and high-risk operating conditions.
      </div>

      <div className="preset-grid" style={{ marginBottom: 18 }}>
        {PRESET_SCENARIOS.map((preset) => (
          <button key={preset.id} className="preset-pill" onClick={() => applyPreset(preset)}>
            <span className="preset-title">{preset.label}</span>
            <span className="preset-desc">{preset.description}</span>
          </button>
        ))}
      </div>

      <div className="responsive-grid-split">

        {/* Left: input form */}
        <div className="card">
          {mode === 'manual' ? (
            <>
              <div className="card-title">Process Sensor Values (±30 min window mean)</div>
              <div className="form-grid">
                {Object.entries(SENSOR_LABELS).map(([key, label]) => (
                  <div className="form-group" key={key}>
                    <label className="form-label">{label}</label>
                    <input
                      className="form-input"
                      type="number"
                      value={sensors[key] ?? ''}
                      onChange={e => setSensors(prev => ({ ...prev, [key]: e.target.value }))}
                    />
                  </div>
                ))}

              </div>
            </>
          ) : (
            <>
              <div className="card-title">Select Prediction Timestamp</div>
              <p style={{ fontSize: 12, color: '#555', marginBottom: 16, fontFamily: 'Space Mono, monospace' }}>
                The model will use sensor readings ±30 minutes around this time from the raw data file.
              </p>
              <div className="form-grid">
                <div className="form-group">
                  <label className="form-label">Date & Time</label>
                  <input className="form-input" type="datetime-local" value={timestamp}
                    onChange={e => setTimestamp(e.target.value)} />
                </div>

              </div>
              <div style={{ padding: '10px 14px', background: '#FFE500', border: '2px solid #1a1a1a', boxShadow: '3px 3px 0 #1a1a1a', marginTop: 8, fontSize: 11, color: '#1a1a1a', fontFamily: 'Space Mono, monospace', fontWeight: 600 }}>
                💡 SHIFT TIMES: Morning 06:00 | Evening 14:00 | Night 22:00
              </div>
            </>
          )}

          {validationMessage && (
            <div className="helper-banner" style={{ marginTop: 16, background: '#FFF3C4' }}>
              {validationMessage}
            </div>
          )}

          <div style={{ marginTop: 20, display: 'flex', gap: 12 }}>
            <button className="btn btn-primary" onClick={handlePredict} disabled={loading}
              style={{ flex: 2, justifyContent: 'center', padding: '14px' }}>
              {loading ? <><div className="spinner" style={{ width: 18, height: 18 }} /> PREDICTING…</> : '🔮 PREDICT'}
            </button>
            <button className="btn" onClick={handleReset} disabled={loading}
              style={{ flex: 1, justifyContent: 'center', padding: '14px' }}>
              🔄 RESET
            </button>
          </div>

          {error && (
            <div style={{ marginTop: 16, padding: 14, background: '#FF5252', border: '2px solid #1a1a1a', boxShadow: '3px 3px 0 #1a1a1a', color: '#fff', fontSize: 12, fontFamily: 'Space Mono, monospace', fontWeight: 700 }}>
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Right: result display */}
        <div>
          <div className="flash-point-hero">
            <div className="flash-point-label">Predicted Flash GC</div>
            <div className="flash-point-value">
              {result ? result.predicted_flash_point.toFixed(1) : '--'}
            </div>
            <div className="flash-point-unit">degrees Celsius</div>
            {result?.confidence_lower !== undefined && result?.confidence_upper !== undefined && (
              <div style={{ fontSize: '0.8rem', fontFamily: 'Space Mono', color: 'var(--text-mid)', marginTop: 6, fontWeight: 700 }}>
                95% CI: [{result.confidence_lower.toFixed(1)}°C – {result.confidence_upper.toFixed(1)}°C]
              </div>
            )}
            <div style={{ marginTop: 12 }}>
              <FlashBadge value={result?.predicted_flash_point} />
            </div>
            {predictionStatus && (
              <div className={`prediction-status ${predictionStatus.tone}`}>
                <strong>{predictionStatus.label}</strong>
                <span>{predictionStatus.note}</span>
              </div>
            )}
          </div>

          {result && (
            <div className="card">
              <div className="card-title">Prediction Details</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <Detail label="Model Used"    value={result.model_used} />
                {result.confidence_lower !== undefined && result.confidence_upper !== undefined && (
                  <Detail label="95% CI Range" value={`${result.confidence_lower.toFixed(1)}°C to ${result.confidence_upper.toFixed(1)}°C`} color="#4D7CFF" />
                )}
                {result.window_start && <Detail label="Window Start" value={new Date(result.window_start).toLocaleTimeString('en-IN')} />}
                {result.window_end   && <Detail label="Window End"   value={new Date(result.window_end).toLocaleTimeString('en-IN')} />}
                {result.readings_used && <Detail label="Readings Used" value={result.readings_used} />}
                <Detail label="Spec Minimum"  value={`${FLASH_POINT_MIN}°C`} />
                <Detail label="Spec Maximum"  value={`${FLASH_POINT_MAX}°C`} />
                <Detail label="Status"
                  value={result.predicted_flash_point < FLASH_POINT_MIN ? 'BELOW SPEC 🚨' : result.predicted_flash_point > FLASH_POINT_MAX ? 'ABOVE SPEC 🚨' : result.predicted_flash_point > FLASH_POINT_WARNING ? 'HIGH ⚠️' : 'IN SPEC ✅'}
                  color={result.predicted_flash_point < FLASH_POINT_MIN || result.predicted_flash_point > FLASH_POINT_MAX ? '#FF5252' : result.predicted_flash_point > FLASH_POINT_WARNING ? '#FF8A50' : '#00E676'} />
              </div>
            </div>
          )}

          {!result && (
            <div className="card" style={{ textAlign: 'center', padding: 40 }}>
              <div style={{ fontSize: 48, marginBottom: 12 }}>🧪</div>
              <p style={{ fontFamily: 'Space Mono, monospace', fontSize: 12, fontWeight: 700, textTransform: 'uppercase', color: '#555' }}>
                Enter process conditions<br />and click Predict
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

export default PredictPage;
