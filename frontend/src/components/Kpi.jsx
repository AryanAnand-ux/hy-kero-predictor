export default function Kpi({ label, value, sub, color = '#FFE500', icon }) {
  return (
    <div className="stat-card" style={{ '--accent-color': color }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div className="stat-label">{label}</div>
          <div className="stat-value">{value}</div>
          {sub && <div className="stat-sub">{sub}</div>}
        </div>
        {icon && <span style={{ fontSize: 28 }}>{icon}</span>}
      </div>
    </div>
  );
}
