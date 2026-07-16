export default function Detail({ label, value, color }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '8px 0', borderBottom: '2px solid #1a1a1a', fontFamily: 'Space Mono, monospace' }}>
      <span style={{ color: '#555', fontWeight: 700, textTransform: 'uppercase', fontSize: 10 }}>{label}</span>
      <span style={{ color: color || '#1a1a1a', fontWeight: 800 }}>{value}</span>
    </div>
  );
}
