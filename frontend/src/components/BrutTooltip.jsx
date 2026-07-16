export default function BrutTooltip({ active, payload, label, showLabel = true, unit = '°C' }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#1a1a1a', border: '3px solid #1a1a1a', padding: '10px 14px', fontSize: 12, fontFamily: 'Space Mono, monospace', color: '#fff' }}>
      {showLabel && label && (
        <p style={{ color: '#FFE500', marginBottom: 4, fontWeight: 700, textTransform: 'uppercase', fontSize: 10 }}>
          {label}
        </p>
      )}
      {payload.map(p => {
        const valColor = p.color || p.stroke || '#FFE500';
        let formattedValue = '';
        if (Array.isArray(p.value)) {
          formattedValue = `[${p.value[0].toFixed(2)} – ${p.value[1].toFixed(2)}]`;
        } else if (typeof p.value === 'number') {
          formattedValue = p.value.toFixed(2);
        } else {
          formattedValue = p.value;
        }
        return (
          <p key={p.name} style={{ color: valColor, fontWeight: 700 }}>
            {p.name}: {formattedValue}{unit}
          </p>
        );
      })}
    </div>
  );
}
