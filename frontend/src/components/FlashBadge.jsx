import { getFlashPointStatus } from '../utils';

export default function FlashBadge({ value }) {
  if (value === undefined || value === null) return null;
  const status = getFlashPointStatus(value);
  if (status.status === 'unknown') return null;

  let cls = 'badge-normal';
  let label = 'IN SPEC';

  if (status.status === 'danger') {
    cls = 'badge-danger';
    label = status.label === 'Below Spec' ? '🚨 BELOW SPEC' : '🚨 ABOVE SPEC';
  } else if (status.status === 'warning') {
    cls = 'badge-warning';
    label = '⚠️ HIGH';
  }

  return <span className={`flash-point-badge ${cls}`}>{label}</span>;
}
