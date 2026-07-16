import { FLASH_POINT_MIN, FLASH_POINT_MAX, FLASH_POINT_WARNING } from './constants';

/**
 * Determines the flash point status based on spec limits.
 * @param {number} value - The flash point value
 * @returns {{ status: string, color: string, emoji: string, label: string }}
 */
export function getFlashPointStatus(value) {
  const num = parseFloat(value);
  if (isNaN(num)) return { status: 'unknown', color: '#888', emoji: '❓', label: 'Unknown' };
  if (num < FLASH_POINT_MIN) return { status: 'danger', color: '#ef4444', emoji: '🔴', label: 'Below Spec' };
  if (num > FLASH_POINT_MAX) return { status: 'danger', color: '#ef4444', emoji: '🔴', label: 'Above Spec' };
  if (num > FLASH_POINT_WARNING) return { status: 'warning', color: '#f59e0b', emoji: '🟡', label: 'Near Limit' };
  return { status: 'normal', color: '#22c55e', emoji: '🟢', label: 'In Spec' };
}
