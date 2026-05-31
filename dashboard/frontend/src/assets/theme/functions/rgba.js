/**
 * Convert a hex colour like "#0075ff" to "rgba(0, 117, 255, alpha)".
 *
 * This is a stand-in for Vision UI's `rgba` helper — we didn't get it
 * uploaded but it's a trivial utility. The Vision UI version probably
 * uses chroma-js; ours is dependency-free.
 *
 * Accepts: 3-digit (#abc) and 6-digit (#aabbcc) hex strings, or any
 * already-rgba/named-colour value (passes through unchanged).
 */
function rgba(color, opacity) {
  if (typeof color !== 'string') return color;
  let hex = color.trim();

  // Pass-through for non-hex inputs (already rgba, named, etc.)
  if (!hex.startsWith('#')) return color;

  hex = hex.slice(1);

  // Expand 3-digit shorthand to 6-digit.
  if (hex.length === 3) {
    hex = hex.split('').map((c) => c + c).join('');
  }
  if (hex.length !== 6) return color; // malformed; fall through

  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${opacity})`;
}

export default rgba;
