/**
 * Build a CSS linear-gradient. Vision UI uses this everywhere — for
 * sidenav background, card surfaces, gauges, etc. The colour and state
 * arguments are CSS colour strings (often including stops, e.g.
 * "rgba(6, 11, 40, 0.94) 19.41%"), and angle is a degree number.
 */
function linearGradient(color, colorState, angle) {
  if (angle === undefined) {
    angle = 310;
  }
  return `linear-gradient(${angle}deg, ${color}, ${colorState})`;
}

export default linearGradient;
