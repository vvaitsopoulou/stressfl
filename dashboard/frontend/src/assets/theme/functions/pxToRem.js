/**
 * Convert px to rem. Vision UI uses this everywhere — typography sizes,
 * border radii, sidenav widths, etc. Default base is 16 (browser default).
 */
function pxToRem(number, baseNumber = 16) {
  return `${number / baseNumber}rem`;
}

export default pxToRem;
