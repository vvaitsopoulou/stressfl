/**
 * Vision UI theme — pruned.
 *
 * Difference from the original Vision UI distribution:
 *   - We KEEP all 6 base files (colors, breakpoints, typography,
 *     boxShadows, borders, globals) and all helper functions.
 *   - We DROP the ~40 MUI component overrides (sidenav, card, button,
 *     appBar, tabs, stepper, ...). Vui components reference the theme
 *     directly via `theme.palette` etc., so they don't need MUI's
 *     <Card> or <Button> to be visually overridden — we already style
 *     our own panel cards manually.
 *
 * One important fix: Vision UI's `text` palette key has shape
 * `{ main, focus }` for Vui components, but MUI's default expects
 * `{ primary, secondary, disabled }`. We merge both shapes so plain
 * MUI components (like the Tooltip, Chip, IconButton in our panels)
 * still get readable text colour.
 */

import { createTheme } from '@mui/material/styles';

import colors from 'assets/theme/base/colors';
import breakpoints from 'assets/theme/base/breakpoints';
import typography from 'assets/theme/base/typography';
import boxShadows from 'assets/theme/base/boxShadows';
import borders from 'assets/theme/base/borders';
import globals from 'assets/theme/base/globals';

import boxShadow from 'assets/theme/functions/boxShadow';
import linearGradient from 'assets/theme/functions/linearGradient';
import pxToRem from 'assets/theme/functions/pxToRem';
import rgba from 'assets/theme/functions/rgba';

// Build the palette: spread Vision UI's keys, then overlay an
// MUI-shape `text` key so default MUI components stay readable.
const palette = {
  ...colors,
  // mode: 'dark' tells MUI to use dark defaults for unstyled bits
  mode: 'dark',
  text: {
    // Vision UI shape (used by VuiTypography color="text")
    ...colors.text,
    // MUI shape (used by Tooltip, Chip, etc.)
    primary: '#e8eaf3',
    secondary: colors.text.main, // #a0aec0
    disabled: 'rgba(232, 234, 243, 0.5)',
  },
  background: {
    ...colors.background,
    paper: '#0f1535',     // for MUI's Paper / Card defaults
    default: '#030c1d',   // matches dark.body
  },
};

export default createTheme({
  breakpoints: { ...breakpoints },
  palette,
  typography: { ...typography },
  boxShadows: { ...boxShadows },
  borders: { ...borders },
  functions: {
    boxShadow,
    linearGradient,
    pxToRem,
    rgba,
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        ...globals,
      },
    },
  },
});
