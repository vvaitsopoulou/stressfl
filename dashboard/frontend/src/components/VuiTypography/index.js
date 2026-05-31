import { forwardRef } from 'react';
import VuiTypographyRoot from 'components/VuiTypography/VuiTypographyRoot';

const VuiTypography = forwardRef(
  (
    {
      color = 'dark',
      fontWeight = false,
      textTransform = 'none',
      verticalAlign = 'unset',
      fontSize = '16px',
      textGradient = false,
      opacity = 1,
      children,
      ...rest
    },
    ref
  ) => (
    <VuiTypographyRoot
      {...rest}
      ref={ref}
      ownerState={{
        color,
        textTransform,
        verticalAlign,
        fontSize,
        fontWeight,
        opacity,
        textGradient,
      }}
    >
      {children}
    </VuiTypographyRoot>
  )
);

// Defaults moved inline.

export default VuiTypography;
