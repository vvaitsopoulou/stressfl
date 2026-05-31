import { forwardRef } from 'react';
import VuiBoxRoot from 'components/VuiBox/VuiBoxRoot';

const VuiBox = forwardRef(
  (
    {
      variant = 'contained',
      bgColor = 'transparent',
      color = 'dark',
      opacity = 1,
      borderRadius = 'none',
      shadow = 'none',
      ...rest
    },
    ref
  ) => (
    <VuiBoxRoot
      {...rest}
      ref={ref}
      ownerState={{ variant, bgColor, color, opacity, borderRadius, shadow }}
    />
  )
);

// Defaults moved inline.

export default VuiBox;
