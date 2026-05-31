/**
 * DashboardNavbar minimal custom replacement for the Vision UI one.
 */

import { useLocation } from 'react-router-dom';
import VuiBox from 'components/VuiBox';
import VuiTypography from 'components/VuiTypography';
import Icon from '@mui/material/Icon';

function DashboardNavbar({ title = '', children = null }) {
  const { pathname } = useLocation();
  const segments = pathname.split('/').filter(Boolean);
  const currentSegment = segments[segments.length - 1] ?? 'dashboard';

  return (
    <VuiBox
      sx={{
        position: 'sticky',
        top: 0,
        zIndex: 10,
        py: 2,
        mb: 2,
        display: 'flex',
        flexDirection: { xs: 'column', md: 'row' },
        alignItems: { xs: 'flex-start', md: 'center' },
        justifyContent: 'space-between',
        gap: 1,
      }}
    >
      <VuiBox>
        {/* Breadcrumb-style path */}
        <VuiBox display="flex" alignItems="center" gap={0.5} mb={0.5}>
          <Icon sx={{ color: '#a0aec0', fontSize: '14px !important' }}>home</Icon>
          <VuiTypography variant="button" color="text" textTransform="capitalize">
            Dashboard
          </VuiTypography>
          <VuiTypography variant="button" color="text" sx={{ mx: 0.5 }}>/</VuiTypography>
          <VuiTypography variant="button" color="white" fontWeight="medium" textTransform="capitalize">
            {currentSegment}
          </VuiTypography>
        </VuiBox>
        {/* Page title */}
        <VuiTypography variant="h4" color="white" fontWeight="bold">
          {title}
        </VuiTypography>
      </VuiBox>
      {/* Right slot for panel-specific status */}
      <VuiBox>
        {children}
      </VuiBox>
    </VuiBox>
  );
}

// Defaults moved inline.

export default DashboardNavbar;
