/**
 * Sidenav — Vision UI sidenav, with the marketing accoutrements cut.
 *
 * Differences from the original Vision UI distribution:
 *
 *   - SidenavCard ("Need help?" call-to-action) removed.
 *   - "Upgrade to PRO" button removed.
 *   - SimmmpleLogo replaced with brand text only.
 */

import { useEffect } from 'react';
import { useLocation, NavLink } from 'react-router-dom';

import List from '@mui/material/List';
import Divider from '@mui/material/Divider';
import Link from '@mui/material/Link';
import Icon from '@mui/material/Icon';

import VuiBox from 'components/VuiBox';
import VuiTypography from 'components/VuiTypography';

import SidenavCollapse from 'examples/Sidenav/SidenavCollapse';
import SidenavRoot from 'examples/Sidenav/SidenavRoot';

import { useVisionUIController, setMiniSidenav, setTransparentSidenav } from 'context';

function Sidenav({ color = 'info', brandName, routes, ...rest }) {
  const [controller, dispatch] = useVisionUIController();
  const { miniSidenav, transparentSidenav } = controller;
  const location = useLocation();
  const { pathname } = location;
  const collapseName = pathname.split('/').slice(1)[0];

  const closeSidenav = () => setMiniSidenav(dispatch, true);

  useEffect(() => {
    function handleMiniSidenav() {
      setMiniSidenav(dispatch, window.innerWidth < 1200);
    }
    window.addEventListener('resize', handleMiniSidenav);
    handleMiniSidenav();
    return () => window.removeEventListener('resize', handleMiniSidenav);
  }, [dispatch, location]);

  useEffect(() => {
    if (window.innerWidth < 1440) {
      setTransparentSidenav(dispatch, false);
    }
  }, [dispatch]);

  // Render the nav items from routes.js
  const renderRoutes = routes.map(({ type, name, icon, title, noCollapse, key, route, href }) => {
    if (type === 'collapse') {
      return href ? (
        <Link
          href={href}
          key={key}
          target="_blank"
          rel="noreferrer"
          sx={{ textDecoration: 'none' }}
        >
          <SidenavCollapse
            color={color}
            name={name}
            icon={icon}
            active={key === collapseName}
            noCollapse={noCollapse}
          />
        </Link>
      ) : (
        <NavLink to={route} key={key}>
          <SidenavCollapse
            color={color}
            key={key}
            name={name}
            icon={icon}
            active={key === collapseName}
            noCollapse={noCollapse}
          />
        </NavLink>
      );
    }
    if (type === 'title') {
      return (
        <VuiTypography
          key={key}
          color="white"
          display="block"
          variant="caption"
          fontWeight="bold"
          textTransform="uppercase"
          pl={3}
          mt={2}
          mb={1}
          ml={1}
        >
          {title}
        </VuiTypography>
      );
    }
    if (type === 'divider') {
      return <Divider light key={key} />;
    }
    return null;
  });

  return (
    <SidenavRoot {...rest} variant="permanent" ownerState={{ transparentSidenav, miniSidenav }}>
      <VuiBox
        pt={3.5}
        pb={0.5}
        px={4}
        textAlign="center"
        sx={{ overflow: 'unset !important' }}
      >
        {/* Close button visible only on screens narrower than xl */}
        <VuiBox
          display={{ xs: 'block', xl: 'none' }}
          position="absolute"
          top={0}
          right={0}
          p={1.625}
          onClick={closeSidenav}
          sx={{ cursor: 'pointer' }}
        >
          <VuiTypography variant="h6" color="text">
            <Icon sx={{ fontWeight: 'bold' }}>close</Icon>
          </VuiTypography>
        </VuiBox>
        {/* Brand */}
        <VuiBox component={NavLink} to="/" display="flex" alignItems="center" justifyContent="center">
          <VuiTypography
            variant="button"
            textGradient
            color="logo"
            fontSize={14}
            letterSpacing={2}
            fontWeight="medium"
            sx={{
              opacity: miniSidenav || (miniSidenav && transparentSidenav) ? 0 : 1,
              maxWidth: miniSidenav || (miniSidenav && transparentSidenav) ? 0 : '100%',
              margin: '0 auto',
            }}
          >
            {brandName}
          </VuiTypography>
        </VuiBox>
      </VuiBox>
      <Divider light />
      <List>{renderRoutes}</List>
    </SidenavRoot>
  );
}

export default Sidenav;
