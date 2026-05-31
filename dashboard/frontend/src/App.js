/**
 * App — top-level component: theme + sidenav + routes.
 *
 * Differences from Vision UI's App.js:
 *   - No RTL plugin or rtl theme. (We don't need RTL.)
 *   - No Configurator. (Settings panel is dev marketing, not useful.)
 *   - No "settings" floating button. (Same.)
 *   - Default route redirects /  →  /live  (was /dashboard).
 */

import { useState, useEffect } from 'react';
import { Route, Switch, Redirect, useLocation } from 'react-router-dom';

import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

import Sidenav from 'examples/Sidenav';
import theme from 'assets/theme';
import routes from 'routes';

import { useVisionUIController, setMiniSidenav } from 'context';

export default function App() {
  const [controller, dispatch] = useVisionUIController();
  const { miniSidenav, layout, sidenavColor } = controller;
  const [onMouseEnter, setOnMouseEnter] = useState(false);
  const { pathname } = useLocation();

  // Mouse-over expansion of mini sidenav.
  const handleOnMouseEnter = () => {
    if (miniSidenav && !onMouseEnter) {
      setMiniSidenav(dispatch, false);
      setOnMouseEnter(true);
    }
  };
  const handleOnMouseLeave = () => {
    if (onMouseEnter) {
      setMiniSidenav(dispatch, true);
      setOnMouseEnter(false);
    }
  };

  // Reset scroll on route change.
  useEffect(() => {
    document.documentElement.scrollTop = 0;
    document.scrollingElement.scrollTop = 0;
  }, [pathname]);

  const getRoutes = (allRoutes) =>
    allRoutes.map((route) => {
      if (route.collapse) return getRoutes(route.collapse);
      if (route.route) {
        return (
          <Route exact path={route.route} component={route.component} key={route.key} />
        );
      }
      return null;
    });

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {layout === 'dashboard' && (
        <Sidenav
          color={sidenavColor}
          brandName="STRESS DEMO"
          routes={routes}
          onMouseEnter={handleOnMouseEnter}
          onMouseLeave={handleOnMouseLeave}
        />
      )}
      <Switch>
        {getRoutes(routes)}
        <Redirect from="*" to="/live" />
      </Switch>
    </ThemeProvider>
  );
}
