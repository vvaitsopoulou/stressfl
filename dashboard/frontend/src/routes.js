/**
 * Routes for the Stress Demo dashboard.
 *
 * Each route follows the Vision UI shape:
 *   { type: "collapse"|"title"|"divider", name, key, route, icon, ... }
 */

import { IoPulseOutline, IoSwapHorizontalOutline, IoLibraryOutline, IoGitNetworkOutline } from 'react-icons/io5';

import LivePanel from 'panels/Live';
import ComparisonPanel from 'panels/Comparison';
import EvidencePanel from 'panels/Evidence';
import PipelinePanel from 'panels/Pipeline';

const routes = [
  {
    type: 'collapse',
    name: 'Live',
    key: 'live',
    route: '/live',
    icon: <IoPulseOutline size="15px" color="inherit" />,
    component: LivePanel,
    noCollapse: true,
  },
  {
    type: 'collapse',
    name: 'Comparison',
    key: 'comparison',
    route: '/comparison',
    icon: <IoSwapHorizontalOutline size="15px" color="inherit" />,
    component: ComparisonPanel,
    noCollapse: true,
  },
  {
    type: 'collapse',
    name: 'Evidence',
    key: 'evidence',
    route: '/evidence',
    icon: <IoLibraryOutline size="15px" color="inherit" />,
    component: EvidencePanel,
    noCollapse: true,
  },
  {
    type: 'collapse',
    name: 'Pipeline',
    key: 'pipeline',
    route: '/pipeline',
    icon: <IoGitNetworkOutline size="15px" color="inherit" />,
    component: PipelinePanel,
    noCollapse: true,
  },
];

export default routes;
