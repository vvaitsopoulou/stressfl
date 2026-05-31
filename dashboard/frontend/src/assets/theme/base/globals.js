import colors from 'assets/theme/base/colors';
import bgAdmin from 'assets/images/body-background.png';

const { info, dark } = colors;

export default {
  html: {
    scrollBehavior: 'smooth',
    // Layered backdrop:
    //   1. Three CSS radial glows on TOP — violet top-left, cyan
    //      right, soft purple bottom — give the page colour that
    //      bleeds through the translucent cards.
    //   2. The body-background image BENEATH the glows — adds the
    //      Vision UI cloud texture. Anchored at center bottom so the
    //      brightest part sits below the content.
    //   3. The dark base #030c1d at the bottom of the stack catches
    //      any uncovered pixels.
    // backgroundAttachment: fixed on every layer so the entire stack
    // stays anchored to the viewport while content scrolls.
    background: [
      'radial-gradient(ellipse 80% 60% at 25% 15%, rgba(67, 24, 255, 0.50), transparent 65%)',
      'radial-gradient(ellipse 55% 70% at 95% 55%, rgba(33, 212, 253, 0.28), transparent 68%)',
      'radial-gradient(ellipse 70% 45% at 55% 100%, rgba(159, 122, 234, 0.22), transparent 70%)',
      `url(${bgAdmin}) no-repeat center bottom / cover`,
      dark.body,
    ].join(', '),
    backgroundAttachment: 'fixed, fixed, fixed, fixed, fixed',
    minHeight: '100vh',
  },
  body: {
    background: 'transparent',
    minHeight: '100vh',
  },
  '*, *::before, *::after': {
    margin: 0,
    padding: 0,
  },
  'a, a:link, a:visited': {
    textDecoration: 'none !important',
  },
  'a.link, .link, a.link:link, .link:link, a.link:visited, .link:visited': {
    color: `${dark.main} !important`,
    transition: 'color 150ms ease-in !important',
  },
  'a.link:hover, .link:hover, a.link:focus, .link:focus': {
    color: `${info.main} !important`,
  },

  // Heartbeat animation for the HR panel's pulsing heart icon. The
  // animation duration is set inline by the component (60 / bpm
  // seconds), so this keyframe just defines the shape of one beat:
  // a tight scale-up at the start (the "thump") and a gentler scale
  // back down (the "release"). Mimics the asymmetric profile of a
  // real cardiac contraction — quick squeeze, slow relax.
  '@keyframes heartbeat': {
    '0%':   { transform: 'scale(1)' },
    '14%':  { transform: 'scale(1.18)' },   // first thump
    '28%':  { transform: 'scale(1)' },
    '42%':  { transform: 'scale(1.10)' },   // smaller second thump (lub-dub)
    '70%':  { transform: 'scale(1)' },
    '100%': { transform: 'scale(1)' },
  },
};
