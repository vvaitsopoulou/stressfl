# Stress Dashboard — Frontend

A routed single-page application that visualises real-time and offline stress-detection
results. The application is built with Vite, React, Material UI (MUI), and Recharts, and
adopts the visual layer of the Vision UI Free template. It comprises four panels — Live,
Comparison, Evidence, and Pipeline — accessed through a persistent side navigation. The
Vision UI side navigation, theme, and `Vui*` component wrappers are integrated into the
application shell.

## Directory layout

```
frontend/
├── index.html                          # Vite entry HTML
├── vite.config.js                      # aliases + JSX-in-.js setting
├── package.json
└── src/
    ├── main.jsx                        # createRoot + Router + Provider
    ├── App.js                          # ThemeProvider + Sidenav + routes
    ├── routes.js                       # 4-panel route table
    ├── protocol/index.js               # mirrors backend/protocol.py
    ├── hooks/
    │   ├── useWebSocket.js             # connection + reducer + ring buffers
    │   └── selectors.js                # buffer-shaping pure functions
    ├── panels/
    │   ├── Live/index.jsx              # Panel 1: live stress monitor
    │   ├── Comparison/index.jsx        # Panel 2: three-model comparison
    │   ├── Evidence/index.jsx          # Panel 3: offline empirical results
    │   └── Pipeline/index.jsx          # Panel 4: data-flow diagram
    ├── components/
    │   ├── VuiBox/                     # design-token Box wrapper
    │   └── VuiTypography/              # design-token Typography wrapper
    ├── context/index.js                # VisionUIController (sidenav state)
    ├── assets/
    │   ├── images/body-background.png  # gradient backdrop
    │   └── theme/                      # MUI theme (colors, typography,
    │       ├── base/                   #   boxShadows, borders, breakpoints,
    │       ├── functions/              #   globals, pxToRem, rgba, …)
    │       └── index.js
    └── examples/
        ├── Sidenav/                    # Vision UI sidenav (pruned)
        │   ├── index.js
        │   ├── SidenavRoot.js
        │   ├── SidenavCollapse.js
        │   └── styles/
        ├── LayoutContainers/
        │   └── DashboardLayout/        # page shell that compensates for sidenav
        └── Navbars/
            └── DashboardNavbar/        # minimal custom navbar (title + slot)
```

The frontend is located alongside the backend within the project root:

```
dashboard/
├── backend/
├── frontend/                           # this directory
└── .venv/                              # backend virtual environment
```

## Installation and execution

Dependencies are installed from the frontend directory:

```bash
cd frontend
npm install
```

The backend is started from the project root:

```bash
source .venv/bin/activate
MOCK_TIME_SCALE=20 python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The frontend development server is started from the frontend directory:

```bash
npm run dev
```

The application is served at `http://127.0.0.1:5173`. The default route (`/`) redirects to
`/live`; subsequent navigation is performed through the side navigation.

## Routing

| Route | Panel |
|-------|-------|
| `/` | Redirects to `/live` |
| `/live` | Panel 1 — Live |
| `/comparison` | Panel 2 — Comparison |
| `/evidence` | Panel 3 — Evidence |
| `/pipeline` | Panel 4 — Pipeline |

Appending the query parameter `?bare=1` to any panel URL suppresses the page navbar, which
is used when capturing figures for embedding in written reports. The side navigation
continues to render in this mode.

## Integration of the Vision UI template

The following elements of the Vision UI Free template are incorporated:

- **Theme**: colours, gradients, typography, box-shadows, borders, breakpoints, and globals.
- **Helper functions**: `linearGradient`, `pxToRem`, `boxShadow`, and `rgba`.
- **Components**: `VuiBox` and `VuiTypography`.
- **Examples**: `Sidenav`, `SidenavRoot`, `SidenavCollapse`, and `DashboardLayout`.
- **Context**: `VisionUIController`, which manages the side navigation's mini/expanded state.

The following template elements are deliberately omitted because they are not relevant to
the application: the `SidenavCard` ("Need help? / Documentation") call-to-action; the
"Upgrade to PRO" button; the `SimmmpleLogo` brand mark (replaced with the text label
"STRESS DEMO"); the template's `DashboardNavbar` with its search, sign-in, notification, and
configurator controls (replaced with a minimal custom navbar); the `Configurator` theme
drawer; the template's forty-plus MUI component overrides, which are unused because the
application does not render the default MUI components that would benefit from them; and
right-to-left layout support.

The application uses a template the way production dashboards commonly do: the four panels,
the WebSocket backend integration, the routing, the protocol layer, and the real-time data
handling are all bespoke, while the Vision UI Free template provides the visual layer.

## Application architecture

### Shared WebSocket connection

A single WebSocket connection is established at the application root through a
`WebSocketProvider`. In an earlier design, each panel invoked `useWebSocket(WS_URL)`
independently; consequently, every route change unmounted one panel and mounted another,
dropping the existing WebSocket and opening a new one. Because the connection reducer
cleared the prediction buffer on each reconnection, buffered prediction history did not
accumulate across panels.

In the current design, `useWebSocket` is invoked once at the application root within the
`WebSocketProvider`, and panels read connection state through `useWebSocketState()`, which
draws from React context. Navigation between panels therefore incurs no reconnection, and
the connection and its buffers persist across panel switches. In a validation run, no
reconnections occurred across nine consecutive navigations cycling through all four panels,
and the prediction buffer grew monotonically from 30 to 95 windows.

During development, React 18 Strict Mode opens the WebSocket three times on initial page
load as part of its mount–unmount–remount cleanup test. This behaviour is specific to
development mode; production builds (`npm run build`) open the connection exactly once.

### Live panel

The Live panel presents the current heart rate and stress probability in real time.

**Stress-probability gauge.** The live stress probability is displayed using a
speedometer-style gauge. The gauge is rendered as a 270° arc that opens at the bottom, with
a central disc showing the current probability and a mood glyph. Below a probability of
0.5, the arc, disc, and glyph adopt a cyan colour scheme with a smiling glyph, indicating a
non-stress state; at or above 0.5, they adopt a red colour scheme with a frowning glyph,
indicating a stress state. The threshold is a hard 0.5, matching the model's argmax decision
rather than applying any re-thresholding within the dashboard. The mood-glyph change
visualises the model's classification at the same threshold the model itself uses, and is
editorial rather than algorithmic. Transitions between predictions are animated over 0.4 s
with an ease curve; this tween is purely visual smoothing between prediction values (which
arrive every 8 s in production) and does not interpolate the model's output. A range strip
below the gauge displays the 0%, current, and 100% values, following the template's
satisfaction-rate pattern, and a quality-flag warning is rendered in amber when the
`quality_pass` flag is false. Quality-flagged windows continue to be rendered with their
flagged values rather than freezing the display, so that predictions are always shown and
quality is indicated alongside them.

The gauge is implemented in SVG using `stroke-dasharray` rather than a conic gradient,
because this affords precise control over arc length and allows the fill to be animated
smoothly between predictions; a conic gradient renders a full 360° ring cleanly but cannot
produce the open-bottom shape. The 270° span equals 0.75 of the circumference, computed from
the radius, and is rendered as a circle whose `strokeDasharray` is set to the arc length and
the remaining circumference so that the stroke is truncated to the visible arc. A
`<g transform="rotate(135 …)">` wrapper rotates the arc so that its start point lies at the
7:30 clock position, the conventional zero position of a speedometer. Tunable parameters in
`Live/index.jsx` include the gauge diameter (`SIZE = 200`), the arc thickness
(`STROKE = 16`), the arc-span fraction (`0.75` in `ARC_LEN = CIRC * 0.75`), the rotation
(`135`), and the decision threshold (`0.5` in `isStress = pStress >= 0.5`). The gauge
replaced an earlier conic-gradient ring, which read as a share-of-total indicator rather
than a measurement on a scale and which lacked a clear threshold affordance.

A complementary live stability metric is displayed as a confidence score on a 0–10 scale,
computed as `(1 − 2·std) · 10`. This metric conveys the ordering of prediction standard
deviation effectively but is not a standard machine-learning metric; it is a measure of live
stability rather than of accuracy, and reported F1 figures are presented separately in the
Evidence panel.

**Mood icons.** The mood glyphs in the gauge are rendered as inline SVG icons through a
small colour-overridable component, `MoodIcon`, rather than as character emoji. The icons
are sourced from the Noun Project: a smiley face by Kero (`noun-smiley-face-8374531.svg`)
and a frown by White Spaces Inc. (`noun-frown-707840.svg`). Inline SVG is used rather than
an `<img>` element because the latter cannot override the fill colour at render time, which
would cause the icon's source `fill="#000"` to render black against the coloured disc; the
inline component accepts a `color` prop and passes it as the path `fill`, yielding white
glyphs on both the red and cyan backgrounds. The path data is inlined as JavaScript
constants rather than imported through a Vite SVG plugin, since for two paths this avoids the
additional dependency and configuration at the cost of re-pasting the path string when the
icons are re-sourced. The component uses a `0 0 100 100` viewBox, tighter than the source
files' viewBoxes (`-5 -10 110 135` and `0 0 100 125`), in order to crop the embedded
attribution text from the rendered output; the attribution text is preserved in the source
files. An early version of the component had the smiley and frown path constants
transposed, which was detected when a low stress probability rendered a frown; the corrected
mapping is documented in the component header.

**Heart-rate readout.** The heart-rate display follows the readout-centric hierarchy used in
consumer fitness applications. The trace is rendered as an `AreaChart` rather than a
`LineChart`, with a vertical gradient fill (cyan at the top of the curve, fading to fully
transparent at the bottom) that gives the trace visual weight without painting a solid
block. Both axes are hidden, so that the chart conveys shape — climbing, sustained, or
falling — while the numeric measurement is carried by the readout below; the y-axis can be
re-enabled by changing `<YAxis hide />` to `<YAxis />`. The y-domain is data-driven, padding
the observed range by ±25% with a minimum padding of 8 BPM, replacing an earlier fixed
[40, 200] range that caused a flat trace to occupy only the lower portion of the chart. The
stroke uses a cyan→blue→cyan gradient and a width of 2.5 (increased from 2). Floating
`min N` and `max N` labels appear at the right edge at 60% opacity and report the range of
the currently visible buffer rather than session-historical extrema. The numeric readout
places the BPM value at the lower left in white (44 px, weight 700), followed by a small grey
"BPM" label and a "right now · live" sublabel.

A heart icon to the left of the readout pulses at the wearer's current heart rate. The icon
uses a standard Material/Lucide heart path with no external dependency, a red fill, and a
two-layer red drop-shadow glow that conveys warmth without changing size, so that the pulse
reads as a contraction rather than as growth. The animation period is synchronised to the
heart rate through an inline `animation` style with a duration of `60 / safeBpm` seconds, so
that the icon pulses once per second at 60 BPM and twice per second at 120 BPM. A "lub-dub"
keyframe defined in `globals.js` produces two scale peaks per cycle (a larger first peak at
1.18× and a smaller second peak at 1.10×), approximating the asymmetric profile of a cardiac
contraction. The displayed rate is clamped to 30–180 BPM to prevent anomalous animation when
a sample is briefly out of range. The rate driving the pulse is a three-sample rolling mean
rather than beat-level timing, which Fitbit does not provide; this yields a responsive pulse
without jitter and is appropriate for a glanceable display. The pulsing icon is a
human–computer-interaction element only: the underlying data are the 1 Hz heart-rate samples
shown in the trace, and the pulse is the icon's animation period set to `60 / current-BPM`
seconds.

In demonstrations, the heart-rate trace is driven by mock data containing a baked-in stress
excursion, following a WESAD-inspired baseline–stressor–baseline schedule; the resulting
climb-and-recover shape originates from the synthetic generator, whereas on real Fitbit data
the shape reflects the wearer's cardiac response.

### Comparison panel

The Comparison panel presents a three-model comparison. It reserves a slot, the
`PersonalisationGainPlaceholder`, for offline ablation bars that are loaded from a static
results file.

### Evidence panel

The Evidence panel presents the offline empirical results and is organised around the study's
contribution structure: the deployable artefact is the no-KD (no knowledge-distillation)
student, which attains a macro-F1 of approximately 0.75; knowledge distillation (KD) is
framed as an empirical investigation of the limits of cross-modal transfer; and an
information bottleneck is presented as the meta-mechanism that explains the principal
findings. The page reads top to bottom as four sections, each anchored by an empirical
question.

Three headline figures are presented at the top of the page. The HRV-only baseline attains a
macro-F1 of 0.75 (the deployable artefact, consistent with Prajod & André, 2022, and Dahal
et al., 2023). The aggregate KD effect is +0.009, a small gain accompanied by high
per-subject variance (with reference to Cho & Hariharan, ICCV 2019). The Fitbit pilot attains
0.4, reflecting the chest-ECG-to-wrist-PPG domain gap.

The four sections are as follows. The first, on why the performance ceiling exists, leads
with the information-bottleneck argument and pairs the ROC curves — in which the
approximately 0.08 AUC gap between teacher and student renders the bottleneck visible — with
the confusion matrices, which show KD's 22% reduction in false positives. The second, on
whether KD raises the ceiling, characterises the effect as modest but high-variance, draws
out the specific S11, S3, and S4 per-fold figures, cites Cho & Hariharan, and pairs the
per-subject F1 chart with the t-SNE embedding plot. The third, on whether personalisation
provides additional benefit on top of KD, reports a null result and pairs the
personalisation-gain figure with the teacher–student agreement figure. The fourth, on whether
the laboratory result transfers to a consumer wearable, presents the Fitbit pilot as a
further instance of the bottleneck argument applied to the chest-ECG-to-wrist-PPG domain
shift.

The information-bottleneck argument is the unifying mechanism that accounts for the no-KD
ceiling, KD's modest effect, the personalisation null result, and the Fitbit gap. It may be
stated concisely: the stress-related signal present in the teacher's BVP morphology is absent
from the six HRV summary statistics, and distillation cannot transfer information that the
student's input does not carry. The mean improvement of +0.0085 is small but is explained by
a principled, non-monotonic pattern — six folds gain and five folds lose, with KD helping
where the no-KD baseline is weak and hurting where the teacher is overconfident — consistent
with the relationship reported by Cho & Hariharan.

The methodological footer documents the hybrid distillation loss (α = 0.3 logit KD + β = 5
FitNets feature KD + class-weighted cross-entropy), the FedRep BatchNorm and head-calibration
warm-start procedure, and the four reference points (Cho & Hariharan; Prajod & André; Dahal;
Collins). Personalisation occurs once at the start of a session, comprising approximately
fifteen minutes of head fine-tuning, and is not part of the inference loop; the head is
loaded already fine-tuned before inference begins. The KD-Cen versus KD-Fed comparison is a
future experiment: it is noted as pending in the methodological footer and the fourth
section, with two diagnostic folds suggesting that per-fold variance is higher still with the
federated teacher.

Layout decisions in this panel include the following. Figures are wrapped on white cards
because matplotlib defaults to a white background; the wrap colour (`bgImageWrap`) can be
changed from `#ffffff` to `transparent` if figures are re-exported with
`savefig(transparent=True)`. Caveat boxes are stacked above the caption rather than beside
it, to prevent text clipping at narrow widths. Figure-pair grids use CSS Grid with
`alignItems: 'start'` so that cards size to their own content, and the layout collapses to a
single column below the `lg` (1024 px) breakpoint. The "22% false-positive reduction" figure
cited in the first-section caption is computed from the panel's own counts (17,016 → 13,268 =
22%).

### Pipeline panel

The Pipeline panel presents an animated SVG diagram of the data flow from the Fitbit watch
to the stress prediction. The diagram comprises two rows — a data pipeline (cyan) and a model
(violet) — with seven stages connected by edges that carry decorative packet animations.

Each stage's box displays its current real value: heart rate in BPM, sample count, window
identifier, the six z-scored features, embedding summary statistics, logits, and the stress
probability. These values are stepped on each prediction message, which arrives every 8 s in
production and more frequently when `MOCK_TIME_SCALE` is set. Each stage is clickable: a side
panel opens with a description and key/value details such as the corresponding file path in
the codebase, layer shapes, and citations; clicking the stage again or the close control
dismisses the panel. Each stage carries an `IMPLEMENTED` or `PLANNED` status badge. The
Fitbit stage is currently the only `PLANNED` stage, as the mock generator stands in for a
real poller; the remaining stages reference real code (`fitbit_preprocessor.py`,
`hrv_feature_extractor.py`, `student_model.py`). When a real Fitbit poller is added, the
Fitbit stage's status is updated in `STAGES[0].status`.

The packet animations along the edges are continuous (a CSS `offset-path` animation with an
approximately 1.8 s cycle and no JavaScript loop), whereas the per-stage values are stepped
(one update per prediction message, with a 200 ms CSS transition between values). This
decoupling is deliberate: the packets convey visual continuity, while the values reflect the
data.

The numeric parameters in the diagram are read directly from the project's source files: the
16 s window and 8 s slide from `kd_config.py`, validated against the `wesad_dl_dataset.py` and
`fl_config.py` defaults; the six features from `kd_config.PRIMARY_HRV_FEATURES`; the cubic
spline interpolation and clipping to [40, 200] from
`fitbit_preprocessor.interpolate_to_uniform`; the gap threshold of greater than 5 s from
`kd_config.FITBIT_MAX_GAP_SEC` (Bottcher, 2022); the 20% flagged-exclusion figure from
`kd_config` and the implementation plan; the layer widths 6 → 64 → 32 → 16 → 2 from
`student_model.py`; and `TEACHER_FEATURE_DIM = 32`, which explains why the student base
output is 32-dimensional.

The diagram is purely architectural and does not include a t-SNE or actual-versus-predicted
view; model-behaviour questions are addressed by the Comparison and Evidence panels rather
than by the Pipeline panel. The diagram also omits the calibration phase, since
personalisation occurs once at session start and is not part of the inference loop, and
including it would obscure the message. The diagram does not auto-highlight a stage as data
flows through it; a per-stage processing indicator was not added, because it would imply a
sequential timing that does not reflect the pipelined operation of the backend, and because
the packet animations already convey data flow without misleading sequencing.

## Visual design

The application's background and cards reproduce the layered-gradient and frosted-glass
appearance of the Vision UI demo.

The bootstrap `<style>` block in `index.html` originally set
`html, body, #root { background: #0b1437 }`, a flat dark navy applied to the document root as
a no-flash-of-white fallback before React mounts. As a side effect, this rule silently
overrode every subsequent backdrop applied by the theme, because in some browsers' computed-
style ordering a rule targeting `html` from a hardcoded `<style>` tag takes precedence over
MUI's runtime-injected styles. This rendered the gradient backdrop invisible. The fix applies
the bootstrap colour to `body` only; the theme's `body { background: transparent }` rule from
`globals.js` then overrides it once React mounts, and the `html` gradient becomes visible.

The backdrop, defined in `globals.js`, is a five-layer stack with
`background-attachment: fixed`:

```
top    → radial-gradient ellipse 80% 60% at 25% 15%, rgba(67, 24, 255, 0.50)
        → radial-gradient ellipse 55% 70% at 95% 55%, rgba(33, 212, 253, 0.28)
        → radial-gradient ellipse 70% 45% at 55% 100%, rgba(159, 122, 234, 0.22)
        → url(body-background.png) no-repeat center bottom / cover
bottom → #030c1d (dark base)
```

The CSS gradients produce the violet and cyan glow seen through the translucent cards; the
PNG image contributes the cloud-like texture of the original template; and the dark base
covers any remaining area. The violet glow can be intensified by raising the 0.50 alpha of
the first gradient, repositioned by changing its `25% 15%` origin, or reduced by lowering all
alphas by approximately 0.15.

Each panel's card background (`bgCard`) is a CSS gradient (the Vision UI card gradient: 94%
opaque at the top-left, 49% opaque at the bottom-right). The `PanelCard` helper applies
`backdrop-filter: blur(45px)` and a soft translucent border (`rgba(255, 255, 255, 0.06)`),
producing a frosted-glass appearance through which the gradient is visible. A solid card
background (`bgCardSolid: '#0f1535'`) is provided for chart tooltips, where opacity is
required for legibility.

Because the application uses the recognisable Vision UI Free template as its visual layer,
the dashboard application itself — the four panels, the WebSocket backend integration, the
routing, the protocol, and the real-time data handling — is custom-built on top of that
layer. Colour intensity may render differently across displays: high-gamut laptop screens
render the violet more saturated than a typical projector, and on a washed-out projector the
gradient may be less visible; testing on the intended display is recommended.

## Typography

The Plus Jakarta Display font is imported from a jsDelivr CDN in `typography.css`, with a
fallback to Helvetica when offline. Self-hosting the font is recommended for offline
reliability.

## Performance

`backdrop-filter: blur(...)` is GPU-composited and is effectively free on modern laptops, but
it can stutter at large window sizes on older or integrated-graphics machines; if necessary,
the blur can be reduced to `blur(20px)` or removed entirely, and the gradient will still show
through because the cards are translucent regardless of blur. The stress gauge uses SVG and
CSS transitions and is lighter than the earlier conic-gradient implementation; its
drop-shadow filter is GPU-composited and fast even on integrated graphics. The heart-rate
`AreaChart` with a fill gradient is marginally heavier than a `LineChart` but remains well
within Recharts' performance envelope; the heartbeat keyframe is GPU-composited (transform:
scale), and the drop-shadow filter on the heart icon, while the most expensive element in the
panel, remains negligible. No `backdrop-filter` or canvas rendering is used in the heart-rate
panel.

## Attribution

The Noun Project royalty-free licence (CC BY) requires attribution, which is provided in the
SVG source files (both files retain the embedded
`<text>Created by … from Noun Project</text>` element, preserved at the file level even
though it is not rendered in the gauge) and in the `MoodIcon` component header (which names
both creators, the source filenames, and the licence URL). A corresponding credit line should
be included in the written report, for example: "Mood icons in the live dashboard are adapted
from the Noun Project: smiley face by Kero, frown face by White Spaces Inc.; both used under
the Noun Project royalty-free licence (CC BY)."

## Known limitations and future work

The four panels are feature-complete. The principal known limitations and outstanding tasks
are as follows.

Several Recharts components set `isAnimationActive={false}`, because animation interpolation
is unsatisfactory at the 8 s prediction cadence and static-frame redraws are more responsive;
a smoother appearance, if desired, would be better achieved with a CSS transition on the path
stroke than with Recharts' built-in animation. The `StudentHead × 3` title in the Pipeline
panel wraps to two lines within its box; this is cosmetic and can be addressed, if desired,
by shortening the title, widening the boxes (from 200 to 220), or splitting the "× 3"
visually. Reported figures in the Evidence panel should be updated when the full
leave-one-subject-out (LOSO) evaluation completes with a different mean improvement: the panel
currently quotes +0.009 and eleven visible LOSO folds, and the headline metric card and the
second-section prose should be edited when the remaining four folds resolve. If KD-Fed full
LOSO results become available, a comparison may be added either as an additional section
(KD-Cen versus KD-Fed) or by overlaying KD-Fed bars on the per-subject F1 chart.

The remaining integration work comprises self-hosting the Plus Jakarta Display font for
offline reliability, providing a JSONL replay source as a demonstration-day backup, and
connecting the real backend (the Fitbit poller and model inference). The Evidence panel's
figures are static images generated from the offline LOSO ablation runs (confusion matrices,
macro-F1 by phase, and personalisation-gain bars).
