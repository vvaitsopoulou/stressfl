/**
 * ComparisonPanel — Panel 2, "Why does personalisation matter?"
 *
 * This is the panel the dissertation depends on. It needs to make
 * three points visually:
 *
 *   1. The three models disagree (stacked timelines)
 *   2. Personalised is more confident and stable (running metric cards)
 *   3. The disagreement clusters in interesting places (diff chart)
 *      — a live proxy for the offline "personalisation gain" claim
 */

import { Box, Card, Typography } from '@mui/material';
import {
  LineChart, Line, XAxis, YAxis, ResponsiveContainer,
  Tooltip, ReferenceLine, ReferenceArea, AreaChart, Area,
} from 'recharts';
import { useWebSocketState } from 'hooks/WebSocketProvider';
import {
  selectPerModelSeries,
  selectRunningStats,
  selectAgreement,
  selectMeanAbsDiff,
} from 'hooks/selectors';
import { MODEL_IDS, HEADLINE_MODEL_ID } from 'protocol';
import DashboardLayout from 'examples/LayoutContainers/DashboardLayout';
import DashboardNavbar from 'examples/Navbars/DashboardNavbar';

const COLORS = {
  bgCard:      'linear-gradient(127.09deg, rgba(6, 11, 40, 0.94) 19.41%, rgba(10, 14, 35, 0.49) 76.65%)',
  bgCardSolid: '#0f1535',
  bgCardEdge:  '#1d2150',
  textPrimary: '#e8eaf3',
  textMuted:   '#a3aed0',
  cyan:        '#2cd9ff',
  blue:        '#0075ff',
  violet:      '#8b5cf6',
  green:       '#01b574',
  amber:       '#ffb547',
  red:         '#e31a1a',
};

const MODEL_COLORS = {
  no_kd:        COLORS.amber,    // weakest, "raw" baseline
  global_kd:    COLORS.cyan,     // mid, distillation alone
  personalised: COLORS.violet,   // strongest, distillation + personalisation
};

const MODEL_LABELS = {
  no_kd:        'No KD (b′)',
  global_kd:    'Global KD (c′)',
  personalised: 'KD + personalised (d′)',
};

function PanelCard({ children, sx = {} }) {
  return (
    <Card
      elevation={0}
      sx={{
        background: COLORS.bgCard,
        backdropFilter: 'blur(45px)',
        border: '1px solid rgba(255, 255, 255, 0.06)',
        borderRadius: '20px',
        color: COLORS.textPrimary,
        p: 3,
        ...sx,
      }}
    >
      {children}
    </Card>
  );
}

function computeSharedDomain(perModelSeries) {
  const all = [
    ...perModelSeries.no_kd,
    ...perModelSeries.global_kd,
    ...perModelSeries.personalised,
  ];
  if (all.length === 0) return ['dataMin', 'dataMax'];
  const ts = all.map((d) => d.t);
  return [Math.min(...ts), Math.max(...ts)];
}

function ModelTimeline({ modelId, data, domain, showXAxis }) {
  const colour = MODEL_COLORS[modelId];
  const label = MODEL_LABELS[modelId];

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
        <Typography variant="caption" sx={{ color: colour, fontWeight: 600, letterSpacing: 0.5 }}>
          {label}
        </Typography>
        <Typography variant="caption" sx={{ color: COLORS.textMuted }}>
          {data.length} pts
        </Typography>
      </Box>
      <Box sx={{ width: '100%', height: showXAxis ? 110 : 90 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 4, right: 16, left: 8, bottom: showXAxis ? 0 : 0 }}>
            <XAxis
              dataKey="t"
              type="number"
              domain={domain}
              tick={showXAxis ? { fill: COLORS.textMuted, fontSize: 10 } : false}
              tickFormatter={(v) => `${Math.round(v / 60)}m`}
              stroke={COLORS.bgCardEdge}
              hide={!showXAxis}
            />
            <YAxis
              domain={[0, 1]}
              ticks={[0, 0.5, 1]}
              tick={{ fill: COLORS.textMuted, fontSize: 9 }}
              stroke={COLORS.bgCardEdge}
              width={28}
            />
            <ReferenceLine y={0.5} stroke={COLORS.bgCardEdge} strokeDasharray="3 3" />
            <Tooltip
              contentStyle={{
                background: COLORS.bgCardEdge,
                border: 'none',
                borderRadius: 8,
                color: COLORS.textPrimary,
              }}
              labelFormatter={(v) => `t=${v.toFixed(0)}s`}
              formatter={(v) => [`${(v * 100).toFixed(1)}%`, label]}
            />
            <Line
              type="monotone"
              dataKey="p"
              stroke={colour}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </Box>
  );
}

function StackedTimelines({ predictionBuffer }) {
  const series = selectPerModelSeries(predictionBuffer);
  const domain = computeSharedDomain(series);

  return (
    <PanelCard sx={{ height: '100%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', mb: 2 }}>
        <Typography variant="overline" sx={{ color: COLORS.textMuted, letterSpacing: 1.5 }}>
          Three-model timelines (P(stress))
        </Typography>
        <Typography variant="caption" sx={{ color: COLORS.textMuted }}>
          shared x-axis · 8 s slide
        </Typography>
      </Box>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
        <ModelTimeline modelId="no_kd"        data={series.no_kd}        domain={domain} showXAxis={false} />
        <ModelTimeline modelId="global_kd"    data={series.global_kd}    domain={domain} showXAxis={false} />
        <ModelTimeline modelId="personalised" data={series.personalised} domain={domain} showXAxis />
      </Box>
    </PanelCard>
  );
}

function ModelMetricCard({ modelId, predictionBuffer }) {
  const stats = selectRunningStats(predictionBuffer, modelId, 30);
  const isHeadline = modelId === HEADLINE_MODEL_ID;
  const colour = MODEL_COLORS[modelId];

  const score = stats.stableConfidence != null
    ? (stats.stableConfidence * 10).toFixed(1)
    : '—';

  const meanPct = stats.mean != null ? `${(stats.mean * 100).toFixed(0)}%` : '—';
  const stdPct  = stats.std  != null ? `${(stats.std  * 100).toFixed(1)}%` : '—';
  const agree = modelId === HEADLINE_MODEL_ID
    ? null
    : selectAgreement(predictionBuffer, modelId, HEADLINE_MODEL_ID, 30);
  const agreePct = agree != null ? `${(agree * 100).toFixed(0)}%` : '—';

  return (
    <PanelCard
      sx={{
        height: '100%',
        ...(isHeadline && {
          border: `1px solid ${colour}`,
          boxShadow: `0 0 32px -16px ${colour}`,
        }),
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: colour }} />
        <Typography variant="caption" sx={{ color: colour, fontWeight: 700, letterSpacing: 0.5 }}>
          {MODEL_LABELS[modelId]}
        </Typography>
      </Box>
      <Typography variant="caption" sx={{ color: COLORS.textMuted, display: 'block', mb: 2 }}>
        rolling 30 windows · ~4 min
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1, mb: 2 }}>
        <Typography variant="h2" sx={{ color: 'white', fontWeight: 700, lineHeight: 1 }}>
          {score}
        </Typography>
        <Typography variant="body2" sx={{ color: COLORS.textMuted }}>
          / 10
        </Typography>
      </Box>
      <Typography variant="caption" sx={{ color: COLORS.textMuted, display: 'block', mb: 1.5 }}>
        confidence stability score
      </Typography>
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1.5 }}>
        <Stat label="mean P(stress)" value={meanPct} />
        <Stat label="std P(stress)"  value={stdPct} />
        <Stat
          label={modelId === HEADLINE_MODEL_ID ? 'role' : 'agree w/ pers'}
          value={modelId === HEADLINE_MODEL_ID ? 'reference' : agreePct}
        />
        <Stat label="windows seen" value={String(stats.n)} />
      </Box>
    </PanelCard>
  );
}

function Stat({ label, value }) {
  return (
    <Box>
      <Typography variant="caption" sx={{ color: COLORS.textMuted, display: 'block' }}>
        {label}
      </Typography>
      <Typography variant="body1" sx={{ color: 'white', fontWeight: 600 }}>
        {value}
      </Typography>
    </Box>
  );
}

function DisagreementChart({ predictionBuffer }) {
  // We render two diff series in the same chart: |personalised - no_kd|
  // and |personalised - global_kd|. Higher diff = bigger disagreement.
  const dNoKD   = selectMeanAbsDiff(predictionBuffer, HEADLINE_MODEL_ID, 'no_kd');
  const dGlob   = selectMeanAbsDiff(predictionBuffer, HEADLINE_MODEL_ID, 'global_kd');

  const merged = dNoKD.map((row, i) => ({
    t: row.t,
    diff_no_kd: row.diff,
    diff_global: dGlob[i]?.diff ?? null,
  }));

  return (
    <PanelCard sx={{ height: '100%' }}>
      <Typography variant="overline" sx={{ color: COLORS.textMuted, letterSpacing: 1.5 }}>
        Disagreement with personalised
      </Typography>
      <Typography variant="caption" sx={{ color: COLORS.textMuted, display: 'block', mb: 2 }}>
        |Δ P(stress)| from headline model · live proxy for personalisation gain
      </Typography>
      <Box sx={{ width: '100%', height: 200 }}>
        <ResponsiveContainer>
          <AreaChart data={merged} margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
            <defs>
              <linearGradient id="diffNoKD" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   stopColor={MODEL_COLORS.no_kd}     stopOpacity={0.7} />
                <stop offset="100%" stopColor={MODEL_COLORS.no_kd}     stopOpacity={0} />
              </linearGradient>
              <linearGradient id="diffGlob" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   stopColor={MODEL_COLORS.global_kd} stopOpacity={0.7} />
                <stop offset="100%" stopColor={MODEL_COLORS.global_kd} stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="t"
              type="number"
              domain={['dataMin', 'dataMax']}
              tick={{ fill: COLORS.textMuted, fontSize: 10 }}
              tickFormatter={(v) => `${Math.round(v / 60)}m`}
              stroke={COLORS.bgCardEdge}
            />
            <YAxis
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5]}
              tick={{ fill: COLORS.textMuted, fontSize: 10 }}
              stroke={COLORS.bgCardEdge}
              width={32}
            />
            <Tooltip
              contentStyle={{
                background: COLORS.bgCardEdge,
                border: 'none',
                borderRadius: 8,
                color: COLORS.textPrimary,
              }}
              labelFormatter={(v) => `t=${v.toFixed(0)}s`}
              formatter={(v, key) => {
                const label = key === 'diff_no_kd' ? 'vs No KD' : 'vs Global KD';
                return [v == null ? '—' : v.toFixed(3), label];
              }}
            />
            <Area
              type="monotone"
              dataKey="diff_no_kd"
              stroke={MODEL_COLORS.no_kd}
              strokeWidth={1.5}
              fill="url(#diffNoKD)"
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="diff_global"
              stroke={MODEL_COLORS.global_kd}
              strokeWidth={1.5}
              fill="url(#diffGlob)"
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </Box>
    </PanelCard>
  );
}

function PersonalisationGainPlaceholder() {
  return (
    <PanelCard sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="overline" sx={{ color: COLORS.textMuted, letterSpacing: 1.5 }}>
        Personalisation gain by phase
      </Typography>
      <Typography variant="caption" sx={{ color: COLORS.textMuted, display: 'block', mb: 2 }}>
        Δ macro-F1 (personalised − global) · loaded from offline ablation
      </Typography>
      <Box
        sx={{
          flex: 1,
          minHeight: 200,
          border: `1px dashed ${COLORS.bgCardEdge}`,
          borderRadius: '12px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 1,
          p: 2,
        }}
      >
        <Typography variant="caption" sx={{ color: COLORS.textMuted, textAlign: 'center' }}>
          Reserved slot — wired to static results in a future session.
        </Typography>
        <Typography variant="caption" sx={{ color: COLORS.textMuted, textAlign: 'center', maxWidth: 400 }}>
          Will display per-phase Δ macro-F1 bars from <code>plot_personalisation_gain</code>.
          Live proxy is the disagreement chart on the left.
        </Typography>
      </Box>
    </PanelCard>
  );
}

function CalibrationFooter({ hello }) {
  const cal = hello?.calibration;
  if (!cal) {
    return (
      <PanelCard sx={{ py: 2 }}>
        <Typography variant="caption" sx={{ color: COLORS.textMuted }}>
          Awaiting calibration metadata…
        </Typography>
      </PanelCard>
    );
  }
  const dt = new Date(cal.completed_at_iso);
  const clockTime = isNaN(dt) ? cal.completed_at_iso : dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return (
    <PanelCard sx={{ py: 2 }}>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, alignItems: 'center' }}>
        <Typography variant="caption" sx={{ color: COLORS.textMuted, letterSpacing: 0.5 }}>
          PERSONALISED HEAD CALIBRATION
        </Typography>
        <Typography variant="caption" sx={{ color: 'white' }}>
          n = <b>{cal.n_windows_used}</b> windows
        </Typography>
        <Typography variant="caption" sx={{ color: 'white' }}>
          duration = <b>{cal.duration_min.toFixed(0)} min</b>
        </Typography>
        <Typography variant="caption" sx={{ color: 'white' }}>
          K = <b>{cal.fine_tune_steps}</b> steps
        </Typography>
        <Typography variant="caption" sx={{ color: 'white' }}>
          lr = <b>{cal.learning_rate}</b>
        </Typography>
        <Typography variant="caption" sx={{ color: 'white' }}>
          completed = <b>{clockTime}</b>
        </Typography>
      </Box>
    </PanelCard>
  );
}

//  Top-level panel

export default function ComparisonPanel() {
  const state = useWebSocketState();
  const { hello, predictionBuffer, connectionState } = state;

  const bare = typeof window !== 'undefined' &&
    new URLSearchParams(window.location.search).get('bare') === '1';

  const content = (
    <Box
      sx={{
        display: 'grid',
        gap: 3,
        gridTemplateColumns: { xs: '1fr', md: '2fr 1fr 1fr 1fr' },
        gridTemplateAreas: {
          xs: `'timelines' 'm0' 'm1' 'm2' 'diff' 'gain' 'cal'`,
          md: `
            'timelines timelines timelines timelines'
            'm0        m1        m2        m2'
            'diff      diff      gain      gain'
            'cal       cal       cal       cal'
          `,
        },
      }}
    >
      <Box sx={{ gridArea: 'timelines' }}>
        <StackedTimelines predictionBuffer={predictionBuffer} />
      </Box>
      <Box sx={{ gridArea: 'm0' }}>
        <ModelMetricCard modelId="no_kd"        predictionBuffer={predictionBuffer} />
      </Box>
      <Box sx={{ gridArea: 'm1' }}>
        <ModelMetricCard modelId="global_kd"    predictionBuffer={predictionBuffer} />
      </Box>
      <Box sx={{ gridArea: 'm2' }}>
        <ModelMetricCard modelId="personalised" predictionBuffer={predictionBuffer} />
      </Box>
      <Box sx={{ gridArea: 'diff' }}>
        <DisagreementChart predictionBuffer={predictionBuffer} />
      </Box>
      <Box sx={{ gridArea: 'gain' }}>
        <PersonalisationGainPlaceholder />
      </Box>
      <Box sx={{ gridArea: 'cal' }}>
        <CalibrationFooter hello={hello} />
      </Box>
    </Box>
  );

  if (bare) {
    return <Box sx={{ p: 3, minHeight: '100vh' }}>{content}</Box>;
  }

  return (
    <DashboardLayout>
      <DashboardNavbar title="Comparison · Why personalisation matters">
        <ConnectionBadge state={connectionState} dataSource={hello?.data_source} />
      </DashboardNavbar>
      {content}
    </DashboardLayout>
  );
}

function ConnectionBadge({ state, dataSource }) {
  const labelMap = {
    idle:       { text: 'idle',                   color: COLORS.textMuted },
    connecting: { text: 'connecting…',            color: COLORS.amber },
    open:       { text: `live · ${dataSource ?? '…'}`, color: COLORS.green },
    closed:     { text: 'reconnecting',           color: COLORS.amber },
    error:      { text: 'error',                  color: COLORS.red },
  };
  const { text, color } = labelMap[state] ?? labelMap.idle;
  return (
    <Box
      sx={{
        display: 'inline-flex',
        alignItems: 'center',
        height: 24,
        px: 1.5,
        borderRadius: '16px',
        border: `1px solid ${color}`,
        color,
        fontSize: 13,
        fontWeight: 600,
        letterSpacing: 0.5,
      }}
    >
      {text}
    </Box>
  );
}
