/**
 * LivePanel — Panel 1 of the dashboard, "What is happening right now?"
 *
 * Layout :
 *
 *   ┌───────────────────────────────┬─────────────────────────────┐
 *   │  Live HR trace (Welcome slot) │  Stress gauge (Sat slot)    │
 *   │  + system metric strip        │  + Model confidence card    │
 *   ├───────────────────────────────┴─────────────────────────────┤
 *   │  Prediction timeline ribbon (full width)                    │
 *   ├──────────────────────────────────┬──────────────────────────┤
 *   │  Quality + features              │  System events log       │
 *   └──────────────────────────────────┴──────────────────────────┘
 */

import { Box, Card, Chip, Typography, LinearProgress } from '@mui/material';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, ResponsiveContainer, Tooltip, ReferenceLine } from 'recharts';
import { useWebSocketState } from 'hooks/WebSocketProvider';
import { HEADLINE_MODEL_ID, FEATURE_NAMES } from 'protocol';
import DashboardLayout from 'examples/LayoutContainers/DashboardLayout';
import DashboardNavbar from 'examples/Navbars/DashboardNavbar';
import MoodIcon from 'components/MoodIcon';

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

// Sub-components

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
        height: '100%',
        ...sx,
      }}
    >
      {children}
    </Card>
  );
}

function ConnectionBadge({ state, dataSource }) {
  const labelMap = {
    idle: { text: 'idle', color: COLORS.textMuted },
    connecting: { text: 'connecting…', color: COLORS.amber },
    open: { text: `live · ${dataSource ?? '…'}`, color: COLORS.green },
    closed: { text: 'reconnecting', color: COLORS.amber },
    error: { text: 'error', color: COLORS.red },
  };
  const { text, color } = labelMap[state] ?? labelMap.idle;
  return (
    <Chip
      size="small"
      label={text}
      sx={{
        bgcolor: 'transparent',
        border: `1px solid ${color}`,
        color,
        fontWeight: 600,
        letterSpacing: '0.5px',
      }}
    />
  );
}

function StressGauge({ pStress, qualityPass }) {

  const SIZE = 200;
  const STROKE = 16;
  const RADIUS = (SIZE - STROKE) / 2;
  const CIRC = 2 * Math.PI * RADIUS;
  const ARC_LEN = CIRC * 0.75;

  const pct = pStress != null ? Math.round(pStress * 100) : null;
  const fillFrac = pStress != null ? Math.max(0, Math.min(1, pStress)) : 0;
  const filledArc = ARC_LEN * fillFrac;

  // Threshold colour. Above 0.5 = stress; below = no stress.
  const isStress = pStress != null && pStress >= 0.5;
  const ringColor = pct == null ? COLORS.textMuted
    : isStress ? COLORS.red
    : COLORS.cyan;
  const emojiMood = pct == null ? null : isStress ? 'frown' : 'smiley';

  return (
    <PanelCard>
      <Typography variant="overline" sx={{ color: COLORS.textMuted, letterSpacing: 1.5 }}>
        Live stress probability
      </Typography>
      <Typography variant="caption" sx={{ color: COLORS.textMuted, display: 'block', mb: 1 }}>
        Personalised model · headline value
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1 }}>
        <Box sx={{ position: 'relative', width: SIZE, height: SIZE }}>
          <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}>
            {/* Rotate the whole gauge so the arc opens at the bottom. */}
            <g transform={`rotate(135 ${SIZE / 2} ${SIZE / 2})`}>
              {/* Background track — the unfilled portion of the arc */}
              <circle
                cx={SIZE / 2}
                cy={SIZE / 2}
                r={RADIUS}
                fill="none"
                stroke={COLORS.bgCardEdge}
                strokeWidth={STROKE}
                strokeLinecap="round"
                strokeDasharray={`${ARC_LEN} ${CIRC}`}
              />
              {/* Foreground arc — the filled portion. */}
              <circle
                cx={SIZE / 2}
                cy={SIZE / 2}
                r={RADIUS}
                fill="none"
                stroke={ringColor}
                strokeWidth={STROKE}
                strokeLinecap="round"
                strokeDasharray={`${filledArc} ${CIRC}`}
                style={{
                  transition: 'stroke-dasharray 0.4s ease, stroke 0.4s ease',
                  filter: `drop-shadow(0 0 8px ${ringColor})`,
                }}
              />
            </g>
          </svg>
          {/* Centre disc — solid coloured circle with emoji + value.*/}
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              width: 110,
              height: 110,
              borderRadius: '50%',
              background: ringColor,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: `0 0 32px -4px ${ringColor}`,
              transition: 'background 0.4s ease, box-shadow 0.4s ease',
            }}
          >
            <Box sx={{ mb: 0.5, height: 36, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {emojiMood ? (
                <MoodIcon mood={emojiMood} size={36} color="white" />
              ) : (
                <Box sx={{ fontSize: 28, lineHeight: 1, color: 'white', opacity: 0.6 }}>—</Box>
              )}
            </Box>
            <Typography sx={{ color: 'white', fontWeight: 700, fontSize: 20, lineHeight: 1 }}>
              {pct == null ? '—' : `${pct}%`}
            </Typography>
          </Box>
        </Box>
      </Box>
      {/*  mirrors Vision UI's "0% — N% — 100%" strip */}
      <Box
        sx={{
          mt: 1.5,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          px: 1,
        }}
      >
        <Typography variant="caption" sx={{ color: COLORS.textMuted }}>0%</Typography>
        <Typography variant="caption" sx={{ color: 'white', fontWeight: 600 }}>
          {pct == null ? 'warming up' : `${pct}% P(stress)`}
        </Typography>
        <Typography variant="caption" sx={{ color: COLORS.textMuted }}>100%</Typography>
      </Box>
      {!qualityPass && pct != null && (
        <Typography variant="caption" sx={{ color: COLORS.amber, mt: 1, display: 'block', textAlign: 'center' }}>
          ⚠ Quality flagged on this window
        </Typography>
      )}
    </PanelCard>
  );
}

/**
 * HeartBeatIcon: The animation is a CSS keyframes scale
 * cycle whose duration is set inline as `60 / bpm` seconds, so the
 * icon literally beats at the rate the chart shows.
 */
function HeartBeatIcon({ bpm, size = 36 }) {
  const safeBpm = bpm == null ? 60 : Math.max(30, Math.min(180, bpm));
  const periodSec = 60 / safeBpm;

  return (
    <Box
      sx={{
        width: size,
        height: size,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        animation: `heartbeat ${periodSec.toFixed(3)}s ease-in-out infinite`,
        filter: `drop-shadow(0 0 6px ${COLORS.red}) drop-shadow(0 0 12px rgba(227, 26, 26, 0.4))`,
      }}
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        xmlns="http://www.w3.org/2000/svg"
        aria-label="heart"
      >
        {/* Standard heart path */}
        <path
          d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"
          fill={COLORS.red}
        />
      </svg>
    </Box>
  );
}

function HRTrace({ samples }) {
  const data = samples.slice(-300).map((s) => ({
    t: s.t,
    hr: s.hr,
    in_gap: s.in_gap,
  }));

  // Pull min, max, and current BPM out of the buffered data.
  const recentN = 3;
  const recent = data.slice(-recentN);
  const currentBpm = recent.length === 0
    ? null
    : Math.round(recent.reduce((sum, d) => sum + d.hr, 0) / recent.length);
  const hrValues = data.map((d) => d.hr);
  const minBpm = hrValues.length ? Math.min(...hrValues) : null;
  const maxBpm = hrValues.length ? Math.max(...hrValues) : null;

  let yDomain = [40, 200];
  if (minBpm != null && maxBpm != null) {
    const pad = Math.max(8, (maxBpm - minBpm) * 0.25);
    yDomain = [
      Math.max(35, Math.floor(minBpm - pad)),
      Math.min(220, Math.ceil(maxBpm + pad)),
    ];
  }

  return (
    <PanelCard sx={{ minHeight: 280, position: 'relative', overflow: 'hidden' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', mb: 1 }}>
        <Typography variant="overline" sx={{ color: COLORS.textMuted, letterSpacing: 1.5 }}>
          Live heart rate
        </Typography>
        <Typography variant="caption" sx={{ color: COLORS.textMuted }}>
          {data.length}/300 samples buffered
        </Typography>
      </Box>


      <Box sx={{ width: '100%', height: 180, position: 'relative' }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <defs>
              <linearGradient id="hrStroke" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor={COLORS.cyan} />
                <stop offset="50%" stopColor={COLORS.blue} />
                <stop offset="100%" stopColor={COLORS.cyan} />
              </linearGradient>
              <linearGradient id="hrFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={COLORS.cyan} stopOpacity={0.45} />
                <stop offset="60%" stopColor={COLORS.cyan} stopOpacity={0.10} />
                <stop offset="100%" stopColor={COLORS.cyan} stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="t" hide />
            <YAxis domain={yDomain} hide />
            <Tooltip
              contentStyle={{ background: COLORS.bgCardSolid, border: 'none', borderRadius: 8 }}
              labelFormatter={(v) => `t=${v.toFixed(0)}s`}
              formatter={(v) => [`${v} BPM`, 'HR']}
            />
            <Area
              type="monotone"
              dataKey="hr"
              stroke="url(#hrStroke)"
              strokeWidth={2.5}
              fill="url(#hrFill)"
              fillOpacity={1}
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>

        {minBpm != null && maxBpm != null && (
          <>
            <Typography
              variant="caption"
              sx={{
                position: 'absolute',
                top: 4,
                right: 8,
                color: COLORS.textMuted,
                fontSize: 11,
                opacity: 0.6,
              }}
            >
              max {maxBpm}
            </Typography>
            <Typography
              variant="caption"
              sx={{
                position: 'absolute',
                bottom: 4,
                right: 8,
                color: COLORS.textMuted,
                fontSize: 11,
                opacity: 0.6,
              }}
            >
              min {minBpm}
            </Typography>
          </>
        )}
      </Box>

      {/* Headline readout — Oura-style. */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.75, mt: 2 }}>
        <HeartBeatIcon bpm={currentBpm} size={42} />
        <Box sx={{ display: 'flex', flexDirection: 'column' }}>
          <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 0.75 }}>
            <Typography sx={{ fontSize: 44, fontWeight: 700, color: 'white', lineHeight: 1 }}>
              {currentBpm == null ? '—' : currentBpm}
            </Typography>
            <Typography sx={{ fontSize: 14, fontWeight: 600, color: COLORS.textMuted, letterSpacing: 0.5 }}>
              BPM
            </Typography>
          </Box>
          <Typography variant="caption" sx={{ color: COLORS.textMuted, fontSize: 11 }}>
            {currentBpm == null ? 'awaiting samples' : 'right now · live'}
          </Typography>
        </Box>
      </Box>
    </PanelCard>
  );
}

function ModelConfidenceCard({ latestPrediction }) {
  const noKD = latestPrediction?.models?.no_kd?.p_stress;
  const glob = latestPrediction?.models?.global_kd?.p_stress;
  const pers = latestPrediction?.models?.personalised?.p_stress;
  const latency = latestPrediction?.inference_latency_ms;
  const winId = latestPrediction?.window_id;

  const fmt = (v) => (v == null ? '—' : `${(v * 100).toFixed(0)}%`);

  return (
    <PanelCard>
      <Typography variant="overline" sx={{ color: COLORS.textMuted, letterSpacing: 1.5 }}>
        Three-model snapshot
      </Typography>
      <Typography variant="caption" sx={{ color: COLORS.textMuted, display: 'block', mb: 2 }}>
        Window #{winId ?? '—'}
      </Typography>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
        <ModelRow label="No KD (b′)"            value={fmt(noKD)} />
        <ModelRow label="Global KD (c′)"        value={fmt(glob)} />
        <ModelRow label="Personalised (d′)"     value={fmt(pers)} highlight />
      </Box>
      <Typography variant="caption" sx={{ color: COLORS.textMuted, display: 'block', mt: 2 }}>
        inference: {latency == null ? '—' : `${latency.toFixed(2)} ms`} (3 models, CPU)
      </Typography>
    </PanelCard>
  );
}

function ModelRow({ label, value, highlight }) {
  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        p: 1.25,
        borderRadius: '12px',
        bgcolor: highlight ? '#1a1f5c' : COLORS.bgCardEdge,
        border: highlight ? `1px solid ${COLORS.cyan}` : '1px solid transparent',
      }}
    >
      <Typography variant="body2" sx={{ color: highlight ? 'white' : COLORS.textMuted }}>
        {label}
      </Typography>
      <Typography variant="body1" sx={{ color: 'white', fontWeight: 700 }}>
        {value}
      </Typography>
    </Box>
  );
}

function PredictionRibbon({ predictionBuffer }) {
  const data = predictionBuffer.slice(-100).map((p) => ({
    t: p.window_end_s,
    p: p.models?.personalised?.p_stress ?? null,
    win: p.window_id,
    quality: p.quality_pass,
  }));

  return (
    <PanelCard sx={{ minHeight: 200 }}>
      <Typography variant="overline" sx={{ color: COLORS.textMuted, letterSpacing: 1.5 }}>
        Prediction timeline · personalised model
      </Typography>
      <Typography variant="caption" sx={{ color: COLORS.textMuted, display: 'block', mb: 1 }}>
        {data.length} windows · 8 s slide
      </Typography>
      <Box sx={{ width: '100%', height: 140 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
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
              ticks={[0, 0.5, 1]}
              tick={{ fill: COLORS.textMuted, fontSize: 10 }}
              stroke={COLORS.bgCardEdge}
              width={36}
            />
            <ReferenceLine y={0.5} stroke={COLORS.bgCardEdge} strokeDasharray="3 3" />
            <Tooltip
              contentStyle={{ background: COLORS.bgCardSolid, border: 'none', borderRadius: 8 }}
              labelFormatter={(v) => `t=${v.toFixed(0)}s`}
              formatter={(v, _, payload) => [
                `${(v * 100).toFixed(1)}%`,
                payload.payload.quality ? 'P(stress)' : 'P(stress) [low quality]',
              ]}
            />
            <Line
              type="monotone"
              dataKey="p"
              stroke={COLORS.violet}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </PanelCard>
  );
}

function QualityBadge({ quality }) {
  if (!quality) {
    return (
      <PanelCard>
        <Typography variant="overline" sx={{ color: COLORS.textMuted, letterSpacing: 1.5 }}>
          Data quality (last 5 min)
        </Typography>
        <Typography variant="body2" sx={{ color: COLORS.textMuted, mt: 2 }}>
          Awaiting first quality report (every 30 s)
        </Typography>
      </PanelCard>
    );
  }
  const headline = quality.pct_unreliable_windows;
  const headlineColor = headline < 5 ? COLORS.green : headline < 15 ? COLORS.amber : COLORS.red;

  return (
    <PanelCard>
      <Typography variant="overline" sx={{ color: COLORS.textMuted, letterSpacing: 1.5 }}>
        Data quality (last 5 min)
      </Typography>
      <Typography variant="h4" sx={{ color: headlineColor, fontWeight: 700, mt: 1 }}>
        {headline.toFixed(1)}%
      </Typography>
      <Typography variant="caption" sx={{ color: COLORS.textMuted }}>
        windows flagged unreliable
      </Typography>
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, mt: 2 }}>
        <Stat label="max gap"        value={`${quality.max_gap_s.toFixed(1)} s`} />
        <Stat label="median interval" value={`${quality.median_interval_s.toFixed(2)} s`} />
        <Stat label="samples ≈ 1 Hz"   value={`${quality.pct_at_1hz.toFixed(1)}%`} />
        <Stat label="windows total"   value={String(quality.n_windows_total)} />
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

function FeatureStrip({ latestPrediction }) {
  const z = latestPrediction?.features_zscore;
  return (
    <PanelCard>
      <Typography variant="overline" sx={{ color: COLORS.textMuted, letterSpacing: 1.5 }}>
        Current window · feature z-scores
      </Typography>
      <Typography variant="caption" sx={{ color: COLORS.textMuted, display: 'block', mb: 2 }}>
        {'|z| > 2 ≈ unusual; sign indicates direction relative to training-set mean'}
      </Typography>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25 }}>
        {FEATURE_NAMES.map((name, i) => (
          <FeatureBar key={name} name={name} value={z ? z[i] : null} />
        ))}
      </Box>
    </PanelCard>
  );
}

function FeatureBar({ name, value }) {
  const pos = value == null ? null : Math.max(0, Math.min(100, 50 + value * 16.67));
  const isExtreme = value != null && Math.abs(value) > 2;

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.25 }}>
        <Typography variant="caption" sx={{ color: COLORS.textMuted, fontFamily: 'monospace' }}>
          {name}
        </Typography>
        <Typography
          variant="caption"
          sx={{ color: isExtreme ? COLORS.amber : 'white', fontFamily: 'monospace' }}
        >
          {value == null ? '—' : value.toFixed(2)}
        </Typography>
      </Box>
      <Box sx={{ position: 'relative', height: 4, bgcolor: COLORS.bgCardEdge, borderRadius: 2 }}>
        {pos != null && (
          <Box
            sx={{
              position: 'absolute',
              left: `${pos}%`,
              top: '50%',
              transform: 'translate(-50%, -50%)',
              width: 8,
              height: 12,
              bgcolor: isExtreme ? COLORS.amber : COLORS.cyan,
              borderRadius: 1,
            }}
          />
        )}
        <Box
          sx={{
            position: 'absolute',
            left: '50%',
            top: 0,
            bottom: 0,
            width: '1px',
            bgcolor: COLORS.textMuted,
            opacity: 0.3,
          }}
        />
      </Box>
    </Box>
  );
}

function EventsLog({ events }) {
  return (
    <PanelCard sx={{ overflow: 'hidden' }}>
      <Typography variant="overline" sx={{ color: COLORS.textMuted, letterSpacing: 1.5 }}>
        System events
      </Typography>
      <Box sx={{ mt: 1.5, maxHeight: 220, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 0.5 }}>
        {events.length === 0 && (
          <Typography variant="body2" sx={{ color: COLORS.textMuted }}>
            (no events yet)
          </Typography>
        )}
        {events.slice().reverse().map((ev, i) => (
          <Box
            key={`${ev.event}-${i}`}
            sx={{
              p: 1,
              borderRadius: 1,
              borderLeft: `3px solid ${
                ev.severity === 'error' ? COLORS.red :
                ev.severity === 'warn' ? COLORS.amber :
                COLORS.cyan
              }`,
              bgcolor: COLORS.bgCardEdge,
            }}
          >
            <Typography variant="caption" sx={{ color: COLORS.textMuted, fontFamily: 'monospace' }}>
              {ev.event}
            </Typography>
            <Typography variant="body2" sx={{ color: 'white' }}>
              {ev.message}
            </Typography>
          </Box>
        ))}
      </Box>
    </PanelCard>
  );
}

// Top-level panel

export default function LivePanel() {
  const state = useWebSocketState();
  const {
    connectionState,
    hello,
    hrBuffer,
    predictionBuffer,
    latestPrediction,
    latestQuality,
    events,
  } = state;

  const headlineP = latestPrediction?.models?.[HEADLINE_MODEL_ID]?.p_stress;
  const headlineQuality = latestPrediction?.quality_pass ?? true;

  const bare = typeof window !== 'undefined' &&
    new URLSearchParams(window.location.search).get('bare') === '1';

  const content = (
    <>
      <Box
        sx={{
          display: 'grid',
          gap: 3,
          gridTemplateColumns: { xs: '1fr', md: '2fr 1fr 1fr' },
          gridTemplateAreas: {
            xs: `'hr' 'gauge' 'conf' 'ribbon' 'qual' 'feat' 'events'`,
            md: `
              'hr     gauge  conf'
              'ribbon ribbon ribbon'
              'qual   feat   events'
            `,
          },
        }}
      >
        <Box sx={{ gridArea: 'hr' }}>      <HRTrace samples={hrBuffer} /></Box>
        <Box sx={{ gridArea: 'gauge' }}>   <StressGauge pStress={headlineP} qualityPass={headlineQuality} /></Box>
        <Box sx={{ gridArea: 'conf' }}>    <ModelConfidenceCard latestPrediction={latestPrediction} /></Box>
        <Box sx={{ gridArea: 'ribbon' }}>  <PredictionRibbon predictionBuffer={predictionBuffer} /></Box>
        <Box sx={{ gridArea: 'qual' }}>    <QualityBadge quality={latestQuality} /></Box>
        <Box sx={{ gridArea: 'feat' }}>    <FeatureStrip latestPrediction={latestPrediction} /></Box>
        <Box sx={{ gridArea: 'events' }}>  <EventsLog events={events} /></Box>
      </Box>
    </>
  );

  if (bare) {
    return <Box sx={{ p: 3, minHeight: '100vh' }}>{content}</Box>;
  }

  return (
    <DashboardLayout>
      <DashboardNavbar title="Live · What's happening now">
        <ConnectionBadge state={connectionState} dataSource={hello?.data_source} />
      </DashboardNavbar>
      {content}
    </DashboardLayout>
  );
}
