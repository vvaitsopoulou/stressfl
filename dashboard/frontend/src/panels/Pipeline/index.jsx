/**
 * PipelinePanel — Panel 4: "How it works"
 *
 * Animated SVG diagram of the data flow from Fitbit watch to stress
 * prediction. Stages are clickable, and clicking opens a side panel with
 * details about that stage.
 */

import { useState } from 'react';
import { Box, Card, Typography } from '@mui/material';
import DashboardLayout from 'examples/LayoutContainers/DashboardLayout';
import DashboardNavbar from 'examples/Navbars/DashboardNavbar';
import { useWebSocketState } from 'hooks/WebSocketProvider';
import { FEATURE_NAMES } from 'protocol';

const COLORS = {
  bgCard:      'linear-gradient(127.09deg, rgba(6, 11, 40, 0.94) 19.41%, rgba(10, 14, 35, 0.49) 76.65%)',
  bgCardSolid: '#0f1535',
  bgCardEdge:  '#1d2150',
  bgStage:     '#1a1f5c',
  bgStageHov:  '#252b70',
  textPrimary: '#e8eaf3',
  textMuted:   '#a3aed0',
  cyan:        '#2cd9ff',
  blue:        '#0075ff',
  violet:      '#8b5cf6',
  green:       '#01b574',
  amber:       '#ffb547',
  red:         '#e31a1a',
};

const STAGE_COLORS = {
  fitbit:       COLORS.cyan,
  preprocessor: COLORS.cyan,
  windowing:    COLORS.cyan,
  features:     COLORS.blue,
  base:         COLORS.violet,
  heads:        COLORS.violet,
  output:       COLORS.green,
};

const STATUS = {
  real:    { label: 'IMPLEMENTED', color: COLORS.green },
  planned: { label: 'PLANNED',     color: COLORS.amber },
};

const STAGES = [
  {
    id: 'fitbit',
    title: 'Fitbit',
    subtitle: 'Charge 6 watch',
    status: 'planned',
    blurb: 'Raw irregular 1 Hz heart-rate samples from the wrist watch, polled via the Fitbit Web API every 60 s after the phone app syncs.',
    details: [
      ['Sampling',       'Nominally 1 Hz, actually irregular'],
      ['Resolution',     'Integer BPM'],
      ['Polling cadence','60 s (foreground)'],
      ['Sync mechanism', 'Phone app → Fitbit cloud → API'],
    ],
  },
  {
    id: 'preprocessor',
    title: 'Preprocessor',
    subtitle: 'Spline + gap flag',
    status: 'real',
    blurb: 'Cubic spline interpolation onto a uniform 1 Hz grid, then physiological clipping to [40, 200] BPM. Samples in raw-data gaps > 5 s are flagged as unreliable.',
    details: [
      ['File',           'fitbit_preprocessor.py'],
      ['Method',         'scipy.CubicSpline'],
      ['Clipping',       'HR_CLIP_LOW=40, HR_CLIP_HIGH=200'],
      ['Gap threshold',  '5 s (Bottcher et al. 2022)'],
    ],
  },
  {
    id: 'windowing',
    title: 'Windowing',
    subtitle: '16 s / 8 s slide',
    status: 'real',
    blurb: 'Sliding window of 16 seconds with 8 second step (50% overlap). Windows with > 20% flagged samples are excluded from labelled evaluation but still emitted as predictions with quality_pass=false.',
    details: [
      ['Window length',  '16 s'],
      ['Slide',          '8 s (50% overlap)'],
      ['Samples/window', '16 (at 1 Hz)'],
      ['Quality cutoff', '> 20% flagged → exclude'],
    ],
  },
  {
    id: 'features',
    title: 'Features',
    subtitle: '6 HRV time-domain',
    status: 'real',
    blurb: 'Pseudo-RR conversion (60000 / HR), then six time-domain HRV features per window. Z-scored using training-set statistics, never deployment-set.',
    details: [
      ['File',           'hrv_feature_extractor.py'],
      ['Pseudo-RR',      'rr_ms = 60000 / hr_bpm'],
      ['Features',       FEATURE_NAMES.join(', ')],
      ['Normalisation',  'z-score (train-set μ, σ)'],
      ['Validation',     'Castaldo 2019, ultra-short HRV'],
    ],
  },
  {
    id: 'base',
    title: 'StudentBase',
    subtitle: 'Linear → 64 → 32',
    status: 'real',
    blurb: 'Shared MLP that maps 6 features to a 32-dim embedding. In FedRep this is the part aggregated across clients. global_kd and personalised share these weights; no_kd has its own.',
    details: [
      ['File',           'student_model.py · StudentBase'],
      ['Layer 1',        'Linear(6, 64) → ReLU → BN → Dropout 0.2'],
      ['Layer 2',        'Linear(64, 32) → ReLU → BN → Dropout 0.2'],
      ['Output dim',     '32 (matches teacher TEACHER_FEATURE_DIM)'],
      ['Shared by',      'global_kd, personalised (NOT no_kd)'],
    ],
  },
  {
    id: 'heads',
    title: 'StudentHead × 3',
    subtitle: 'Linear → 16 → 2',
    status: 'real',
    blurb: 'Three personal classification heads, one per model condition. Each is a tiny 2-layer MLP from 32 to 2 logits. In FedRep this is the part that stays local — never communicated.',
    details: [
      ['File',           'student_model.py · StudentHead'],
      ['Layer 1',        'Linear(32, 16) → ReLU'],
      ['Layer 2',        'Linear(16, 2)'],
      ['Output',         '2 logits'],
      ['Heads',          'no_kd, global_kd, personalised'],
    ],
  },
  {
    id: 'output',
    title: 'P(stress)',
    subtitle: 'softmax · per model',
    status: 'real',
    blurb: 'Softmax over the 2 logits gives the probability of the "stress" class. The personalised model is the headline value displayed in Panel 1; all three are tracked in Panel 2.',
    details: [
      ['Activation',     'softmax(logits)'],
      ['Cadence',        'one prediction per 8 s'],
      ['Latency',        'sub-ms on CPU'],
      ['Headline model', 'personalised (d′)'],
    ],
  },
];

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
        ...sx,
      }}
    >
      {children}
    </Card>
  );
}

function StageBox({ x, y, w, h, stage, isSelected, onClick, currentValue }) {
  const colour = STAGE_COLORS[stage.id];
  const status = STATUS[stage.status];
  return (
    <foreignObject x={x} y={y} width={w} height={h}>
      <Box
        onClick={onClick}
        sx={{
          width: '100%',
          height: '100%',
          background: isSelected ? COLORS.bgStageHov : COLORS.bgStage,
          border: `1.5px solid ${isSelected ? colour : COLORS.bgCardEdge}`,
          boxShadow: isSelected ? `0 0 24px -8px ${colour}` : 'none',
          borderRadius: '12px',
          padding: 1.25,
          cursor: 'pointer',
          display: 'flex',
          flexDirection: 'column',
          gap: 0.25,
          transition: 'all 0.18s ease',
          '&:hover': { background: COLORS.bgStageHov, borderColor: colour },
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <Box>
            <Typography sx={{ fontSize: '13px', fontWeight: 700, color: 'white', lineHeight: 1.2 }}>
              {stage.title}
            </Typography>
            <Typography sx={{ fontSize: '10px', color: COLORS.textMuted, lineHeight: 1.2 }}>
              {stage.subtitle}
            </Typography>
          </Box>
          <Box
            sx={{
              fontSize: '8px',
              fontWeight: 700,
              letterSpacing: 0.5,
              color: status.color,
              border: `1px solid ${status.color}`,
              borderRadius: '4px',
              px: 0.5,
              py: 0.1,
              flexShrink: 0,
            }}
          >
            {status.label}
          </Box>
        </Box>
        {currentValue && (
          <Box
            sx={{
              mt: 'auto',
              fontFamily: 'monospace',
              fontSize: '10px',
              color: colour,
              transition: 'color 0.2s ease',
              wordBreak: 'break-all',
              lineHeight: 1.3,
            }}
          >
            {currentValue}
          </Box>
        )}
      </Box>
    </foreignObject>
  );
}

function EdgeArrow({ id, d, colour = COLORS.cyan }) {
  return (
    <g>
      <path
        d={d}
        fill="none"
        stroke={colour}
        strokeWidth={1.5}
        strokeOpacity={0.35}
        strokeLinecap="round"
        markerEnd={`url(#arrow-${id})`}
      />
      {[0, 0.33, 0.66].map((delay, i) => (
        <circle
          key={i}
          r={3}
          fill={colour}
          style={{
            offsetPath: `path('${d}')`,
            offsetRotate: '0deg',
            animation: `pkt-flow 1.8s ${delay * 1.8}s linear infinite`,
          }}
        />
      ))}
    </g>
  );
}

function ArrowMarker({ id, colour }) {
  return (
    <marker
      id={`arrow-${id}`}
      viewBox="0 0 10 10"
      refX="8"
      refY="5"
      markerWidth="6"
      markerHeight="6"
      orient="auto"
    >
      <path d="M 0 0 L 10 5 L 0 10 z" fill={colour} fillOpacity={0.55} />
    </marker>
  );
}

function StageDetails({ stage, onClose }) {
  if (!stage) {
    return (
      <PanelCard sx={{ height: '100%' }}>
        <Typography variant="overline" sx={{ color: COLORS.textMuted, letterSpacing: 1.5 }}>
          Stage details
        </Typography>
        <Box
          sx={{
            mt: 2,
            p: 2,
            border: `1px dashed ${COLORS.bgCardEdge}`,
            borderRadius: '12px',
            textAlign: 'center',
          }}
        >
          <Typography variant="body2" sx={{ color: COLORS.textMuted }}>
            Click any stage in the diagram to inspect it.
          </Typography>
        </Box>
      </PanelCard>
    );
  }
  const colour = STAGE_COLORS[stage.id];
  const status = STATUS[stage.status];
  return (
    <PanelCard sx={{ height: '100%', borderColor: colour }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
        <Box>
          <Typography variant="overline" sx={{ color: COLORS.textMuted, letterSpacing: 1.5 }}>
            Stage details
          </Typography>
          <Typography sx={{ fontSize: '20px', fontWeight: 700, color: 'white' }}>
            {stage.title}
          </Typography>
          <Typography variant="caption" sx={{ color: COLORS.textMuted }}>
            {stage.subtitle}
          </Typography>
        </Box>
        <Box
          onClick={onClose}
          sx={{
            cursor: 'pointer',
            color: COLORS.textMuted,
            fontSize: 20,
            lineHeight: 1,
            px: 1,
            '&:hover': { color: 'white' },
          }}
        >
          ×
        </Box>
      </Box>
      <Box
        sx={{
          display: 'inline-block',
          mb: 2,
          fontSize: '9px',
          fontWeight: 700,
          letterSpacing: 0.5,
          color: status.color,
          border: `1px solid ${status.color}`,
          borderRadius: '4px',
          px: 0.7,
          py: 0.2,
        }}
      >
        {status.label}
      </Box>
      <Typography variant="body2" sx={{ color: COLORS.textPrimary, mb: 2, lineHeight: 1.5 }}>
        {stage.blurb}
      </Typography>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
        {stage.details.map(([k, v]) => (
          <Box key={k} sx={{ display: 'flex', gap: 1, fontSize: '12px' }}>
            <Box
              sx={{
                color: COLORS.textMuted,
                width: '110px',
                flexShrink: 0,
                fontFamily: 'monospace',
              }}
            >
              {k}
            </Box>
            <Box sx={{ color: 'white', fontFamily: 'monospace', wordBreak: 'break-word' }}>
              {v}
            </Box>
          </Box>
        ))}
      </Box>
    </PanelCard>
  );
}

function PipelineDiagram({ selectedId, onSelect, latestPrediction, hrBuffer }) {
  // Live values per stage.
  const latestHR = hrBuffer.length > 0 ? hrBuffer[hrBuffer.length - 1].hr : null;
  const featuresZ = latestPrediction?.features_zscore;
  const baseEmb = latestPrediction?.models?.personalised?.embedding_summary;
  const headLogits = latestPrediction?.models?.personalised?.logits;
  const pStress = latestPrediction?.models?.personalised?.p_stress;
  const winId = latestPrediction?.window_id;

  const valueByStage = {
    fitbit:       latestHR != null ? `HR = ${latestHR} BPM` : 'no data yet',
    preprocessor: hrBuffer.length > 0 ? `1 Hz · ${hrBuffer.length} samples` : 'no data yet',
    windowing:    winId != null ? `window #${winId}` : 'no windows yet',
    features:     featuresZ ? `z = [${featuresZ.map((v) => v.toFixed(1)).join(', ')}]` : 'awaiting first window',
    base:         baseEmb ? `μ=${baseEmb.mean.toFixed(2)} σ=${baseEmb.std.toFixed(2)}` : 'awaiting',
    heads:        headLogits ? `logits = [${headLogits.map((v) => v.toFixed(1)).join(', ')}]` : 'awaiting',
    output:       pStress != null ? `${(pStress * 100).toFixed(0)}% P(stress)` : 'awaiting',
  };

  const W = 200;
  const H = 90;
  const Y1 = 50;
  const Y2 = 320;
  const ROW1_X = [40, 280, 520, 760];
  const ROW2_X = [280, 520, 760];

  const stagePositions = {
    fitbit:       { x: ROW1_X[0], y: Y1, w: W, h: H },
    preprocessor: { x: ROW1_X[1], y: Y1, w: W, h: H },
    windowing:    { x: ROW1_X[2], y: Y1, w: W, h: H },
    features:     { x: ROW1_X[3], y: Y1, w: W, h: H },
    base:         { x: ROW2_X[0], y: Y2, w: W, h: H },
    heads:        { x: ROW2_X[1], y: Y2, w: W, h: H },
    output:       { x: ROW2_X[2], y: Y2, w: W, h: H },
  };

  const edges = [
    {
      id: 'fitbit', d: `M ${ROW1_X[0] + W} ${Y1 + H/2} L ${ROW1_X[1]} ${Y1 + H/2}`,
      colour: STAGE_COLORS.fitbit,
    },
    {
      id: 'preprocessor', d: `M ${ROW1_X[1] + W} ${Y1 + H/2} L ${ROW1_X[2]} ${Y1 + H/2}`,
      colour: STAGE_COLORS.preprocessor,
    },
    {
      id: 'windowing', d: `M ${ROW1_X[2] + W} ${Y1 + H/2} L ${ROW1_X[3]} ${Y1 + H/2}`,
      colour: STAGE_COLORS.windowing,
    },
    // Diagonal: features (row 1, far right) → base (row 2, second from left)
    {
      id: 'features',
      d: `M ${ROW1_X[3] + W/2} ${Y1 + H} Q ${ROW1_X[3] + W/2} ${(Y1 + H + Y2)/2}, ${ROW2_X[0] + W/2} ${Y2}`,
      colour: STAGE_COLORS.features,
    },
    {
      id: 'base', d: `M ${ROW2_X[0] + W} ${Y2 + H/2} L ${ROW2_X[1]} ${Y2 + H/2}`,
      colour: STAGE_COLORS.base,
    },
    {
      id: 'heads', d: `M ${ROW2_X[1] + W} ${Y2 + H/2} L ${ROW2_X[2]} ${Y2 + H/2}`,
      colour: STAGE_COLORS.heads,
    },
  ];

  return (
    <Box
      sx={{
        '@keyframes pkt-flow': {
          '0%':   { offsetDistance: '0%',   opacity: 0 },
          '10%':  { opacity: 1 },
          '90%':  { opacity: 1 },
          '100%': { offsetDistance: '100%', opacity: 0 },
        },
      }}
    >
      <svg viewBox="0 0 1000 460" style={{ width: '100%', height: 'auto' }}>
        <defs>
          {Object.entries(STAGE_COLORS).map(([id, colour]) => (
            <ArrowMarker key={id} id={id} colour={colour} />
          ))}
        </defs>
        <text x={10} y={Y1 - 10} fill={COLORS.textMuted} fontSize={10} fontFamily="monospace" letterSpacing={1.5}>
          DATA PIPELINE
        </text>
        <text x={10} y={Y2 - 10} fill={COLORS.textMuted} fontSize={10} fontFamily="monospace" letterSpacing={1.5}>
          MODEL
        </text>
        {edges.map((e) => (
          <EdgeArrow key={e.id} id={e.id} d={e.d} colour={e.colour} />
        ))}
        {STAGES.map((stage) => {
          const pos = stagePositions[stage.id];
          return (
            <StageBox
              key={stage.id}
              {...pos}
              stage={stage}
              isSelected={selectedId === stage.id}
              onClick={() => onSelect(stage.id)}
              currentValue={valueByStage[stage.id]}
            />
          );
        })}
      </svg>
    </Box>
  );
}

export default function PipelinePanel() {
  const { latestPrediction, hrBuffer, hello, connectionState } = useWebSocketState();
  const [selectedId, setSelectedId] = useState(null);
  const selectedStage = STAGES.find((s) => s.id === selectedId);

  const handleSelect = (id) => {
    setSelectedId((curr) => (curr === id ? null : id));
  };

  const bare = typeof window !== 'undefined' &&
    new URLSearchParams(window.location.search).get('bare') === '1';

  const content = (
    <Box
      sx={{
        display: 'grid',
        gap: 3,
        gridTemplateColumns: { xs: '1fr', lg: '2fr 1fr' },
      }}
    >
      <PanelCard sx={{ height: '100%' }}>
        <Typography variant="overline" sx={{ color: COLORS.textMuted, letterSpacing: 1.5 }}>
          Pipeline · watch to prediction
        </Typography>
        <Typography variant="caption" sx={{ color: COLORS.textMuted, display: 'block', mb: 1 }}>
          Click any stage to inspect it. Live values update every 8 s with each new prediction.
        </Typography>
        <PipelineDiagram
          selectedId={selectedId}
          onSelect={handleSelect}
          latestPrediction={latestPrediction}
          hrBuffer={hrBuffer}
        />
      </PanelCard>
      <StageDetails stage={selectedStage} onClose={() => setSelectedId(null)} />
    </Box>
  );

  if (bare) {
    return <Box sx={{ p: 3, minHeight: '100vh' }}>{content}</Box>;
  }

  return (
    <DashboardLayout>
      <DashboardNavbar title="Pipeline · How it works">
        <ConnectionBadge state={connectionState} dataSource={hello?.data_source} />
      </DashboardNavbar>
      {content}
    </DashboardLayout>
  );
}

function ConnectionBadge({ state, dataSource }) {
  const labelMap = {
    idle:       { text: 'idle',                       color: COLORS.textMuted },
    connecting: { text: 'connecting…',                color: COLORS.amber },
    open:       { text: `live · ${dataSource ?? '…'}`, color: COLORS.green },
    closed:     { text: 'reconnecting',               color: COLORS.amber },
    error:      { text: 'error',                      color: COLORS.red },
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
