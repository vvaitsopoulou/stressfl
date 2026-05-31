/**
 * EvidencePanel — Panel 3: "What does the data actually say?"
 *
 * Empirical findings from offline LOSO ablation across 15 WESAD
 * subjects, four conditions (Teacher CNN-LSTM, Student no-KD, Student
 * KD, Student KD + personalised head), plus a single-subject Fitbit
 * pilot deployment.
 */

import { Box, Card, Typography } from '@mui/material';
import DashboardLayout from 'examples/LayoutContainers/DashboardLayout';
import DashboardNavbar from 'examples/Navbars/DashboardNavbar';
import { useWebSocketState } from 'hooks/WebSocketProvider';

import figConfusion           from 'assets/images/figures/confusion_matrices.png';
import figFitbitTimeline      from 'assets/images/figures/fitbit_prediction_timeline.png';
import figPerSubject          from 'assets/images/figures/per_subject_f1.png';
import figPersonalisationGain from 'assets/images/figures/personalisation_gain.png';
import figROC                 from 'assets/images/figures/roc_curves.png';
import figTeacherStudent      from 'assets/images/figures/teacher_student_agreement.png';
import figTSNE                from 'assets/images/figures/tsne_embeddings.png';

const COLORS = {
  bgCard:      'linear-gradient(127.09deg, rgba(6, 11, 40, 0.94) 19.41%, rgba(10, 14, 35, 0.49) 76.65%)',
  bgCardSolid: '#0f1535',
  bgCardEdge:  '#1d2150',
  bgImageWrap: '#ffffff',
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
        ...sx,
      }}
    >
      {children}
    </Card>
  );
}

function SectionHeader({ number, question, framing }) {
  return (
    <Box sx={{ mb: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1.5, mb: 0.5 }}>
        <Typography
          sx={{
            fontSize: '12px',
            fontWeight: 700,
            color: COLORS.textMuted,
            letterSpacing: 1.5,
            fontFamily: 'monospace',
          }}
        >
          §{number}
        </Typography>
        <Typography sx={{ fontSize: '22px', fontWeight: 700, color: 'white' }}>
          {question}
        </Typography>
      </Box>
      <Typography variant="body2" sx={{ color: COLORS.textMuted, lineHeight: 1.6, maxWidth: 1000 }}>
        {framing}
      </Typography>
    </Box>
  );
}

function FigureCard({ src, alt, caption, note }) {
  return (
    <PanelCard sx={{ p: 2 }}>
      <Box
        sx={{
          background: COLORS.bgImageWrap,
          borderRadius: '12px',
          p: 1,
          mb: 1.5,
          textAlign: 'center',
        }}
      >
        <Box
          component="img"
          src={src}
          alt={alt}
          sx={{
            maxWidth: '100%',
            height: 'auto',
            display: 'block',
            margin: '0 auto',
          }}
        />
      </Box>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25, alignItems: 'flex-start' }}>
        {note && (
          <Box
            sx={{
              borderLeft: `3px solid ${COLORS.amber}`,
              pl: 1,
              py: 0.25,
            }}
          >
            <Typography
              sx={{
                fontSize: '9px',
                fontWeight: 700,
                color: COLORS.amber,
                letterSpacing: 0.5,
                lineHeight: 1.2,
              }}
            >
              CAVEAT
            </Typography>
            <Typography sx={{ fontSize: '11px', color: COLORS.textMuted, lineHeight: 1.5 }}>
              {note}
            </Typography>
          </Box>
        )}
        <Typography variant="caption" sx={{ color: COLORS.textMuted, lineHeight: 1.5 }}>
          {caption}
        </Typography>
      </Box>
    </PanelCard>
  );
}

function MetricCard({ label, value, unit, verdict, verdictColor = COLORS.textMuted, sx = {} }) {
  return (
    <PanelCard sx={{ display: 'flex', flexDirection: 'column', height: '100%', ...sx }}>
      <Typography variant="overline" sx={{ color: COLORS.textMuted, letterSpacing: 1.5, lineHeight: 1.5 }}>
        {label}
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 0.75, mt: 1, mb: 1.5, flexWrap: 'wrap' }}>
        <Typography sx={{ fontSize: '40px', fontWeight: 700, color: 'white', lineHeight: 1 }}>
          {value}
        </Typography>
        {unit && (
          <Typography sx={{ fontSize: '14px', color: COLORS.textMuted }}>
            {unit}
          </Typography>
        )}
      </Box>
      <Box sx={{ borderLeft: `3px solid ${verdictColor}`, pl: 1.25, mt: 'auto' }}>
        <Typography variant="caption" sx={{ color: verdictColor, fontWeight: 700, letterSpacing: 0.5, display: 'block' }}>
          VERDICT
        </Typography>
        <Typography variant="body2" sx={{ color: COLORS.textPrimary, lineHeight: 1.5 }}>
          {verdict}
        </Typography>
      </Box>
    </PanelCard>
  );
}

// Main panel

export default function EvidencePanel() {
  const { hello, connectionState } = useWebSocketState();
  const bare = typeof window !== 'undefined' &&
    new URLSearchParams(window.location.search).get('bare') === '1';

  const content = (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {/* Page intro */}
      <Box>
        <Typography variant="body2" sx={{ color: COLORS.textMuted, maxWidth: 1100, lineHeight: 1.6 }}>
          Empirical findings from leave-one-subject-out (LOSO) cross-validation across 15 WESAD subjects
          with four conditions (Teacher CNN-LSTM, Student no-KD, Student KD, Student KD + personalised
          head), plus a single-subject Fitbit pilot. The deployable artefact is the no-KD student
          (macro-F1 ≈ 0.75, matching published HRV-only WESAD baselines). KD is positioned as an
          empirical investigation of whether multimodal teacher knowledge can transfer across the
          modality gap to a 6-feature HRV-only student. Numbers and figures below are direct exports
          from the offline ablation pipeline; nothing is post-hoc.
        </Typography>
      </Box>

      {/* Headline metrics strip */}
      <Box
        sx={{
          display: 'grid',
          gap: 3,
          gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' },
        }}
      >
        <MetricCard
          label="HRV-only student baseline"
          value="0.75"
          unit="macro-F1, no KD"
          verdict="Matches published HRV-only WESAD ceiling (Prajod & André 2022; Dahal et al. 2023). The deployable artefact."
          verdictColor={COLORS.cyan}
        />
        <MetricCard
          label="KD effect"
          value="+0.009"
          unit="mean Δ macro-F1"
          verdict="Small aggregate gain with high per-subject variance (range −0.07 to +0.08). Nonmonotonic in teacher confidence — consistent with Cho & Hariharan (ICCV 2019)."
          verdictColor={COLORS.amber}
        />
        <MetricCard
          label="Fitbit pilot"
          value="0.4"
          unit="max P(stress), n=1"
          verdict="On real Fitbit-derived HRV the model never crossed 0.5. Suggests a chest-ECG → wrist-PPG domain gap distillation does not bridge."
          verdictColor={COLORS.red}
        />
      </Box>

      {/* SECTION 1 — bottleneck */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <SectionHeader
          number={1}
          question="Why the ceiling exists at all"
          framing={
            'The student receives 6 scalar HRV features per 16-second window: mean_rr, sdnn, rmssd, ' +
            'pnn50, mean_hr, std_hr. The teacher receives raw BVP (32 Hz), EDA (4 Hz), ACC (32 Hz), ' +
            'TEMP (4 Hz). Stress signal that is visible in BVP morphology — systolic upstroke slope, ' +
            'dicrotic notch — but absent from HRV summary statistics is unrecoverable through ' +
            'distillation. Even a perfect teacher cannot transfer information the student\'s input ' +
            'does not carry. This is the dissertation\'s core mechanism: the same bottleneck explains ' +
            'why the no-KD baseline plateaus at ≈0.75, why KD\'s effect is modest, and why the Fitbit ' +
            'pilot fails on a real wrist sensor.'
          }
        />
        <Box
          sx={{
            display: 'grid',
            gap: 3,
            gridTemplateColumns: { xs: '1fr', lg: '1fr 1fr' },
            alignItems: 'start',
          }}
        >
          <FigureCard
            src={figROC}
            alt="ROC curves — all conditions"
            caption="ROC curves for the four conditions on aggregated LOSO test windows. Teacher
              AUC = 0.952. The three student curves are visually close (AUC = 0.870 / 0.875 / 0.876).
              The ≈0.08 AUC gap between teacher and best student is the bottleneck made visible."
          />
          <FigureCard
            src={figConfusion}
            alt="Confusion matrices"
            caption="Confusion matrices per condition (aggregated LOSO test windows). Teacher
              achieves 26807 true-stress, 65734 true-non-stress. The students cluster around
              ≈23000 true-stress and ≈54000–58000 true-non-stress. KD shifts the boundary toward
              higher non-stress precision — false positives drop from 17016 (no-KD) to 13268 (KD),
              a 22% reduction — while paying a small increase in false negatives (6132 → 7569)."
          />
        </Box>
      </Box>

      {/* SECTION 2 — does KD lift the ceiling? */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <SectionHeader
          number={2}
          question="Does KD lift the ceiling?"
          framing={
            'Modestly, with high variance. Mean Δ macro-F1 across 11 visible LOSO folds is +0.0085, ' +
            'with per-fold range −0.07 to +0.08. Six folds gain from KD; five lose. The pattern is ' +
            'principled: KD helps most where the no-KD baseline is weakest (S11: +0.076 with no-KD ' +
            '0.77; S3: +0.060 with no-KD 0.54), and hurts most where the teacher is overconfident ' +
            '(S4: −0.066 with teacher F1 0.98). On near-perfect folds the teacher\'s softmax ' +
            'collapses toward one-hot, soft labels become indistinguishable from hard labels, and ' +
            'distilling the teacher\'s extreme-confidence errors actively misleads the student. This ' +
            'nonmonotonic pattern is exactly what Cho & Hariharan (ICCV 2019) report.'
          }
        />
        <Box
          sx={{
            display: 'grid',
            gap: 3,
            gridTemplateColumns: { xs: '1fr', lg: '1fr 1fr' },
            alignItems: 'start',
          }}
        >
          <FigureCard
            src={figPerSubject}
            alt="Per-subject macro-F1, four conditions"
            caption="Macro-F1 per held-out subject under each LOSO fold. Teacher (blue) leads
              consistently. No-KD (red) and KD (green) trade places per subject — the
              dissertation's core observation that KD's effect is subject-specific. KD + personal
              (yellow) tracks KD almost exactly."
          />
          <FigureCard
            src={figTSNE}
            alt="t-SNE of student embeddings — three conditions"
            caption="t-SNE projections of the 32-dim StudentBase embedding for each condition.
              No-KD (left) shows blobby unstructured class overlap. KD (middle) and KD + personal
              (right) are visually identical — they share the base — and reveal striking radial
              filamentary structure. KD reshapes the geometry of the embedding even where it does
              not lift classification metrics."
          />
        </Box>
      </Box>

      {/* SECTION 3 — does personalisation help? */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <SectionHeader
          number={3}
          question="Does personalisation help on top of KD?"
          framing={
            'On these data, no — at least not measurably. Adding K=10 head fine-tuning steps on ' +
            'each held-out subject\'s calibration windows produces a mean Δ macro-F1 of ≈0.000 over ' +
            'plain KD, with a per-subject range of ±0.002. Half the subjects gain a fraction of a ' +
            'percentage point, half lose one. This is a null result. It does not prove personalisation ' +
            'cannot work; it does say that head-only fine-tuning with K=10 steps on this 6-feature ' +
            'HRV input does not produce subject-specific decision boundaries that meaningfully ' +
            'diverge from the global KD ones. The information bottleneck argument applies again: ' +
            'subject-specific HRV signal that is reachable through fine-tuning may simply not exist ' +
            'in 6 summary statistics.'
          }
        />
        <Box
          sx={{
            display: 'grid',
            gap: 3,
            gridTemplateColumns: { xs: '1fr', lg: '1fr 1fr' },
            alignItems: 'start',
          }}
        >
          <FigureCard
            src={figPersonalisationGain}
            alt="Personalisation gain per subject"
            note="Y-axis range is ±0.002. The 'Mean gain: +0.000' label means the gain is genuinely
              indistinguishable from zero, not that the plotting precision is too coarse."
            caption="Δ macro-F1 from KD to KD + personalised head, per held-out subject. Green bars
              are gains; pink bars are losses. Mean across all subjects ≈ zero."
          />
          <FigureCard
            src={figTeacherStudent}
            alt="Teacher-student agreement scatter"
            caption="Each point is one window: x = teacher P(stress), y = student P(stress). Green
              dots are windows where the student's argmax matches ground truth; pink × marks are
              windows where it doesn't. Dense green clusters in the bottom-left and top-right
              corners are the expected pattern: students agree with the teacher on confident-and-
              correct cases, diverge on ambiguous ones."
          />
        </Box>
      </Box>

      {/* SECTION 4 — Fitbit pilot */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <SectionHeader
          number={4}
          question="Does the lab story transfer to a real wearable?"
          framing={
            'The Fitbit pilot delivers a stark counterpoint. On a single subject wearing a Charge 6 ' +
            'through a known stressor protocol, the personalised student model never produced a ' +
            'P(stress) above 0.4 — never crossing the 0.5 threshold even during the labelled stress ' +
            'block. The HR signal itself shows a clear cardiac response to the stressor; the model ' +
            'simply does not classify it as such. This is plausibly another instance of the bottleneck ' +
            'argument: chest-ECG-derived HRV (training) and wrist-PPG-derived HRV (deployment) share ' +
            'feature names but differ in noise characteristics and dynamic range. The ceiling that ' +
            'limits in-domain performance becomes a wall when you change domains.'
          }
        />
        <FigureCard
          src={figFitbitTimeline}
          alt="Fitbit pilot prediction timeline"
          note="Single-subject pilot (n=1). Findings are illustrative of a domain gap, not a
            statistical claim about its magnitude. A larger Fitbit deployment is the natural next
            step."
          caption="Top: raw Fitbit HR over the recording session. Middle: model P(stress) over time
            (green). Bottom: ground-truth protocol phases (red bars indicate stress). The model's
            P(stress) never exceeds ≈0.4 even during the stress block — predicting non-stress
            throughout. This indicates a domain shift between WESAD chest-ECG features (training)
            and Fitbit wrist-PPG features (deployment) that the present pipeline does not handle."
        />
      </Box>

      {/* Methodological footer */}
      <PanelCard sx={{ py: 2 }}>
        <Typography variant="overline" sx={{ color: COLORS.textMuted, letterSpacing: 1.5, mb: 1, display: 'block' }}>
          Methodological notes
        </Typography>
        <Box
          sx={{
            display: 'grid',
            gap: 1.5,
            gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' },
            color: COLORS.textPrimary,
            fontSize: '13px',
            lineHeight: 1.6,
          }}
        >
          <Box>
            <b>Cross-validation:</b> leave-one-subject-out across 15 WESAD subjects. Each fold
            trains on 14 subjects, evaluates on the held-out one. Calibration windows (first 15
            minutes) are excluded from evaluation in the personalised condition.
          </Box>
          <Box>
            <b>Conditions:</b> Teacher = CNN-LSTM on chest BVP/EDA/ACC/TEMP (~80K params); Student
            no-KD = MLP on 6 HRV features trained from scratch; Student KD = MLP trained with
            hybrid distillation loss (logit KD α=0.3 + FitNets feature KD β=5 + class-weighted
            cross-entropy on hard labels); Student KD + personal = same as KD plus K=10 head
            fine-tuning steps on the held-out subject's calibration split.
          </Box>
          <Box>
            <b>Features:</b> mean_rr, sdnn, rmssd, pnn50, mean_hr, std_hr — six time-domain HRV
            features per 16-second window with 8-second slide. Z-scored using training-set μ/σ.
          </Box>
          <Box>
            <b>Reference points:</b> KD literature on bottlenecked students — Cho & Hariharan ICCV
            2019; HRV-only WESAD baseline ceiling — Prajod & André 2022, Dahal et al. 2023; FedRep
            base/head split — Collins et al. ICML 2021.
          </Box>
        </Box>
      </PanelCard>
    </Box>
  );

  if (bare) {
    return <Box sx={{ p: 3, minHeight: '100vh' }}>{content}</Box>;
  }

  return (
    <DashboardLayout>
      <DashboardNavbar title="Evidence · What does the data say?">
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
