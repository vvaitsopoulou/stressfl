/**
 * MoodIcon — smiley or frown SVG, colour-overridable.
 *
 * The path data here is extracted from two Noun Project icons:
 *   • Smiley by Kero       (noun-smiley-face-8374531.svg)
 *   • Frown by White Spaces Inc.  (noun-frown-707840.svg)
 *
 * Original SVG files preserved at src/assets/icons/ with their
 * embedded "Created by ... from Noun Project" attribution text.
 * That attribution is the licence-required acknowledgement; this
 * component renders just the glyph for clean display in the gauge.
 *
 * Licence: Noun Project royalty-free / CC-BY (see
 *   https://thenounproject.com/legal/terms-of-use/#icon-licenses).
 * Attribution should also appear in the dissertation's
 * acknowledgements or figures-credits section.
 *
 * Both source SVGs were drawn on roughly the same canvas (100×100
 * unit viewBox with attribution text taking up the bottom 25 units).
 * We use viewBox="0 0 100 100" here to crop the attribution text
 * out of the rendered glyph while keeping it intact in the source
 * files on disk.
 */

// Path data extracted from the source SVGs. Names match the icon
// represented (NOT the source filename — Noun Project filenames can
// be confusing because numeric IDs aren't meaningful).
//
// Smiley = "m50 5c..." path (originally from noun-smiley-face-8374531.svg)
// Frown  = "M49.70..." path (originally from noun-frown-707840.svg)
const SMILEY_PATH = 'm50 5c24.801 0 45 20.199 45 45s-20.199 45-45 45-45-20.199-45-45 20.199-45 45-45zm17.301 18.5c-3.6992 0-6.6016 4.1016-6.6016 9.3008s3 9.3008 6.6016 9.3008c3.6016 0 6.6016-4.1016 6.6016-9.3008s-3-9.3008-6.6016-9.3008zm12 34.102c-7.5 2.3008-17.898 3.8008-29.301 3.8008-11.398 0-21.801-1.5-29.301-3.8008 1.1016 15.301 13.801 27.301 29.301 27.301s28.199-12 29.301-27.301zm-46.602-34.102c-3.6992 0-6.6016 4.1016-6.6016 9.3008s3 9.3008 6.6016 9.3008c3.6016 0 6.6016-4.1016 6.6016-9.3008s-3-9.3008-6.6016-9.3008z';

const FROWN_PATH = 'M49.7032894,61 C57.0291615,61 63.7929239,64.370227 68.2230576,70.0319919 C69.2440744,71.3368638 71.1295798,71.5669738 72.4344517,70.545957 C73.7393236,69.5249403 73.9694336,67.6394349 72.9484168,66.334563 C67.3929162,59.2345654 58.8944992,55 49.7032894,55 C41.2294988,55 33.3281167,58.5962658 27.7701369,64.7718388 C26.6617638,66.0033724 26.7616051,67.9002416 27.9931388,69.0086147 C29.2246724,70.1169878 31.1215416,70.0171465 32.2299147,68.7856129 C36.6616795,63.8613979 42.948478,61 49.7032894,61 Z M50,90 C72.09139,90 90,72.09139 90,50 C90,27.90861 72.09139,10 50,10 C27.90861,10 10,27.90861 10,50 C10,72.09139 27.90861,90 50,90 Z M64,46 C66.7614237,46 69,43.7614237 69,41 C69,38.2385763 66.7614237,36 64,36 C61.2385763,36 59,38.2385763 59,41 C59,43.7614237 61.2385763,46 64,46 Z M36,46 C38.7614237,46 41,43.7614237 41,41 C41,38.2385763 38.7614237,36 36,36 C33.2385763,36 31,38.2385763 31,41 C31,43.7614237 33.2385763,46 36,46 Z';

export default function MoodIcon({ mood, size = 36, color = 'white', sx = {} }) {
  // mood: 'smiley' | 'frown' | null  (null/anything else → blank)
  if (mood !== 'smiley' && mood !== 'frown') {
    return null;
  }
  const path = mood === 'smiley' ? SMILEY_PATH : FROWN_PATH;
  // Both icons fit comfortably in a 0–100 box. The frown path uses
  // m50 5 ... z so its glyph bounds are roughly y=5..95.
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      xmlns="http://www.w3.org/2000/svg"
      style={{ display: 'block', ...sx }}
      aria-label={mood === 'smiley' ? 'low stress' : 'high stress'}
    >
      <path d={path} fill={color} fillRule="evenodd" />
    </svg>
  );
}
