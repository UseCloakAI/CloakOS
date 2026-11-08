/* Keeps the 3D views smooth on modest hardware: renders sharper when there's headroom,
   backs off the pixel ratio when frames take too long. */
export function adaptive(renderer, { min = 0.6, max = 1.5 } = {}) {
  let ratio = Math.min(max, window.devicePixelRatio || 1), t0 = performance.now(), n = 0, slow = 0, last = t0;
  renderer.setPixelRatio(ratio);
  return () => {
    const now = performance.now(), dt = now - last;
    last = now;
    if (dt > 250) { t0 = now; n = 0; return; }       // tab was hidden or the page was idle: don't count it
    n++;
    if (n < 45) return;
    const avg = (now - t0) / n;
    const next = avg > 26 ? Math.max(min, ratio * 0.8) : avg < 15 && ratio < Math.min(max, window.devicePixelRatio || 1) ? Math.min(max, ratio * 1.15) : ratio;
    if (Math.abs(next - ratio) > 0.01) { ratio = next; renderer.setPixelRatio(ratio); renderer.setSize(renderer.domElement.clientWidth, renderer.domElement.clientHeight, false); }
    t0 = now; n = 0; slow = 0;
  };
}
