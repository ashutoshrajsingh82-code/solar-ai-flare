import React from 'react';

const CLASSES = ['Quiet', 'Pre-Flare', 'Flare', 'Decay'];

export default function ConfusionMatrix({
  matrix = [
    [1240, 45, 12, 8],
    [32, 280, 24, 10],
    [5, 15, 195, 20],
    [12, 8, 18, 220],
  ],
}) {
  // Calculate max for color scale
  const flattened = matrix.flat();
  const maxVal = Math.max(...flattened, 1);

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-xs uppercase tracking-wider font-semibold text-slate-300 font-mono">
          Nowcast 4-Class Confusion Matrix
        </h4>
        <span className="text-[11px] font-mono text-slate-400">
          Normalized Counts
        </span>
      </div>

      <div className="overflow-x-auto">
        <div className="min-w-[340px]">
          <div className="grid grid-cols-5 gap-1 text-center font-mono text-xs">
            <div className="p-2 text-[10px] text-slate-500 font-sans flex items-center justify-center">
              Actual \ Pred
            </div>
            {CLASSES.map((cls) => (
              <div key={cls} className="p-2 font-semibold text-slate-300 text-[11px] bg-space-850 rounded">
                {cls}
              </div>
            ))}

            {matrix.map((row, rIdx) => (
              <React.Fragment key={`row-${rIdx}`}>
                <div className="p-2 font-semibold text-slate-300 text-[11px] bg-space-850 rounded flex items-center justify-center">
                  {CLASSES[rIdx]}
                </div>
                {row.map((val, cIdx) => {
                  const isDiag = rIdx === cIdx;
                  const intensity = Math.min(1, val / maxVal);
                  const bg = isDiag
                    ? `rgba(16, 185, 129, ${Math.max(0.15, intensity * 0.8)})` // Emerald for correct
                    : val > 0
                    ? `rgba(244, 63, 94, ${Math.max(0.1, intensity * 0.7)})` // Rose for misclassification
                    : 'rgba(15, 23, 42, 0.4)';

                  return (
                    <div
                      key={`cell-${rIdx}-${cIdx}`}
                      style={{ backgroundColor: bg }}
                      className={`p-3 rounded border border-space-800/60 flex flex-col items-center justify-center transition-all ${
                        isDiag ? 'font-bold text-white shadow-sm' : 'text-slate-300'
                      }`}
                    >
                      <span className="text-xs font-mono">{val}</span>
                      <span className="text-[9px] text-slate-400">
                        {((val / Math.max(1, row.reduce((a, b) => a + b, 0))) * 100).toFixed(0)}%
                      </span>
                    </div>
                  );
                })}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between mt-3 text-[11px] font-mono text-slate-400 border-t border-space-800 pt-2">
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded bg-emerald-500 inline-block" />
          Diagonal: Correct Classifications
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded bg-rose-500/70 inline-block" />
          Off-diagonal: False Transitions
        </span>
      </div>
    </div>
  );
}
