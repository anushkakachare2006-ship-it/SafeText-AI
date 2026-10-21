import { AlertTriangle, Check, Info } from 'lucide-react'

export default function PredictionResult({ result }) {
  if (!result) {
    return (
      <section className="result-panel result-empty" aria-labelledby="result-title">
        <div className="empty-mark"><Info size={21} aria-hidden="true" /></div>
        <p className="eyebrow">Your result</p>
        <h2 id="result-title">Ready when you are</h2>
        <p className="empty-copy">Submit text to see the model's classification and estimated hate probability.</p>
        <div className="empty-rule" />
        <p className="empty-note">Each check is analyzed by the trained machine learning model.</p>
      </section>
    )
  }

  const isHate = result.is_hate
  const percentage = Number(result.hate_probability)
  const safePercentage = Math.min(100, Math.max(0, Number.isFinite(percentage) ? percentage : 0))

  return (
    <section className={`result-panel ${isHate ? 'result-hate' : 'result-clear'}`} aria-labelledby="result-title" aria-live="polite">
      <div className="result-topline">
        <span className="eyebrow">Model assessment</span>
        <span className="result-symbol">
          {isHate ? <AlertTriangle size={20} aria-hidden="true" /> : <Check size={20} aria-hidden="true" />}
        </span>
      </div>
      <h2 id="result-title">{result.result}</h2>
      <p className="result-message">{result.message}</p>

      <div className="probability-block">
        <div className="probability-heading">
          <span>Estimated hate probability</span>
          <strong>{Math.round(safePercentage)}%</strong>
        </div>
        <div
          className="probability-track"
          role="progressbar"
          aria-label="Estimated hate probability"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(safePercentage)}
        >
          <span className="probability-fill" style={{ width: `${safePercentage}%` }} />
        </div>
        <div className="scale-labels"><span>0%</span><span>100%</span></div>
      </div>

      <div className="result-caveat">
        <Info size={15} aria-hidden="true" />
        <p>This is a model estimate, not a definitive judgment. Offensive language and neither are grouped into “No Hate Speech.”</p>
      </div>
    </section>
  )
}