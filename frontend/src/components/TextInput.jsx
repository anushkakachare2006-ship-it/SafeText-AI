import { ArrowUpRight, RotateCcw } from 'lucide-react'

const MAX_LENGTH = 5000

export default function TextInput({
  text,
  onTextChange,
  onSubmit,
  onClear,
  onExample,
  isLoading,
}) {
  const examples = [
    'Everyone deserves respect, regardless of their background.',
    'I strongly disagree with your argument and the way you presented it.',
    'I hate waiting in long lines, especially after a busy day.',
  ]

  return (
    <section className="composer" aria-labelledby="composer-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Text analysis</p>
          <h2 id="composer-title">Check a message</h2>
        </div>
        <span className="input-limit">Up to {MAX_LENGTH.toLocaleString()} characters</span>
      </div>

      <label className="sr-only" htmlFor="message-text">Text to analyze</label>
      <textarea
        id="message-text"
        className="message-input"
        value={text}
        onChange={(event) => onTextChange(event.target.value.slice(0, MAX_LENGTH))}
        placeholder="Paste or write a sentence, paragraph, or short message..."
        maxLength={MAX_LENGTH}
        rows={8}
        disabled={isLoading}
      />

      <div className="input-meta">
        <span>Text stays in this session and is not saved.</span>
        <span>{text.length.toLocaleString()} / {MAX_LENGTH.toLocaleString()}</span>
      </div>

      <div className="action-row">
        <button className="button button-primary" type="button" onClick={onSubmit} disabled={isLoading || !text.trim()}>
          {isLoading ? <span className="spinner" aria-hidden="true" /> : <ArrowUpRight size={17} aria-hidden="true" />}
          {isLoading ? 'Analyzing...' : 'Check text'}
        </button>
        <button className="button button-quiet" type="button" onClick={onClear} disabled={isLoading && !text}>
          <RotateCcw size={16} aria-hidden="true" />
          Clear
        </button>
      </div>

      <div className="examples-block">
        <p className="examples-label">Try an example</p>
        <div className="example-list">
          {examples.map((example, index) => (
            <button
              className="example-button"
              key={example}
              type="button"
              onClick={() => onExample(example)}
              disabled={isLoading}
            >
              <span className={`example-index example-index-${index + 1}`}>0{index + 1}</span>
              <span>{example}</span>
            </button>
          ))}
        </div>
      </div>
    </section>
  )
}