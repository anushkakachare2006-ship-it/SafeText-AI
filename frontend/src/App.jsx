import { useEffect, useState } from 'react'
import { Activity, ShieldCheck } from 'lucide-react'
import { getHealth, predictText } from './api.js'
import PredictionResult from './components/PredictionResult.jsx'
import TextInput from './components/TextInput.jsx'

function describeError(error) {
  const message = error instanceof Error ? error.message : ''
  if (message.toLowerCase().includes('train the model')) {
    return 'The model is not trained yet. Run the training command in the project README, then try again.'
  }
  if (message.toLowerCase().includes('fetch')) {
    return 'Could not reach the backend. Start the FastAPI server at http://127.0.0.1:8000 and check the API URL.'
  }
  return message || 'Something went wrong while analyzing the text. Please try again.'
}

export default function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [health, setHealth] = useState('checking')
  const [modelAvailable, setModelAvailable] = useState(false)

  useEffect(() => {
    let isActive = true
    getHealth()
      .then((status) => {
        if (!isActive) return
        setHealth('connected')
        setModelAvailable(Boolean(status.model_available))
      })
      .catch(() => {
        if (isActive) setHealth('offline')
      })
    return () => { isActive = false }
  }, [])

  async function handleSubmit() {
    if (!text.trim() || isLoading) return
    setIsLoading(true)
    setError('')
    setResult(null)
    try {
      const prediction = await predictText(text.trim())
      if (
        typeof prediction.is_hate !== 'boolean'
        || typeof prediction.result !== 'string'
        || typeof prediction.hate_probability !== 'number'
        || !Number.isFinite(prediction.hate_probability)
        || prediction.hate_probability < 0
        || prediction.hate_probability > 100
      ) {
        throw new Error('The backend returned an unexpected response. Restart the backend and try again.')
      }
      setResult(prediction)
    } catch (requestError) {
      setError(describeError(requestError))
    } finally {
      setIsLoading(false)
    }
  }

  function handleClear() {
    setText('')
    setResult(null)
    setError('')
    setIsLoading(false)
  }

  const statusText = health === 'checking'
    ? 'Checking service'
    : health === 'offline'
      ? 'Backend offline'
      : modelAvailable
        ? 'Model ready'
        : 'Model not trained'

  return (
    <main className="page-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="SafeText AI home">
          <span className="brand-mark"><ShieldCheck size={21} strokeWidth={2.2} aria-hidden="true" /></span>
          <span className="brand-name">SafeText <span>AI</span></span>
        </a>
        <div className={`service-status status-${health}-${modelAvailable}`} aria-live="polite">
          <span className="status-dot" />
          {statusText}
        </div>
      </header>

      <section className="intro" id="top">
        <div className="intro-copy">
          <p className="eyebrow intro-eyebrow"><Activity size={14} aria-hidden="true" /> AI-powered text analysis</p>
          <h1>Understand the tone.<br /><span>Respond with care.</span></h1>
          <p className="intro-description">Analyze text for potential hate speech using a machine learning model.</p>
        </div>
        <div className="intro-aside" aria-label="Classification overview">
          <span className="aside-number">01</span>
          <p>One message at a time.<br />No accounts. No saved history.</p>
        </div>
      </section>

      <div className="analysis-layout">
        <div className="left-column">
          <TextInput
            text={text}
            onTextChange={(value) => { setText(value); setError(''); setResult(null) }}
            onSubmit={handleSubmit}
            onClear={handleClear}
            onExample={(example) => { setText(example); setError(''); setResult(null) }}
            isLoading={isLoading}
          />
          {error && <div className="error-banner" role="alert"><span className="error-mark">!</span><p>{error}</p></div>}
          {isLoading && <p className="loading-copy" role="status">Analyzing your text with the trained model...</p>}
        </div>
        <PredictionResult result={result} />
      </div>

      <footer className="page-footer">
        <p className="disclaimer">Predictions are machine learning estimates and may be incorrect. The result should not be treated as a definitive judgment about a person or message.</p>
        <span className="footer-tag">SAFETEXT AI <span>/</span> TEXT CLASSIFICATION</span>
      </footer>
    </main>
  )
}