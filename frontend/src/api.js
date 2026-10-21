function apiUrl(path) {
  const configuredUrl = import.meta.env.VITE_API_BASE_URL?.trim()
  if (!configuredUrl && import.meta.env.PROD) {
    throw new Error('Production API URL is not configured. Set VITE_API_BASE_URL in the Netlify site environment.')
  }
  const baseUrl = (configuredUrl || 'http://127.0.0.1:8000').replace(/\/+$/, '')
  return `${baseUrl}${path}`
}

async function readResponse(response) {
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = typeof payload.detail === 'string' ? payload.detail : 'The request could not be completed.'
    throw new Error(detail)
  }
  return payload
}

export async function getHealth() {
  const response = await fetch(apiUrl('/health'))
  return readResponse(response)
}

export async function predictText(text) {
  const response = await fetch(apiUrl('/predict'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  return readResponse(response)
}