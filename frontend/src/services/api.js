/**
 * CineMatch API Service Layer
 * All backend communication is centralized here.
 * Vite proxy rewrites /api/* → http://localhost:8000/*
 */

const BASE = '/api'

const handleResponse = async (res) => {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

/**
 * Search movie titles (autocomplete)
 * @param {string} q - search query
 * @param {number} limit - max results
 */
export const searchMovies = async (q, limit = 10) => {
  const params = new URLSearchParams({ q, limit })
  const res = await fetch(`${BASE}/search?${params}`)
  return handleResponse(res)
}

/**
 * Hybrid recommendations (primary)
 * @param {string} movie - movie title
 * @param {number} top_n - number of results
 * @param {number|null} alpha_override - optional alpha [0,1]
 */
export const getHybridRecommendations = async (movie, top_n = 12, alpha_override = null) => {
  const params = new URLSearchParams({ movie, top_n })
  if (alpha_override !== null) params.set('alpha_override', alpha_override)
  const res = await fetch(`${BASE}/recommend/hybrid?${params}`)
  return handleResponse(res)
}

/**
 * Content-based recommendations
 * @param {string} movie - movie title
 * @param {number} top_n - number of results
 */
export const getContentRecommendations = async (movie, top_n = 12) => {
  const params = new URLSearchParams({ movie, top_n })
  const res = await fetch(`${BASE}/recommend?${params}`)
  return handleResponse(res)
}

/**
 * Collaborative filtering recommendations
 * @param {string} movie - movie title
 * @param {number} top_n - number of results
 */
export const getCollaborativeRecommendations = async (movie, top_n = 12) => {
  const params = new URLSearchParams({ movie, top_n })
  const res = await fetch(`${BASE}/recommend/collaborative?${params}`)
  return handleResponse(res)
}

/**
 * Model statistics
 */
export const getModelStats = async () => {
  const res = await fetch(`${BASE}/stats`)
  return handleResponse(res)
}

/**
 * Health check
 */
export const getHealth = async () => {
  const res = await fetch(`${BASE}/health`)
  return handleResponse(res)
}
