import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Clapperboard } from 'lucide-react'
import SearchBar from '../../components/SearchBar/SearchBar'
import TabBar from '../../components/TabBar/TabBar'
import RecommendationGrid from '../../components/RecommendationGrid/RecommendationGrid'
import StrategyBadge from '../../components/StrategyBadge/StrategyBadge'
import {
  getHybridRecommendations,
  getContentRecommendations,
  getCollaborativeRecommendations,
} from '../../services/api'
import './ResultsPage.css'

const TOP_N = 12

export default function ResultsPage() {
  const { movie } = useParams()
  const navigate = useNavigate()
  const decodedTitle = decodeURIComponent(movie)

  const [activeTab, setActiveTab] = useState('hybrid')
  const [data, setData] = useState({ hybrid: null, content: null, collaborative: null })
  const [loading, setLoading] = useState({ hybrid: false, content: false, collaborative: false })
  const [errors, setErrors] = useState({ hybrid: null, content: null, collaborative: null })

  const fetchTab = useCallback(async (tab) => {
    if (data[tab] !== null) return // already fetched

    setLoading(prev => ({ ...prev, [tab]: true }))
    setErrors(prev => ({ ...prev, [tab]: null }))

    try {
      let result
      if (tab === 'hybrid') result = await getHybridRecommendations(decodedTitle, TOP_N)
      else if (tab === 'content') result = await getContentRecommendations(decodedTitle, TOP_N)
      else result = await getCollaborativeRecommendations(decodedTitle, TOP_N)

      setData(prev => ({ ...prev, [tab]: result }))
    } catch (e) {
      setErrors(prev => ({ ...prev, [tab]: e.message }))
    } finally {
      setLoading(prev => ({ ...prev, [tab]: false }))
    }
  }, [decodedTitle, data])

  // Fetch hybrid on mount
  useEffect(() => {
    fetchTab('hybrid')
  }, [decodedTitle]) // eslint-disable-line

  // Fetch tab on selection
  const handleTabChange = (tab) => {
    setActiveTab(tab)
    fetchTab(tab)
  }

  const handleSearch = (title) => {
    navigate(`/recommend/${encodeURIComponent(title)}`)
    // Reset all data for new search
    setData({ hybrid: null, content: null, collaborative: null })
    setErrors({ hybrid: null, content: null, collaborative: null })
  }

  // Derive display values
  const hybridMeta = data.hybrid
  const activeData = data[activeTab]
  const isLoading = loading[activeTab]
  const error = errors[activeTab]

  const recommendations = (() => {
    if (!activeData) return null
    if (activeTab === 'hybrid') return activeData.recommendations
    if (activeTab === 'content') return activeData.recommendations
    if (activeTab === 'collaborative') return activeData?.recommendations
    return null
  })()

  const suggestions = activeData?.suggestions

  return (
    <div className="results-page">
      {/* ── Header ────────────────────────────────────────── */}
      <header className="results-header">
        <div className="results-header__inner container">
          <div className="results-header__nav">
            <button
              className="back-btn"
              onClick={() => navigate('/')}
              aria-label="Back to home"
              id="back-to-home-btn"
            >
              <ArrowLeft size={16} />
              <span>Back</span>
            </button>

            <div className="results-header__logo">
              <Clapperboard size={20} />
              <span>CineMatch</span>
            </div>
          </div>

          <div className="results-header__search">
            <SearchBar onSearch={handleSearch} initialValue={decodedTitle} />
          </div>
        </div>
      </header>

      {/* ── Main ──────────────────────────────────────────── */}
      <main className="results-main container">
        {/* Title + strategy info */}
        <div className="results-title-section">
          <h1 className="results-title">
            Results for{' '}
            <span className="text-gradient">"{decodedTitle}"</span>
          </h1>

          {hybridMeta && activeTab === 'hybrid' && (
            <StrategyBadge
              strategy={hybridMeta.strategy}
              alpha={hybridMeta.alpha}
              fusionTimeMs={hybridMeta.fusion_time_ms}
              cbCandidates={hybridMeta.cb_candidates}
              cfCandidates={hybridMeta.cf_candidates}
            />
          )}

          {activeData && !isLoading && (
            <p className="results-count">
              {recommendations?.length ?? 0} recommendations
            </p>
          )}
        </div>

        {/* Tab bar */}
        <TabBar activeTab={activeTab} onChange={handleTabChange} />

        {/* Grid */}
        <div
          id={`panel-${activeTab}`}
          role="tabpanel"
          aria-labelledby={`tab-${activeTab}`}
          className="results-panel"
        >
          <RecommendationGrid
            recommendations={recommendations}
            isLoading={isLoading}
            error={error}
            mode={activeTab}
            suggestions={suggestions}
          />
        </div>
      </main>
    </div>
  )
}
