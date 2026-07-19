import './StrategyBadge.css'

const STRATEGY_CONFIG = {
  hybrid_full: {
    label: '⚡ Hybrid Full',
    className: 'badge--success',
    description: 'Both AI models combined',
  },
  hybrid_partial: {
    label: '🔀 Hybrid Partial',
    className: 'badge--warning',
    description: 'Limited collaborative signal',
  },
  content_based_only: {
    label: '🎭 Content Based',
    className: 'badge--info',
    description: 'Genre similarity only',
  },
}

export default function StrategyBadge({ strategy, alpha, fusionTimeMs, cbCandidates, cfCandidates }) {
  // strategy may have a suffix like "hybrid_full(alpha_override=0.8)"
  const baseStrategy = Object.keys(STRATEGY_CONFIG).find(k => strategy?.startsWith(k)) || 'content_based_only'
  const config = STRATEGY_CONFIG[baseStrategy]

  return (
    <div className="strategy-wrapper">
      <span className={`strategy-badge ${config.className}`}>
        {config.label}
      </span>
      <div className="strategy-meta">
        {alpha !== undefined && (
          <span className="meta-chip" title="Content-based weight (alpha)">
            α = {alpha.toFixed(2)}
          </span>
        )}
        {fusionTimeMs !== undefined && (
          <span className="meta-chip" title="Fusion computation time">
            {Math.round(fusionTimeMs)}ms
          </span>
        )}
        {cbCandidates !== undefined && (
          <span className="meta-chip" title="Content-based candidates">
            CB: {cbCandidates}
          </span>
        )}
        {cfCandidates !== undefined && (
          <span className="meta-chip" title="Collaborative candidates">
            CF: {cfCandidates}
          </span>
        )}
      </div>
    </div>
  )
}
