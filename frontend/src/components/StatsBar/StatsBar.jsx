import { useEffect, useState } from 'react'
import { Film, Users, Star, Cpu } from 'lucide-react'
import { getModelStats } from '../../services/api'
import './StatsBar.css'

const formatNumber = (n) => {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`
  return n?.toString() ?? '—'
}

export default function StatsBar() {
  const [stats, setStats] = useState(null)

  useEffect(() => {
    getModelStats()
      .then(setStats)
      .catch(() => {}) // silent — stats bar is non-critical
  }, [])

  const movieCount = stats?.content_based?.movie_count
  const featureCount = stats?.content_based?.feature_count
  const cfMovies = stats?.collaborative?.movie_count

  return (
    <div className="stats-bar">
      <div className="stats-bar__inner container">
        <StatItem icon={<Film size={14} />} label="Films indexed" value={movieCount ? formatNumber(movieCount) : '…'} />
        <div className="stats-bar__divider" />
        <StatItem icon={<Star size={14} />} label="Rating interactions" value="20M+" />
        <div className="stats-bar__divider" />
        <StatItem icon={<Cpu size={14} />} label="TF-IDF features" value={featureCount ? formatNumber(featureCount) : '…'} />
        <div className="stats-bar__divider" />
        <StatItem icon={<Users size={14} />} label="CF coverage" value={cfMovies ? formatNumber(cfMovies) : '…'} />
        <div className="stats-bar__divider" />
        <StatItem icon={null} label="Models active" value="3 AI Models" highlight />
      </div>
    </div>
  )
}

function StatItem({ icon, label, value, highlight }) {
  return (
    <div className={`stat-item ${highlight ? 'stat-item--highlight' : ''}`}>
      {icon && <span className="stat-item__icon">{icon}</span>}
      <span className="stat-item__value">{value}</span>
      <span className="stat-item__label">{label}</span>
    </div>
  )
}
