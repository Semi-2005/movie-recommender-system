import { useNavigate } from 'react-router-dom'
import SearchBar from '../../components/SearchBar/SearchBar'
import StatsBar from '../../components/StatsBar/StatsBar'
import './HomePage.css'

const TRENDING = [
  'Toy Story (1995)',
  'Inception (2010)',
  'The Matrix (1999)',
  'Interstellar (2014)',
  'Pulp Fiction (1994)',
  'The Dark Knight (2008)',
]

export default function HomePage() {
  const navigate = useNavigate()

  const handleSearch = (title) => {
    navigate(`/recommend/${encodeURIComponent(title)}`)
  }

  return (
    <div className="home-page">
      {/* ── Hero ──────────────────────────────────────────── */}
      <section className="hero">
        <div className="hero__inner container">
          <div className="hero__badge animate-fade-in">
            <span className="hero__badge-dot" />
            3 AI Models Active
          </div>

          <h1 className="hero__title animate-fade-in" style={{ animationDelay: '0.05s' }}>
            Your next favorite film<br />
            <span className="text-gradient">is one search away.</span>
          </h1>

          <p className="hero__subtitle animate-fade-in" style={{ animationDelay: '0.1s' }}>
            CineMatch fuses content-based and collaborative AI to surface
            recommendations that actually match your taste.
          </p>

          <div className="hero__search animate-fade-in" style={{ animationDelay: '0.15s' }}>
            <SearchBar onSearch={handleSearch} autoFocus />
          </div>

          {/* Trending chips */}
          <div className="hero__trending animate-fade-in" style={{ animationDelay: '0.2s' }}>
            <span className="trending-label">Trending:</span>
            {TRENDING.map(title => (
              <button
                key={title}
                className="trending-chip"
                onClick={() => handleSearch(title)}
              >
                {title}
              </button>
            ))}
          </div>
        </div>

        {/* Decorative orbs */}
        <div className="hero__orb hero__orb--1" aria-hidden="true" />
        <div className="hero__orb hero__orb--2" aria-hidden="true" />
      </section>

      {/* ── Stats Bar ────────────────────────────────────── */}
      <StatsBar />

      {/* ── Feature Section ──────────────────────────────── */}
      <section className="features container">
        <div className="feature-card animate-fade-in">
          <div className="feature-card__icon">🎭</div>
          <h3>Content-Based</h3>
          <p>TF-IDF genre analysis finds movies with similar themes and style.</p>
        </div>
        <div className="feature-card animate-fade-in" style={{ animationDelay: '0.1s' }}>
          <div className="feature-card__icon">👥</div>
          <h3>Collaborative</h3>
          <p>Item-based filtering from 20M user ratings surfaces hidden gems.</p>
        </div>
        <div className="feature-card animate-fade-in" style={{ animationDelay: '0.2s' }}>
          <div className="feature-card__icon">⚡</div>
          <h3>Hybrid AI</h3>
          <p>Adaptive ensemble blends both models for superior accuracy.</p>
        </div>
      </section>
    </div>
  )
}
