import MovieCard from '../MovieCard/MovieCard'
import SkeletonCard from '../SkeletonCard/SkeletonCard'
import { AlertCircle, SearchX } from 'lucide-react'
import './RecommendationGrid.css'

export default function RecommendationGrid({ recommendations, isLoading, error, mode, suggestions }) {
  if (isLoading) {
    return (
      <div className="rec-grid">
        {Array.from({ length: 8 }).map((_, i) => (
          <SkeletonCard key={i} />
        ))}
      </div>
    )
  }

  if (error) {
    return (
      <div className="rec-empty">
        <AlertCircle size={40} className="rec-empty__icon rec-empty__icon--error" />
        <h3>Something went wrong</h3>
        <p>{error}</p>
      </div>
    )
  }

  if (!recommendations || recommendations.length === 0) {
    return (
      <div className="rec-empty">
        <SearchX size={40} className="rec-empty__icon" />
        <h3>No results found</h3>
        <p>Try a different title or check your spelling.</p>
        {suggestions && suggestions.length > 0 && (
          <div className="rec-empty__suggestions">
            <p className="rec-empty__suggestions-label">Did you mean:</p>
            {suggestions.map(s => (
              <span key={s} className="rec-empty__suggestion-chip">{s}</span>
            ))}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="rec-grid">
      {recommendations.map((movie, i) => (
        <MovieCard
          key={movie.movie_id ?? movie.movieId ?? i}
          movie={movie}
          rank={i + 1}
          mode={mode}
        />
      ))}
    </div>
  )
}
