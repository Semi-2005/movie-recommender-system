import { Star, Users, Hash } from 'lucide-react'
import './MovieCard.css'

const MAX_GENRES = 3

function StarRating({ rating }) {
  const fullStars = Math.floor(rating / 1) // rating is out of 5
  const normalizedRating = rating // already 0-5
  const filledWidth = `${(normalizedRating / 5) * 100}%`

  return (
    <div className="star-rating" title={`${rating.toFixed(2)} / 5`}>
      <div className="stars-track">
        {'★★★★★'.split('').map((_, i) => (
          <span key={i} className="star star--empty">★</span>
        ))}
        <div className="stars-fill" style={{ width: filledWidth }}>
          {'★★★★★'.split('').map((_, i) => (
            <span key={i} className="star star--full">★</span>
          ))}
        </div>
      </div>
      <span className="star-value">{rating.toFixed(2)}</span>
    </div>
  )
}

function ScoreBar({ score, label, color }) {
  return (
    <div className="score-bar-row">
      <span className="score-bar-label">{label}</span>
      <div className="score-bar-track">
        <div
          className="score-bar-fill"
          style={{
            width: `${(score ?? 0) * 100}%`,
            background: color,
          }}
        />
      </div>
      <span className="score-bar-value">{score !== null && score !== undefined ? score.toFixed(3) : '—'}</span>
    </div>
  )
}

export default function MovieCard({ movie, rank, mode = 'hybrid' }) {
  const {
    title,
    genres,
    rating,
    rating_count,
    hybrid_score,
    content_score,
    collaborative_score,
    final_score,
    similarity_score,
  } = movie

  // Normalize genres string (pipe or space separated)
  const genreList = genres
    ? genres.replace(/\|/g, ' ').split(' ').filter(Boolean).slice(0, MAX_GENRES)
    : []

  // Primary score depending on mode
  const primaryScore = hybrid_score ?? final_score ?? similarity_score ?? 0

  return (
    <article className="movie-card animate-fade-in">
      <div className="movie-card__rank">
        <Hash size={11} />
        <span>{rank}</span>
      </div>

      <div className="movie-card__content">
        <h3 className="movie-card__title">{title}</h3>

        <div className="movie-card__meta">
          {rating !== undefined && (
            <StarRating rating={rating} />
          )}
          {rating_count !== undefined && (
            <span className="movie-card__votes">
              <Users size={12} />
              {rating_count.toLocaleString()}
            </span>
          )}
        </div>

        {genreList.length > 0 && (
          <div className="movie-card__genres">
            {genreList.map(g => (
              <span key={g} className="genre-pill">{g}</span>
            ))}
          </div>
        )}

        <div className="movie-card__scores">
          {mode === 'hybrid' && (
            <>
              <ScoreBar
                score={primaryScore}
                label="Hybrid"
                color="linear-gradient(90deg, var(--primary), var(--primary-light))"
              />
              {content_score !== null && content_score !== undefined && (
                <ScoreBar
                  score={content_score}
                  label="Content"
                  color="linear-gradient(90deg, #10b981, #34d399)"
                />
              )}
              {collaborative_score !== null && collaborative_score !== undefined && (
                <ScoreBar
                  score={collaborative_score}
                  label="Collab"
                  color="linear-gradient(90deg, #f59e0b, #fbbf24)"
                />
              )}
            </>
          )}
          {mode === 'content' && (
            <ScoreBar
              score={final_score ?? primaryScore}
              label="Score"
              color="linear-gradient(90deg, #10b981, #34d399)"
            />
          )}
          {mode === 'collaborative' && (
            <ScoreBar
              score={similarity_score ?? primaryScore}
              label="Similarity"
              color="linear-gradient(90deg, #f59e0b, #fbbf24)"
            />
          )}
        </div>
      </div>
    </article>
  )
}
