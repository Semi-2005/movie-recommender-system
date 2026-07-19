import './SkeletonCard.css'

export default function SkeletonCard() {
  return (
    <div className="skeleton-card" aria-hidden="true">
      <div className="skeleton-rank" />
      <div className="skeleton-body">
        <div className="skeleton-title" />
        <div className="skeleton-subtitle" />
        <div className="skeleton-genres">
          <div className="skeleton-pill" />
          <div className="skeleton-pill" />
          <div className="skeleton-pill" />
        </div>
        <div className="skeleton-score" />
      </div>
    </div>
  )
}
