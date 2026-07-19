import './TabBar.css'

const TABS = [
  { id: 'hybrid',        label: '🤖 Hybrid',        description: 'Best of both models' },
  { id: 'content',       label: '🎭 Content',        description: 'Genre similarity' },
  { id: 'collaborative', label: '👥 Collaborative',  description: 'User patterns' },
]

export default function TabBar({ activeTab, onChange }) {
  return (
    <div className="tab-bar" role="tablist" aria-label="Recommendation model">
      {TABS.map(tab => (
        <button
          key={tab.id}
          role="tab"
          id={`tab-${tab.id}`}
          aria-selected={activeTab === tab.id}
          aria-controls={`panel-${tab.id}`}
          className={`tab-bar__tab ${activeTab === tab.id ? 'is-active' : ''}`}
          onClick={() => onChange(tab.id)}
        >
          <span className="tab-bar__label">{tab.label}</span>
          <span className="tab-bar__desc">{tab.description}</span>
        </button>
      ))}
    </div>
  )
}
