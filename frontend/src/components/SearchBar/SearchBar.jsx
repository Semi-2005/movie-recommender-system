import { useState, useEffect, useRef, useCallback } from 'react'
import { Search, X, Loader2 } from 'lucide-react'
import { searchMovies } from '../../services/api'
import './SearchBar.css'

const DEBOUNCE_MS = 300

export default function SearchBar({ onSearch, initialValue = '', autoFocus = false }) {
  const [query, setQuery] = useState(initialValue)
  const [suggestions, setSuggestions] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [isOpen, setIsOpen] = useState(false)
  const [activeIndex, setActiveIndex] = useState(-1)
  const inputRef = useRef(null)
  const dropdownRef = useRef(null)
  const timerRef = useRef(null)

  useEffect(() => {
    if (autoFocus && inputRef.current) {
      inputRef.current.focus()
    }
  }, [autoFocus])

  // Click outside to close
  useEffect(() => {
    const handler = (e) => {
      if (
        dropdownRef.current && !dropdownRef.current.contains(e.target) &&
        inputRef.current && !inputRef.current.contains(e.target)
      ) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const fetchSuggestions = useCallback(async (q) => {
    if (q.trim().length < 2) {
      setSuggestions([])
      setIsOpen(false)
      return
    }
    setIsLoading(true)
    try {
      const data = await searchMovies(q, 8)
      setSuggestions(data.results || [])
      setIsOpen(true)
      setActiveIndex(-1)
    } catch {
      setSuggestions([])
    } finally {
      setIsLoading(false)
    }
  }, [])

  const handleChange = (e) => {
    const val = e.target.value
    setQuery(val)
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => fetchSuggestions(val), DEBOUNCE_MS)
  }

  const handleSelect = (title) => {
    setQuery(title)
    setIsOpen(false)
    setSuggestions([])
    onSearch(title)
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!query.trim()) return
    if (activeIndex >= 0 && suggestions[activeIndex]) {
      handleSelect(suggestions[activeIndex])
    } else {
      handleSelect(query.trim())
    }
  }

  const handleKeyDown = (e) => {
    if (!isOpen || suggestions.length === 0) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveIndex(i => Math.min(i + 1, suggestions.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIndex(i => Math.max(i - 1, -1))
    } else if (e.key === 'Escape') {
      setIsOpen(false)
    }
  }

  const clearQuery = () => {
    setQuery('')
    setSuggestions([])
    setIsOpen(false)
    inputRef.current?.focus()
  }

  return (
    <div className="searchbar-wrapper">
      <form className="searchbar" onSubmit={handleSubmit} role="search">
        <span className="searchbar__icon">
          {isLoading
            ? <Loader2 size={18} className="searchbar__spinner" />
            : <Search size={18} />
          }
        </span>
        <input
          ref={inputRef}
          id="movie-search-input"
          type="search"
          className="searchbar__input"
          placeholder="Search a movie title…"
          value={query}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onFocus={() => suggestions.length > 0 && setIsOpen(true)}
          autoComplete="off"
          spellCheck="false"
          aria-label="Search movies"
          aria-autocomplete="list"
          aria-expanded={isOpen}
          aria-controls="search-suggestions"
        />
        {query && (
          <button
            type="button"
            className="searchbar__clear"
            onClick={clearQuery}
            aria-label="Clear search"
          >
            <X size={15} />
          </button>
        )}
        <button type="submit" className="searchbar__btn" id="search-submit-btn">
          Search
        </button>
      </form>

      {isOpen && suggestions.length > 0 && (
        <ul
          ref={dropdownRef}
          id="search-suggestions"
          className="searchbar__dropdown"
          role="listbox"
          aria-label="Movie suggestions"
        >
          {suggestions.map((title, i) => (
            <li
              key={title}
              role="option"
              aria-selected={i === activeIndex}
              className={`searchbar__suggestion ${i === activeIndex ? 'is-active' : ''}`}
              onMouseDown={() => handleSelect(title)}
              onMouseEnter={() => setActiveIndex(i)}
            >
              <Search size={13} className="suggestion-icon" />
              <span>{title}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
