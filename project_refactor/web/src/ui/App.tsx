import * as React from 'react'
import { Link, Outlet, useLocation } from 'react-router-dom'

const API = (import.meta as any).env?.VITE_API_URL || 'http://127.0.0.1:8000'

export default function App(){
  const location = useLocation()
  const isBuilder = location.pathname.startsWith('/builder')
  return (
    <div className={`container ${isBuilder ? 'fullwidth' : ''}`}>
      <header className="row headerRow">
        <h2 className="noMargin">Pathfind Orchestrator</h2>
        <nav className="row gap8">
          <Link to="/">Dashboard</Link>
          <Link to="/wizard">New Run</Link>
          <Link to="/builder">Visual Builder</Link>
        </nav>
      </header>
      <div className="mt8 muted">API: {API}</div>
  <main className="mt8 w-full">
        <Outlet />
      </main>
    </div>
  )
}
