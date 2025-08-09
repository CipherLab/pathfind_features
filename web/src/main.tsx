import * as React from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import App from './ui/App'
import './ui/app.css'
import DashboardPage from './ui/pages/DashboardPage'
import WizardPage from './ui/pages/WizardPage'
import RunDetailPage from './ui/pages/RunDetailPage'
import BuilderPage from './ui/pages/BuilderPage'

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: 'wizard', element: <WizardPage /> },
  { path: 'builder', element: <BuilderPage /> },
      { path: 'runs/:name', element: <RunDetailPage /> },
    ],
  },
])

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
)
