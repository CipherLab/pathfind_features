import * as React from 'react'
import { Link, useParams } from 'react-router-dom'
import RunTabs from '../components/RunTabs'

export default function RunDetailPage(){
  const { name = '' } = useParams()
  return (
    <div>
      <Link to="/">← Back</Link>
      <h2 className="mt8">{name}</h2>
      {name && <RunTabs name={name} />}
    </div>
  )
}
