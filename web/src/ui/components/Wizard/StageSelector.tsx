import * as React from 'react'

type Props = {
  previousRuns: string[]
  value: string
  onChange: (v: string) => void
  label: string
}

export default function StageSelector({ previousRuns, value, onChange, label }: Props){
  return (
    <label>
      {label}
      <select title={label} value={value} onChange={e=>onChange(e.target.value)}>
        <option value="">(none)</option>
        {previousRuns.map(r=> <option key={r} value={r}>{r}</option>)}
      </select>
    </label>
  )
}
