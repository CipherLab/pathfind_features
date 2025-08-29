import * as React from 'react'

type Props = {
  previousRuns: string[]
  value: string
  onChange: (v: string) => void
  label: string
}

export default function StageSelector({ previousRuns, value, onChange, label }: Props){
  return (
    <label className="flex flex-col gap-2">
      <span className="text-sm font-medium">{label}</span>
      <select
        className="rounded-md border border-slate-600 bg-slate-800 p-2 text-sm"
        title={label}
        value={value}
        onChange={e=>onChange(e.target.value)}
      >
        <option value="">(none)</option>
        {previousRuns.map(r=> <option key={r} value={r}>{r}</option>)}
      </select>
    </label>
  )
}
