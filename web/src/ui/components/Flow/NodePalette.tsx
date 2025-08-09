import React from 'react'
import { NodeKind } from './types'

type Props = {
  onAdd: (kind: NodeKind) => void
}

export default function NodePalette({ onAdd }: Props) {
  const items: { kind: NodeKind; label: string; icon: string }[] = [
    { kind: 'data-source', label: 'Data', icon: '📁' },
    { kind: 'target-discovery', label: 'Targets', icon: '🎯' },
    { kind: 'pathfinding', label: 'Pathfind', icon: '🔍' },
    { kind: 'feature-engineering', label: 'Features', icon: '⚗️' },
    { kind: 'output', label: 'Output', icon: '📊' },
  ]
  return (
    <div className="flex flex-col gap-2 p-2">
      <div className="text-xs font-semibold text-slate-300">Palette</div>
      {items.map(it => (
        <div
          key={it.kind}
          className="btn cursor-grab active:cursor-grabbing"
          draggable
          onDragStart={e => {
            e.dataTransfer.setData('application/reactflow', it.kind)
            e.dataTransfer.effectAllowed = 'move'
          }}
          onDoubleClick={() => onAdd(it.kind)}
          role="button"
          tabIndex={0}
          onKeyDown={e => {
            if (e.key === 'Enter') onAdd(it.kind)
          }}
        >
          <span className="w-6 text-center">{it.icon}</span> {it.label}
        </div>
      ))}
    </div>
  )
}
