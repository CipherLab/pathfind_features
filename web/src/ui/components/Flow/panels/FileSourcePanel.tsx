import React, { useRef, useState, useCallback } from 'react'
import { uploadFile } from '../../../lib/api'
import { PayloadType } from '../../Flow/types'

// Map of friendly options to PayloadType and picker mode
const TYPE_OPTIONS: { key: string; label: string; payload: PayloadType; picker: 'parquet' | 'features' | 'none' }[] = [
  { key: 'PARQUET', label: 'Numerai parquet', payload: 'PARQUET', picker: 'parquet' },
  { key: 'JSON_ARTIFACT', label: 'Features JSON (model/features)', payload: 'JSON_ARTIFACT', picker: 'features' },
  { key: 'RELATIONSHIPS', label: 'Pathfinding relationships (json)', payload: 'RELATIONSHIPS', picker: 'none' },
  { key: 'ADAPTIVE_TARGETS_PARQUET', label: 'Adaptive targets parquet', payload: 'ADAPTIVE_TARGETS_PARQUET', picker: 'parquet' },
  { key: 'TD_DISCOVERY_META', label: 'Target discovery meta (json)', payload: 'TD_DISCOVERY_META', picker: 'none' },
]

type Props = {
  cfg: any
  updateData: (patch: any) => void
}

export default function FileSourcePanel({ cfg, updateData }: Props) {
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [chosenName, setChosenName] = useState<string | undefined>(undefined)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const selected = TYPE_OPTIONS.find(o => o.payload === cfg.payloadType) || TYPE_OPTIONS[0]

  const chooseType = (optKey: string) => {
    const opt = TYPE_OPTIONS.find(o => o.key === optKey) || TYPE_OPTIONS[0]
    updateData({ payloadType: opt.payload })
  }

  const doUpload = useCallback(async (file: File) => {
    setError(null)
    setUploading(true)
    try {
      // Namespace by payload type for tidiness
      const subdir = (cfg.payloadType || 'PARQUET').toString().toLowerCase()
      const { path } = await uploadFile(file, { subdir })
      updateData({ inputPath: path })
    } catch (e: any) {
      setError(e?.message || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }, [cfg.payloadType, updateData])

  const onPickFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) {
      setChosenName(f.name)
      doUpload(f)
    }
  }, [doUpload])

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    const f = e.dataTransfer.files?.[0]
    if (f) {
      setChosenName(f.name)
      doUpload(f)
    }
  }, [doUpload])

  const onDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
  }, [])

  const isJsonType = selected.payload === 'JSON_ARTIFACT' || selected.payload === 'TD_DISCOVERY_META' || selected.payload === 'RELATIONSHIPS'
  const accept = isJsonType ? '.json' : '.parquet'

  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className="text-sm font-medium mb-2">File type</div>
        <div className="grid grid-cols-1 gap-2 justify-items-start align-left">
          {TYPE_OPTIONS.map(opt => (
            <label key={opt.key} className="flex items-center gap-2 text-sm justify-start w-full">
              <input
                type="radio"
                name="fileType"
                className="accent-indigo-400"
                checked={(cfg.payloadType || 'PARQUET') === opt.payload}
                onChange={() => chooseType(opt.key)}
              />
              <span>{opt.label}</span>
            </label>
          ))}
        </div>
        <div className="text-xs text-slate-400 mt-1">Select what kind of artifact this file represents. Connections will be validated against this.</div>
      </div>

      <div>
        <div className="text-sm font-medium mb-2">{isJsonType ? 'Select or drop a file' : 'Set file path'}</div>
        {isJsonType ? (
          <>
            <div
              className={`border border-dashed rounded-md p-4 text-sm cursor-pointer ${uploading ? 'opacity-70' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDrop={onDrop}
              onDragOver={onDragOver}
              title={cfg.inputPath}
            >
              <div className="flex items-center justify-between">
                <div className="truncate">
                  {cfg.inputPath ? (
                    <span className="text-slate-200" title={cfg.inputPath}>{cfg.inputPath}</span>
                  ) : chosenName ? (
                    <span className="text-slate-200" title={chosenName}>{chosenName}</span>
                  ) : (
                    <span className="text-slate-400">Drop a JSON file here or click to choose…</span>
                  )}
                </div>
                <div className="text-xs text-slate-400">{uploading ? 'Uploading…' : 'Browse'}</div>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept={accept}
                onChange={onPickFile}
                aria-label="Choose JSON file to upload"
                className="hidden"
              />
            </div>
            <div className="text-xs text-slate-400 mt-1">
              JSON file (features.json, relationships.json, or discovery meta).
            </div>
            {error && <div className="text-xs text-red-500 mt-2">{error}</div>}
          </>
        ) : (
          <>
            <input
              className="input text-sm"
              placeholder={selected.payload === 'PARQUET' ? 'e.g. v5.0/train.parquet' : 'e.g. pipeline_runs/exp/01_adaptive_targets.parquet'}
              value={cfg.inputPath || ''}
              onChange={e => updateData({ inputPath: e.target.value })}
            />
            <div className="flex gap-2 mt-2">
              {selected.payload === 'PARQUET' && (
                <>
                  <button className="btn btn-xs" onClick={() => updateData({ inputPath: 'v5.0/train.parquet' })}>Use v5.0/train.parquet</button>
                  <button className="btn btn-xs" onClick={() => updateData({ inputPath: 'v5.0/validation.parquet' })}>Use v5.0/validation.parquet</button>
                  <button className="btn btn-xs" onClick={() => updateData({ inputPath: 'v5.0/live.parquet' })}>Use v5.0/live.parquet</button>
                </>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
