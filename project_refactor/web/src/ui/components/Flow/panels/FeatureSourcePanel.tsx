import React, { useState } from 'react'
import GlobalPickerModal from '../../Wizard/GlobalPickerModal'

type Props = {
  cfg: any
  updateData: (patch: any) => void
}

export default function FeatureSourcePanel({ cfg, updateData }: Props) {
  const [open, setOpen] = useState<null | 'features'>(null)
  const FEATURES_JSON_PLACEHOLDER = 'Select a features.json file';
  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className="text-sm font-medium mb-2">Features JSON</div>
        <button
          className="btn w-full justify-start"
          onClick={() => setOpen('features')}
          title={cfg.featuresJson}
        >
          {cfg.featuresJson || FEATURES_JSON_PLACEHOLDER}
        </button>
        <div className="text-xs text-slate-400 mt-1">
          Feature definition file with feature_sets.medium.
        </div>
      </div>
      {open && (
        <GlobalPickerModal
          mode="features"
          onSelect={v => {
            updateData({ featuresJson: v })
            setOpen(null)
          }}
          onClose={() => setOpen(null)}
        />
      )}
    </div>
  )
}
