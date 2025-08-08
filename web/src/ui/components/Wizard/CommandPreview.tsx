import * as React from 'react'

export default function CommandPreview({ cmd }: { cmd: string }){
  return (
    <div className="card mt8">
      <div className="bold">Command preview</div>
      <pre className="pre">{cmd}</pre>
    </div>
  )
}
