import * as React from 'react'

export default function CommandPreview({ cmd }: { cmd: string }){
  return (
    <div className="card mt8">
  <div className="row-between">
        <div className="bold">Command preview</div>
        <button className="btn" onClick={()=> navigator.clipboard?.writeText(cmd)}>Copy</button>
      </div>
  <pre className="pre pre-panel">{cmd}</pre>
    </div>
  )
}
