#!/usr/bin/env node
import React, { useEffect, useState } from "react";
import { render, Text, Box, useApp, useInput } from "ink";
import got from "got";

const API = process.env.PATHFIND_API || "http://127.0.0.1:8000";

function useInterval(cb, ms) {
  useEffect(() => {
    const id = setInterval(cb, ms);
    return () => clearInterval(id);
  }, [cb, ms]);
}

const Runs = () => {
  const [runs, setRuns] = useState([]);
  useInterval(async () => {
    try {
      const r = await got(`${API}/runs`, { timeout: { request: 2000 } }).json();
      setRuns((r || []).sort((a, b) => a.created_at - b.created_at));
    } catch (e) {
      // show a brief error only once
    }
  }, 1500);
  return (
    <Box flexDirection="column">
      <Text color="cyan">Runs</Text>
      {runs.map((r) => (
        <Text key={r.id}>
          {String(r.status || "").padEnd(8)} {String(r.id || "").slice(0, 8)} → {r.params?.run_name || ""} dir={r.run_dir}
        </Text>
      ))}
    </Box>
  );
};

const History = () => {
  const [items, setItems] = useState([]);
  const [sel, setSel] = useState(0);
  const [logs, setLogs] = useState("");
  const [summary, setSummary] = useState("");
  useEffect(()=>{
    (async ()=>{
      try {
        const r = await got(`${API}/runs/list-fs`).json();
        setItems(r.reverse());
      } catch {}
    })();
  },[]);
  useInput(async (input, key)=>{
    if (key.upArrow) setSel(s=> Math.max(0, s-1));
    if (key.downArrow) setSel(s=> Math.min(items.length-1, s+1));
    if (input === 'l' && items[sel]){
      const name = items[sel].name;
      try {
        const r = await got(`${API}/runs/list-fs/${name}/logs`).json();
        setLogs(r.content || "");
      } catch {}
    }
    if (input === 's' && items[sel]){
      const name = items[sel].name;
      try {
        const r = await got(`${API}/runs/list-fs/${name}/summary`).text();
        setSummary(r);
      } catch {}
    }
  });
  const item = items[sel];
  return (
    <Box flexDirection="column">
      <Text color="cyan">History (↑/↓ to select, l to view logs, s to view summary)</Text>
      {items.map((it, i)=> (
        <Text key={it.path} color={i===sel? 'yellow':'white'}>
          {i===sel? '➤':' '} {it.status?.padEnd(8)} {it.name}
        </Text>
      ))}
      {logs && (<>
        <Text color="magenta">─ logs tail ─</Text>
        <Text>{logs}</Text>
      </>)}
      {summary && (<>
        <Text color="magenta">─ summary json ─</Text>
        <Text>{summary}</Text>
      </>)}
    </Box>
  );
};

const NewRun = () => {
  const [msg, setMsg] = useState("");
  const [pending, setPending] = useState(false);
  const { exit } = useApp();
  useInput(async (input, key) => {
    if (key.escape) exit();
    if (key.return && !pending) {
      setPending(true);
      try {
        await got
          .post(`${API}/runs`, {
            json: {
              input_data: "v5.0/train.parquet",
              features_json: "v5.0/features.json",
              run_name: "api_cli",
              pretty: true,
              smoke_mode: true,
              smoke_max_eras: 60,
              smoke_row_limit: 150000,
              smoke_feature_limit: 300,
              max_new_features: 8,
              seed: 42,
            },
          })
          .json();
        setMsg("Run submitted");
      } catch (e) {
        setMsg("Failed to submit run");
      }
      setPending(false);
    }
  });
  return (
    <Box flexDirection="column" borderStyle="round" borderColor="green" padding={1}>
      <Text>Press Enter to submit a smoke test run. Esc to quit.</Text>
      {msg && <Text color="green">{msg}</Text>}
    </Box>
  );
};

const App = () => (
  <Tabs />
);

const Tabs = () => {
  const [tab,setTab]=useState(0);
  const { exit } = useApp();
  useInput((input, key)=>{
    if (key.escape) exit();
    if (input==='1') setTab(0);
    if (input==='2') setTab(1);
    if (input==='3') setTab(2);
  });
  return (
    <Box flexDirection="column">
      <Text color="yellow">Pathfind Experiments CLI</Text>
      <Text>API: {API}</Text>
      <Text>Tabs: [1] New Run  [2] Active (API)  [3] History (fs)</Text>
      <Box height={1} />
      {tab===0 && <NewRun/>}
      {tab===1 && <Runs/>}
      {tab===2 && <History/>}
    </Box>
  );
};

render(<App />);
