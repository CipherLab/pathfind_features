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
      setRuns(r.sort((a, b) => a.created_at - b.created_at));
    } catch {}
  }, 1500);
  return (
    <Box flexDirection="column">
      <Text color="cyan">Runs</Text>
      {runs.map((r) => (
        <Text key={r.id}>
          {r.status.padEnd(8)} {r.id.slice(0, 8)} → {r.params?.run_name || ""}{" "}
          dir={r.run_dir}
        </Text>
      ))}
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
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor="green"
      padding={1}
    >
      <Text>Press Enter to submit a smoke test run. Esc to quit.</Text>
      {msg && <Text color="green">{msg}</Text>}
    </Box>
  );
};

const App = () => (
  <Box flexDirection="column">
    <Text color="yellow">Pathfind Experiments CLI</Text>
    <Text>API: {API}</Text>
    <Text>Hint: Press Enter to submit a smoke run. Press Esc to exit.</Text>
    <Box height={1} />
    <NewRun />
    <Box height={1} />
    <Runs />
  </Box>
);

render(<App />);
