export const API_BASE =
  (import.meta as any).env?.VITE_API_URL || "http://127.0.0.1:8000";

export async function jget<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json() as Promise<T>;
}

export async function jpost(path: string, body: any) {
  const r = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export async function uploadFile(
  file: File,
  opts?: { subdir?: string; filename?: string }
): Promise<{ path: string }> {
  const form = new FormData();
  form.append("file", file);
  const params = new URLSearchParams();
  if (opts?.subdir) params.set("subdir", opts.subdir);
  if (opts?.filename) params.set("filename", opts.filename);
  const url = `${API_BASE}/files/upload${
    params.toString() ? `?${params.toString()}` : ""
  }`;
  const r = await fetch(url, { method: "POST", body: form });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  const data = await r.json();
  return { path: data.path as string };
}
