'use client';

import { useState, useEffect, useRef } from 'react';

const API = 'http://localhost:8080';

export default function Home() {
  const [repos, setRepos] = useState([]);
  const [selected, setSelected] = useState('');
  const [tab, setTab] = useState('chat');
  const [initUrl, setInitUrl] = useState('');
  const [initStatus, setInitStatus] = useState('');

  useEffect(() => {
    fetchRepos();
    const interval = setInterval(fetchRepos, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchRepos = async () => {
    try {
      const res = await fetch(`${API}/repos`);
      const data = await res.json();
      setRepos(data.repos || []);
      setSelected(prev => {
        if (prev && data.repos?.includes(prev)) return prev;
        return data.repos?.[0] || '';
      });
    } catch {}
  };

  const handleInit = async () => {
    if (!initUrl.trim()) return;
    setInitStatus('Cloning & indexing... (this takes ~30s)');
    try {
      const res = await fetch(`${API}/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_url: initUrl }),
      });
      if (res.ok) {
        setInitStatus('Indexing in background... The repo will appear in the dropdown when ready.');
        const poll = setInterval(async () => {
          try {
            const r = await fetch(`${API}/repos`);
            const d = await r.json();
            const repos = d.repos || [];
            const slug = initUrl.split('/').slice(-2).join('-').replace('.git','');
            if (repos.includes(slug)) {
              clearInterval(poll);
              setInitStatus('Done! Select the repo above to chat.');
              setInitUrl('');
              fetchRepos();
            }
          } catch {}
        }, 5000);
        setTimeout(() => clearInterval(poll), 600000);
      }
    } catch {
      setInitStatus('Backend not reachable. Use CLI: repomate init ' + initUrl);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-[var(--border)] px-6 py-3 flex items-center gap-4">
        <h1 className="text-lg font-bold flex items-center gap-2">
          <span className="text-[var(--accent)]">repomate</span>
          <span className="text-sm font-normal text-[var(--muted)]">Code If You Can</span>
        </h1>
      </header>

      {/* Repo bar */}
      <div className="flex items-center gap-3 px-6 py-3 border-b border-[var(--border)]">
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          className="bg-[var(--surface)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm flex-1 max-w-xs"
        >
          {repos.length === 0 && <option value="">No repos yet</option>}
          {repos.map((r) => <option key={r} value={r}>{r}</option>)}
        </select>
        <input
          type="text"
          placeholder="https://github.com/org/repo"
          value={initUrl}
          onChange={(e) => setInitUrl(e.target.value)}
          className="bg-[var(--surface)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm flex-1 max-w-sm"
        />
        <button
          onClick={handleInit}
          className="bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white px-4 py-2 rounded-lg text-sm font-medium transition"
        >
          Add Repo
        </button>
      </div>
      {initStatus && (
        <div className="px-6 py-2 text-sm text-[var(--muted)]">{initStatus}</div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 px-6 border-b border-[var(--border)]">
        {['chat', 'train'].map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-3 text-sm font-medium capitalize border-b-2 transition ${
              tab === t
                ? 'border-[var(--accent)] text-[var(--accent)]'
                : 'border-transparent text-[var(--muted)] hover:text-[var(--text)]'
            }`}
          >
            {t === 'train' ? 'Fine-tune' : t}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {tab === 'chat' && <ChatTab slug={selected} />}
        {tab === 'train' && <TrainTab slug={selected} />}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// CHAT TAB
// ═══════════════════════════════════════════════════════════════════════
function ChatTab({ slug }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('rag');
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = async () => {
    if (!input.trim() || !slug) return;
    const userMsg = { role: 'user', content: input };
    setMessages((m) => [...m, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_slug: slug, query: input, mode }),
      });
      const data = await res.json();
      setMessages((m) => [...m, {
        role: 'assistant',
        content: data.answer,
        citations: data.citations || [],
        elapsed: data.elapsed_ms,
        mode: data.mode,
        showSources: false,
      }]);
    } catch (e) {
      setMessages((m) => [...m, { role: 'assistant', content: `Error: ${e}` }]);
    }
    setLoading(false);
  };

  const toggleSources = (idx) => {
    setMessages((m) => m.map((msg, i) =>
      i === idx ? { ...msg, showSources: !msg.showSources } : msg
    ));
  };

  return (
    <div className="flex flex-col h-full">
      {/* Mode toggle */}
      <div className="flex items-center gap-2 px-6 py-2">
        <button
          onClick={() => setMode('rag')}
          className={`px-3 py-1 rounded-full text-xs font-medium transition ${
            mode === 'rag' ? 'bg-[var(--accent)] text-white' : 'bg-[var(--surface)] text-[var(--muted)]'
          }`}
        >RAG-only (base)</button>
        <button
          onClick={() => setMode('hybrid')}
          className={`px-3 py-1 rounded-full text-xs font-medium transition ${
            mode === 'hybrid' ? 'bg-[var(--accent)] text-white' : 'bg-[var(--surface)] text-[var(--muted)]'
          }`}
        >Hybrid (fine-tuned)</button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-[var(--muted)] mt-20">
            <p className="text-lg">Ask a question about {slug || 'your repo'}</p>
            <p className="text-sm mt-2">e.g. "What does the main function do?"</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-3xl rounded-2xl px-4 py-3 ${
              msg.role === 'user'
                ? 'bg-[var(--accent)] text-white'
                : 'bg-[var(--surface)] border border-[var(--border)]'
            }`}>
              <div className="whitespace-pre-wrap text-sm">{msg.content}</div>

              {msg.citations?.length > 0 && (
                <div className="mt-2">
                  <button
                    onClick={() => toggleSources(i)}
                    className="text-xs text-[var(--accent)] hover:underline"
                  >
                    {msg.showSources ? 'Hide sources' : `Show sources (${msg.citations.length})`}
                  </button>
                  {msg.showSources && (
                    <div className="mt-2 pt-2 border-t border-[var(--border)] space-y-1">
                      {msg.citations.map((c, j) => (
                        <div key={j} className="text-xs text-[var(--muted)] font-mono">
                          {c.file}:{c.start_line}-{c.end_line}
                          {c.symbol && ` — ${c.symbol}`}
                          {c.relation !== 'target' && ` [${c.relation}]`}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {msg.elapsed && (
                <div className="mt-2 text-xs text-[var(--muted)]">{msg.mode} · {msg.elapsed}ms</div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-[var(--surface)] border border-[var(--border)] rounded-2xl px-4 py-3">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-[var(--muted)] rounded-full animate-bounce" style={{animationDelay:'0ms'}}></span>
                <span className="w-2 h-2 bg-[var(--muted)] rounded-full animate-bounce" style={{animationDelay:'150ms'}}></span>
                <span className="w-2 h-2 bg-[var(--muted)] rounded-full animate-bounce" style={{animationDelay:'300ms'}}></span>
              </div>
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* Input */}
      <div className="border-t border-[var(--border)] px-6 py-3">
        <div className="flex gap-2 max-w-3xl mx-auto">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !loading && send()}
            placeholder="Ask about this codebase..."
            className="flex-1 bg-[var(--surface)] border border-[var(--border)] rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-[var(--accent)]"
            disabled={!slug || loading}
          />
          <button
            onClick={send}
            disabled={!input.trim() || loading || !slug}
            className="bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white px-5 py-3 rounded-xl text-sm font-medium transition disabled:opacity-40"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// TRAIN TAB
// ═══════════════════════════════════════════════════════════════════════
function TrainTab({ slug }) {
  const [epochs, setEpochs] = useState(1);
  const [maxChunks, setMaxChunks] = useState(200);
  const [skipData, setSkipData] = useState(false);
  const [training, setTraining] = useState(false);
  const [log, setLog] = useState('');
  const [progress, setProgress] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const startTrain = async () => {
    setTraining(true);
    setLog('Starting training...');
    setProgress({ phase: 'starting', percent: 0 });
    try {
      const res = await fetch(`${API}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_slug: slug,
          epochs,
          max_chunks: maxChunks,
          skip_data: skipData,
        }),
      });
      const data = await res.json();
      setLog(`Training started!`);

      // Poll for progress
      pollRef.current = setInterval(async () => {
        try {
          const sres = await fetch(`${API}/train/status?slug=${slug}`);
          const sdata = await sres.json();

          if (sdata.adapter_ready) {
            setProgress({ phase: 'done', percent: 100 });
            setLog('Training complete! Adapter is ready.');
            setTraining(false);
            if (pollRef.current) clearInterval(pollRef.current);
            return;
          }

          // Parse log for progress clues
          const logText = sdata.log || '';
          let phase = 'running';
          let percent = 50;

          if (logText.includes('Phase 3: Data generation')) {
            phase = 'Generating training data';
            percent = 20;
          } else if (logText.includes('Generating synth pairs')) {
            phase = 'Generating Q&A pairs (LLM API)';
            percent = 30;
          } else if (logText.includes('Assembling')) {
            phase = 'Assembling dataset';
            percent = 45;
          } else if (logText.includes('Phase 5: QLoRA')) {
            phase = 'QLoRA fine-tuning on GPU';
            percent = 55;
          } else if (logText.includes('Starting QLoRA training')) {
            phase = 'Training model';
            percent = 65;
          } else if (logText.includes('Saved adapter')) {
            phase = 'Adapter saved';
            percent = 85;
          } else if (logText.includes('Eval:')) {
            phase = 'Running evaluation';
            percent = 90;
          } else if (logText.includes('Training complete')) {
            phase = 'Complete';
            percent = 100;
          } else if (logText.includes('Error') || logText.includes('Traceback')) {
            phase = 'Error — check log';
            percent = 0;
            setTraining(false);
            if (pollRef.current) clearInterval(pollRef.current);
          }

          setProgress({ phase, percent });
          setLog(logText.slice(-500));
        } catch {}
      }, 3000);

    } catch (e) {
      setLog(`Error: ${e}`);
      setTraining(false);
    }
  };

  const stopTrain = () => {
    setTraining(false);
    setProgress({ phase: 'stopped', percent: 0 });
    setLog('Training stopped (subprocess may still be running).');
    if (pollRef.current) clearInterval(pollRef.current);
  };

  return (
    <div className="max-w-2xl mx-auto px-6 py-8 space-y-6">
      <div>
        <h2 className="text-xl font-bold mb-2">Fine-tune on {slug}</h2>
        <p className="text-sm text-[var(--muted)]">
          Generates Q&A training data from your code then fine-tunes
          the model with QLoRA to learn your repo's style and patterns.
        </p>
      </div>

      {/* Config */}
      <div className="bg-[var(--surface)] border border-[var(--border)] rounded-xl p-6 space-y-4">
        <h3 className="text-sm font-semibold uppercase text-[var(--muted)]">Training Config</h3>
        <div className="grid grid-cols-2 gap-4">
          <label className="block">
            <span className="text-xs text-[var(--muted)]">Epochs</span>
            <input
              type="number" min="1" max="4" value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm mt-1"
            />
          </label>
          <label className="block">
            <span className="text-xs text-[var(--muted)]">Max chunks for data gen</span>
            <input
              type="number" min="10" max="500" value={maxChunks}
              onChange={(e) => setMaxChunks(parseInt(e.target.value))}
              className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm mt-1"
            />
          </label>
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox" checked={skipData}
            onChange={(e) => setSkipData(e.target.checked)}
            className="accent-[var(--accent)]"
          />
          <span>Skip data generation (use existing train.jsonl)</span>
        </label>
      </div>

      {/* Progress */}
      {progress && (
        <div className="bg-[var(--surface)] border border-[var(--border)] rounded-xl p-6 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">{progress.phase}</span>
            <span className="text-xs text-[var(--muted)]">{progress.percent}%</span>
          </div>
          <div className="w-full h-2 bg-[var(--bg)] rounded-full overflow-hidden">
            <div
              className="h-full bg-[var(--accent)] rounded-full transition-all duration-500"
              style={{ width: `${progress.percent}%` }}
            />
          </div>
          {progress.phase === 'done' && (
            <div className="text-sm text-green-500">
              Adapter ready! Restart serve and switch to Hybrid mode to use it.
            </div>
          )}
          {progress.phase === 'error' && (
            <div className="text-sm text-red-400">
              Training failed. Check the log below for details.
            </div>
          )}
        </div>
      )}

      {/* Log */}
      {log && (
        <div className="bg-[var(--surface)] border border-[var(--border)] rounded-xl p-4">
          <pre className="text-xs text-[var(--muted)] whitespace-pre-wrap font-mono max-h-48 overflow-y-auto">{log}</pre>
        </div>
      )}

      {/* Buttons */}
      <div className="flex gap-3">
        <button
          onClick={startTrain}
          disabled={training || !slug}
          className="flex-1 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white py-3 rounded-xl font-medium transition disabled:opacity-40"
        >
          {training ? 'Training in progress...' : 'Start Fine-tuning'}
        </button>
        {training && (
          <button
            onClick={stopTrain}
            className="px-5 py-3 bg-red-900/30 border border-red-800 text-red-400 rounded-xl text-sm font-medium transition hover:bg-red-900/50"
          >
            Stop
          </button>
        )}
      </div>
    </div>
  );
}
