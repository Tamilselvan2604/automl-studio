import './App.css';
import { useMemo, useRef, useState } from 'react';
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts';

const API_BASE = (import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '')).replace(/\/$/, '');

const SAMPLE_HINTS = [
  { ext: 'CSV / TSV', note: 'Best for most tabular datasets and labeled text data.' },
  { ext: 'XLSX', note: 'Excel sheets with a single data table.' },
  { ext: 'JSON', note: 'An array of records or newline-delimited JSON objects.' },
  { ext: 'TXT', note: 'Raw text for profiling, or one label<TAB>text pair per line for text classification.' },
];

function Icon({ name, size = 18 }) {
  const icons = {
    spark: '✦', upload: '↑', dataset: '▦', settings: '⚙', play: '▶',
    check: '✓', file: '▤', chart: '▥', shield: '◇', arrow: '↗',
    flask: '⚗', brain: '◉', download: '↓', warning: '!', refresh: '↻',
  };
  return <span className="icon-glyph" style={{ fontSize: size }}>{icons[name] || '•'}</span>;
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  return Number(value).toFixed(digits);
}

function percent(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return 'Unknown size';
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} bytes`;
}

function StatCard({ label, value, detail, tone = '' }) {
  return (
    <div className={`stat-card ${tone}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {detail && <div className="stat-detail">{detail}</div>}
    </div>
  );
}

function SectionHeading({ eyebrow, title, detail, right }) {
  return (
    <div className="section-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h2>{title}</h2>
        {detail && <p>{detail}</p>}
      </div>
      {right}
    </div>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <strong>{label}</strong>
      {payload.map((item) => (
        <div key={item.dataKey} className="tooltip-row">
          <span>{item.name || item.dataKey}</span>
          <b>{typeof item.value === 'number' ? item.value.toFixed(3) : item.value}</b>
        </div>
      ))}
    </div>
  );
}

function ModelChart({ results }) {
  const chartData = useMemo(() => (results?.charts?.model_comparison || []).map((item) => ({
    ...item,
    displayScore: results.task === 'classification' ? item.score * 100 : item.score,
  })), [results]);
  if (!chartData.length) return <div className="empty-chart">Run an experiment to compare model scores.</div>;
  return (
    <ResponsiveContainer width="100%" height={Math.max(220, chartData.length * 52)}>
      <BarChart data={chartData} layout="vertical" margin={{ top: 8, right: 22, bottom: 0, left: 8 }}>
        <CartesianGrid stroke="#263244" strokeDasharray="3 4" horizontal={false} />
        <XAxis type="number" tick={{ fill: '#8998ad', fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis type="category" dataKey="model" width={145} tick={{ fill: '#c4cede', fontSize: 11 }} axisLine={false} tickLine={false} />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(129,140,248,.08)' }} />
        <Bar dataKey="displayScore" name={results.task === 'classification' ? 'Accuracy (%)' : 'R²'} radius={[0, 7, 7, 0]} barSize={23}>
          {chartData.map((entry) => (
            <Cell key={entry.model} fill={entry.is_baseline ? '#546176' : entry.model === results.best_model ? '#a3e635' : '#7c8cf8'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function MissingChart({ results }) {
  const data = results?.charts?.missing_by_column || [];
  if (!data.length) return <div className="empty-chart">Missingness chart appears after training.</div>;
  if (data.every((d) => d.missing_percent === 0)) return <div className="chart-empty-state"><span className="success-mark">✓</span><b>No missing cells detected</b><span>This dataset is complete in the uploaded sample.</span></div>;
  return (
    <ResponsiveContainer width="100%" height={Math.max(220, data.length * 34)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 18, left: 8, bottom: 0 }}>
        <CartesianGrid stroke="#263244" strokeDasharray="3 4" horizontal={false} />
        <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fill: '#8998ad', fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis type="category" dataKey="column" width={118} tick={{ fill: '#c4cede', fontSize: 10 }} axisLine={false} tickLine={false} />
        <Tooltip content={<ChartTooltip />} />
        <Bar dataKey="missing_percent" name="Missing (%)" fill="#fb923c" radius={[0, 6, 6, 0]} barSize={18} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function DistributionChart({ results }) {
  const data = results?.charts?.target_distribution || [];
  if (!data.length) return <div className="empty-chart">Target distribution will appear here.</div>;
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -18, bottom: 38 }}>
        <CartesianGrid stroke="#263244" strokeDasharray="3 4" vertical={false} />
        <XAxis dataKey="label" interval={0} angle={-22} textAnchor="end" height={58} tick={{ fill: '#8998ad', fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: '#8998ad', fontSize: 10 }} axisLine={false} tickLine={false} />
        <Tooltip content={<ChartTooltip />} />
        <Bar dataKey="count" name="Rows" fill="#38bdf8" radius={[6, 6, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function FeatureChart({ results }) {
  const data = results?.charts?.feature_importance || [];
  if (!data.length) return <div className="empty-chart">This model doesn't expose a simple global feature-importance measure.</div>;
  return (
    <ResponsiveContainer width="100%" height={Math.max(240, data.length * 32)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 14, left: 12, bottom: 0 }}>
        <CartesianGrid stroke="#263244" strokeDasharray="3 4" horizontal={false} />
        <XAxis type="number" tick={{ fill: '#8998ad', fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis type="category" dataKey="feature" width={150} tick={{ fill: '#c4cede', fontSize: 10 }} axisLine={false} tickLine={false} />
        <Tooltip content={<ChartTooltip />} />
        <Bar dataKey="importance" name="Relative importance" fill="#c084fc" radius={[0, 6, 6, 0]} barSize={18} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function ConfusionMatrix({ data }) {
  if (!data) return <div className="empty-chart">Matrix hidden for high-cardinality classification targets to keep it readable.</div>;
  const maxValue = Math.max(1, ...data.matrix.flat());
  return (
    <div className="matrix-wrap">
      <div className="matrix-title">Actual labels × predicted labels</div>
      <div className="matrix" style={{ '--matrix-size': data.labels.length }}>
        <div className="matrix-corner" />
        {data.labels.map((label) => <div className="matrix-label" key={`top-${label}`} title={label}>{label}</div>)}
        {data.matrix.map((row, rowIndex) => (
          <div className="matrix-row" key={`row-${data.labels[rowIndex]}`}>
            <div className="matrix-label" title={data.labels[rowIndex]}>{data.labels[rowIndex]}</div>
            {row.map((value, colIndex) => (
              <div
                className="matrix-cell"
                key={`${rowIndex}-${colIndex}`}
                title={`Actual ${data.labels[rowIndex]}, predicted ${data.labels[colIndex]}: ${value}`}
                style={{ backgroundColor: `rgba(163, 230, 53, ${0.08 + (value / maxValue) * 0.82})` }}
              >{value}</div>
            ))}
          </div>
        ))}
      </div>
      <div className="matrix-note">Darker green means more predictions in that cell. Diagonal cells are correct predictions.</div>
    </div>
  );
}

function App() {
  const fileInputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [inspection, setInspection] = useState(null);
  const [results, setResults] = useState(null);
  const [target, setTarget] = useState('');
  const [task, setTask] = useState('auto');
  const [testSize, setTestSize] = useState(0.2);
  const [seed, setSeed] = useState(42);
  const [busy, setBusy] = useState(false);
  const [busyMode, setBusyMode] = useState('');
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('overview');
  const [dragging, setDragging] = useState(false);

  const handleFile = async (selectedFile) => {
    if (!selectedFile) return;
    const allowed = /\.(csv|tsv|xlsx|json|txt)$/i.test(selectedFile.name);
    if (!allowed) {
      setError('Choose a CSV, TSV, XLSX, JSON, or TXT dataset.');
      return;
    }
    if (selectedFile.size > 120 * 1024 * 1024) {
      setError('File is larger than 120 MB. Please upload a smaller export or CSV sample.');
      return;
    }
    setFile(selectedFile);
    setInspection(null);
    setResults(null);
    setError('');
    setBusy(true);
    setBusyMode('inspect');
    try {
      if (!API_BASE) throw new Error('This deployed frontend has no API URL configured yet. Set the VITE_API_URL repository variable to your Python API URL, then redeploy the site.');
      const form = new FormData();
      form.append('file', selectedFile);
      const response = await fetch(`${API_BASE}/api/inspect`, { method: 'POST', body: form });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Inspection failed (${response.status}).`);
      setInspection(payload);
      setTarget(payload.target_suggestions?.[0] || payload.columns?.[0]?.name || '');
      setActiveTab('overview');
    } catch (err) {
      setError(API_BASE ? `${err.message || 'Could not inspect this file.'} Make sure the Python API is running at ${API_BASE}.` : (err.message || 'Set VITE_API_URL to your Python API URL and redeploy.'));
    } finally {
      setBusy(false);
      setBusyMode('');
    }
  };

  const loadSampleDataset = async () => {
    setError('');
    try {
      const sampleUrl = `${import.meta.env.BASE_URL || '/'}sample-dataset.csv`;
      const response = await fetch(sampleUrl);
      if (!response.ok) throw new Error('The sample dataset could not be loaded. You can still upload your own CSV or Excel file.');
      const blob = await response.blob();
      const sampleFile = new File([blob], 'employee_attrition_sample.csv', { type: 'text/csv' });
      await handleFile(sampleFile);
    } catch (err) {
      setError(err.message || 'Unable to load the sample dataset.');
    }
  };

  const runTraining = async (event) => {
    event.preventDefault();
    if (!file || !target) {
      setError('Upload a dataset and select a target column first.');
      return;
    }
    setBusy(true);
    setBusyMode('train');
    setError('');
    setResults(null);
    try {
      if (!API_BASE) throw new Error('This deployed frontend has no API URL configured yet. Set the VITE_API_URL repository variable to your Python API URL, then redeploy the site.');
      const form = new FormData();
      form.append('file', file);
      form.append('target', target);
      form.append('task', task);
      form.append('test_size', String(testSize));
      form.append('random_state', String(seed));
      const response = await fetch(`${API_BASE}/api/train`, { method: 'POST', body: form });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Training failed (${response.status}).`);
      setResults(payload);
      setActiveTab('results');
      setTimeout(() => document.getElementById('results-panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
    } catch (err) {
      setError(err.message || 'Training failed. Check your dataset and target column.');
    } finally {
      setBusy(false);
      setBusyMode('');
    }
  };

  const downloadReport = () => {
    if (!results) return;
    const rows = results.leaderboard;
    const headers = Object.keys(rows[0] || {});
    const csv = [headers.join(','), ...rows.map((row) => headers.map((header) => {
      const value = row[header] ?? '';
      return `"${String(value).replaceAll('"', '""')}"`;
    }).join(','))].join('\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8;' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'automl-studio-model-comparison.csv';
    link.click();
    URL.revokeObjectURL(url);
  };

  const summary = results?.file_summary;
  const primaryScore = results?.task === 'classification' ? percent(results.best_score) : formatNumber(results?.best_score);
  const scoreCaption = results?.task === 'classification' ? 'holdout accuracy' : 'holdout R²';

  return (
    <div className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <header className="topbar">
        <a className="brand" href="#top" aria-label="AutoML Studio home">
          <span className="brand-icon"><Icon name="spark" size={22} /></span>
          <span><b>Xplore</b><span className="brand-light">InnoQuest</span><small>AUTO ML STUDIO</small></span>
        </a>
        <div className="topbar-right">
          <span className="status-pill"><span className="status-dot" /> EXPERIMENT WORKSPACE</span>
          {API_BASE && <a className="docs-link" href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">API docs <Icon name="arrow" size={14} /></a>}
        </div>
      </header>

      <main id="top" className="main-content">
        <section className="hero">
          <div className="hero-copy">
            <div className="hero-kicker"><span className="kicker-line" /> YOUR DATA. TESTED BY SCIENCE.</div>
            <h1>Find the model<br />that <span>fits your data.</span></h1>
            <p className="hero-description">A transparent AutoML workspace that profiles messy datasets, builds preprocessing pipelines, tests multiple models, and explains what performed best.</p>
            <div className="hero-points">
              <span><Icon name="check" size={14} /> Automatic data profiling</span>
              <span><Icon name="check" size={14} /> Honest holdout metrics</span>
              <span><Icon name="check" size={14} /> Explainable comparisons</span>
            </div>
          </div>
          <div className="hero-visual" aria-hidden="true">
            <div className="orbit orbit-large" /><div className="orbit orbit-small" />
            <div className="visual-core"><span className="core-spark">✳</span><span className="core-label">AUTO<br />ML</span></div>
            <div className="float-chip chip-top"><span className="chip-glyph purple">⌁</span><span><b>Data prep</b><small>Normalize · encode</small></span></div>
            <div className="float-chip chip-right"><span className="chip-glyph green">↗</span><span><b>Model score</b><small>Measure · compare</small></span></div>
            <div className="float-chip chip-bottom"><span className="chip-glyph blue">▥</span><span><b>Top 3</b><small>Ranked candidates</small></span></div>
            <div className="visual-star star-a">✦</div><div className="visual-star star-b">✳</div>
          </div>
        </section>

        <section className="workspace-grid" id="workspace">
          <aside className="sidebar panel">
            <div className="side-title"><span className="side-icon"><Icon name="flask" /></span><div><b>Experiment setup</b><small>01 — Configure</small></div></div>
            <div className="side-rule" />
            <div className="field-label"><span className="step-number">1</span> Upload dataset</div>
            <div
              className={`upload-zone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
              role="button"
              tabIndex={0}
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') fileInputRef.current?.click(); }}
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => { event.preventDefault(); setDragging(false); handleFile(event.dataTransfer.files?.[0]); }}
            >
              <input ref={fileInputRef} type="file" accept=".csv,.tsv,.xlsx,.json,.txt" hidden onClick={(event) => event.stopPropagation()} onChange={(event) => { const selected = event.target.files?.[0]; event.target.value = ''; if (selected) handleFile(selected); }} />
              <div className="upload-icon"><Icon name={file ? 'check' : 'upload'} size={22} /></div>
              {file ? <><b className="file-name">{file.name}</b><span>{formatBytes(file.size)} · click to replace</span></> : <><b>Drop your dataset here</b><span>or browse from your device</span></>}
              <div className="file-badges"><span>CSV</span><span>XLSX</span><span>JSON</span><span>TXT</span></div>
              <small className="upload-limit">Up to 120 MB · max 250k rows</small>
            </div>
            {inspection && (
              <div className="mini-profile">
                <div className="mini-profile-head"><span className="status-dot" /> Dataset profiled</div>
                <div className="mini-profile-stats"><div><b>{inspection.rows.toLocaleString()}</b><small>Rows</small></div><div><b>{inspection.columns_count}</b><small>Columns</small></div><div><b>{inspection.duplicate_rows.toLocaleString()}</b><small>Duplicates</small></div></div>
                <div className="data-kind">{inspection.data_kind}</div>
              </div>
            )}

            <form onSubmit={runTraining} className="config-form">
              <div className="field-label"><span className="step-number">2</span> Define target</div>
              <label className="input-label" htmlFor="target-select">What should the model predict?</label>
              <select id="target-select" value={target} onChange={(event) => setTarget(event.target.value)} disabled={!inspection}>
                {!inspection && <option value="">Upload data first</option>}
                {inspection?.columns?.map((column) => <option value={column.name} key={column.name}>{column.name}</option>)}
              </select>
              {inspection?.data_kind === 'Unstructured text' && <p className="inline-note">Raw text has no labels yet. For a supervised accuracy score, provide labeled examples using CSV columns or a label, then a tab character, then text on each TXT line.</p>}

              <label className="input-label" htmlFor="task-select">Prediction task</label>
              <select id="task-select" value={task} onChange={(event) => setTask(event.target.value)}>
                <option value="auto">Auto-detect task</option><option value="classification">Classification</option><option value="regression">Regression</option>
              </select>
              <div className="split-heading"><label className="input-label" htmlFor="split-range">Test set size</label><b>{Math.round(testSize * 100)}%</b></div>
              <input id="split-range" className="range-input" type="range" min="0.15" max="0.35" step="0.05" value={testSize} onChange={(event) => setTestSize(Number(event.target.value))} />
              <div className="range-captions"><span>{Math.round((1 - testSize) * 100)}% train</span><span>{Math.round(testSize * 100)}% test</span></div>

              <label className="input-label" htmlFor="seed-input">Random seed <span className="label-muted">(reproducible runs)</span></label>
              <input id="seed-input" className="text-input" type="number" min="0" max="100000" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
              <button className="run-button" type="submit" disabled={!inspection || !target || busy}>
                {busy && busyMode === 'train' ? <span className="spinner" /> : <Icon name="play" size={15} />}
                {busy && busyMode === 'train' ? 'Training candidates…' : 'Train & compare models'}
                {!busy && <span className="button-arrow">↗</span>}
              </button>
              <p className="run-hint">Trains a baseline plus four candidate models on your selected split.</p>
            </form>
            <div className="privacy-note"><Icon name="shield" size={16} /><span><b>Prototype privacy</b><small>Uploads are processed for the request and not saved as persistent files by this API. Don't upload sensitive, personal or regulated datasets.</small></span></div>
          </aside>

          <section className="right-workspace">
            {error && <div className="error-banner" role="alert"><span className="error-symbol">!</span><div><b>Something needs attention</b><p>{error}</p></div><button onClick={() => setError('')} aria-label="Dismiss error">×</button></div>}
            {busy && busyMode === 'inspect' && <div className="loading-banner"><span className="spinner" /><div><b>Reading your dataset…</b><p>Profiling column types, missing values, duplicates, and preview rows.</p></div></div>}

            {!inspection && !busy && (
              <div className="welcome-panel panel">
                <div className="welcome-art"><div className="welcome-ring ring-1" /><div className="welcome-ring ring-2" /><div className="welcome-center"><Icon name="dataset" size={42} /></div><div className="welcome-node node-1">CSV</div><div className="welcome-node node-2">ML</div><div className="welcome-node node-3">AI</div></div>
                <div className="eyebrow">READY WHEN YOU ARE</div>
                <h2>Turn raw data into<br /><span>evidence-backed choices.</span></h2>
                <p>Upload a dataset to begin. AutoML Studio will inspect its shape, highlight data-quality issues, and prepare an experiment you can understand—not just a score you have to trust.</p>
                <div className="welcome-actions">
                  <button className="secondary-button primary-action" onClick={() => fileInputRef.current?.click()}><Icon name="upload" size={16} /> Upload your dataset <Icon name="arrow" size={15} /></button>
                  <button className="sample-button" onClick={loadSampleDataset}><Icon name="play" size={13} /> Try a sample dataset</button>
                </div>
                <div className="supported-formats">{SAMPLE_HINTS.map((item) => <div key={item.ext}><b>{item.ext}</b><span>{item.note}</span></div>)}</div>
              </div>
            )}

            {inspection && (
              <>
                <div className="content-tabs">
                  <button className={activeTab === 'overview' ? 'active' : ''} onClick={() => setActiveTab('overview')}><Icon name="dataset" size={15} /> Data overview</button>
                  <button className={activeTab === 'results' ? 'active' : ''} onClick={() => setActiveTab('results')} disabled={!results}><Icon name="chart" size={15} /> Model results {!results && <span className="tab-lock">· run first</span>}</button>
                  <span className="tab-spacer" />
                  {results && <button className="export-button" onClick={downloadReport}><Icon name="download" size={14} /> Export CSV</button>}
                </div>

                {activeTab === 'overview' && (
                  <div className="overview-stack">
                    <div className="section-heading compact-heading"><div><div className="eyebrow">DATASET INTELLIGENCE</div><h2>Know your data.</h2><p>A quick profile before the experiment begins.</p></div><span className="file-chip"><Icon name="file" size={14} /> {inspection.file_name}</span></div>
                    <div className="stats-grid four">
                      <StatCard label="Total rows" value={inspection.rows.toLocaleString()} detail="Raw observations" />
                      <StatCard label="Features" value={inspection.columns_count} detail="Available columns" />
                      <StatCard label="Missing cells" value={inspection.missing_cells.toLocaleString()} detail={`${inspection.rows ? ((inspection.missing_cells / (inspection.rows * inspection.columns_count)) * 100).toFixed(1) : 0}% of all cells`} tone={inspection.missing_cells ? 'warning-stat' : ''} />
                      <StatCard label="Duplicate rows" value={inspection.duplicate_rows.toLocaleString()} detail="Exact row matches" tone={inspection.duplicate_rows ? 'warning-stat' : ''} />
                    </div>
                    <div className="panel data-profile-panel">
                      <SectionHeading eyebrow="FIRST LOOK" title="Sample records" detail="Preview only. The uploaded dataset is not modified on your device." right={<span className="tag tag-cyan">{inspection.data_kind}</span>} />
                      <div className="table-scroll"><table className="data-table"><thead><tr>{inspection.columns.map((col) => <th key={col.name}>{col.name}<small>{col.kind}</small></th>)}</tr></thead><tbody>{inspection.preview.map((row, index) => <tr key={index}>{inspection.columns.map((col) => <td key={col.name} title={String(row[col.name] ?? '')}>{row[col.name] === null || row[col.name] === '' ? <span className="null-cell">NULL</span> : String(row[col.name]).slice(0, 72)}</td>)}</tr>)}</tbody></table></div>
                      <div className="table-foot"><span>Showing {inspection.preview.length} of {inspection.rows.toLocaleString()} rows</span><span>{inspection.columns_count} columns detected</span></div>
                    </div>
                    <div className="panel profile-table-panel">
                      <SectionHeading eyebrow="QUALITY SCAN" title="Column profile" detail="Column types and missingness help guide preprocessing." />
                      <div className="table-scroll"><table className="data-table profile-table"><thead><tr><th>Column</th><th>Detected type</th><th>Missing</th><th>Unique</th><th>Signal</th></tr></thead><tbody>{inspection.columns.map((col) => <tr key={col.name}><td><b>{col.name}</b></td><td><span className={`type-pill ${col.kind}`}>{col.kind}</span></td><td><div className="missing-cell"><div className="mini-progress"><span style={{ width: `${col.missing_percent}%` }} /></div><span>{col.missing_percent}%</span></div></td><td>{col.unique.toLocaleString()}</td><td>{col.kind === 'text' ? <span className="signal-text">TF-IDF candidate</span> : col.missing ? <span className="signal-warn">Imputation needed</span> : <span className="signal-good">Looks complete</span>}</td></tr>)}</tbody></table></div>
                    </div>
                    <div className="info-callout"><span className="callout-icon">i</span><div><b>One thing to know before training</b><p>Choose a target that represents the output you want to predict. Columns containing future information, IDs, or duplicated labels can inflate scores. AutoML can help compare models, but it cannot decide whether your dataset represents the real problem.</p></div></div>
                  </div>
                )}

                {activeTab === 'results' && results && (
                  <div id="results-panel" className="results-stack">
                    <div className="section-heading compact-heading"><div><div className="eyebrow">EXPERIMENT COMPLETE</div><h2>Evidence over guesswork.</h2><p>Ranked using a held-out test set the models did not train on.</p></div><span className="tag tag-green"><span className="status-dot" /> RUN COMPLETE</span></div>
                    <div className="stats-grid four results-stats">
                      <StatCard label="Top model" value={results.best_model} detail="Best candidate by primary metric" />
                      <StatCard label={results.task === 'classification' ? 'Accuracy' : 'R² score'} value={primaryScore} detail={scoreCaption} tone="success-stat" />
                      <StatCard label="Training rows" value={summary.train_rows.toLocaleString()} detail={`${Math.round((1 - results.test_size) * 100)}% of usable rows`} />
                      <StatCard label="Test rows" value={summary.test_rows.toLocaleString()} detail={`${Math.round(results.test_size * 100)}% holdout`} />
                    </div>

                    <div className="top3-grid">
                      {results.top3.map((model, index) => (
                        <article className={`top-model-card rank-${index + 1}`} key={model.model}>
                          <div className="rank-label"><span>0{index + 1}</span>{index === 0 ? ' TOP PERFORMER' : index === 1 ? ' RUNNER UP' : ' THIRD PLACE'}</div>
                          <h3>{model.model}</h3>
                          <div className="model-score">{results.task === 'classification' ? percent(model.accuracy) : formatNumber(model.r2)}<small>{results.task === 'classification' ? ' accuracy' : ' R² score'}</small></div>
                          <p>{model.reason}</p>
                          <div className="model-mini-metrics">{results.task === 'classification' ? <><span>Weighted F1 <b>{formatNumber(model.weighted_f1)}</b></span><span>Balanced acc. <b>{formatNumber(model.balanced_accuracy)}</b></span></> : <><span>MAE <b>{formatNumber(model.mae)}</b></span><span>RMSE <b>{formatNumber(model.rmse)}</b></span></>}</div>
                        </article>
                      ))}
                    </div>

                    {results.warnings?.length > 0 && <div className="warnings-list">{results.warnings.map((warning, index) => <div className="warning-item" key={index}><span>!</span><p>{warning}</p></div>)}</div>}

                    <div className="chart-grid">
                      <div className="panel chart-panel chart-wide"><SectionHeading eyebrow="MODEL LEADERBOARD" title={results.task === 'classification' ? 'Accuracy comparison' : 'R² comparison'} detail={results.task === 'classification' ? 'Higher is better. Dummy baseline is included for context.' : 'R² can be negative when a model performs worse than predicting the target mean.'} /><ModelChart results={results} /><div className="chart-legend"><span><i className="legend-green" /> Best candidate</span><span><i className="legend-purple" /> Other candidates</span><span><i className="legend-gray" /> Dummy baseline</span></div></div>
                      <div className="panel chart-panel"><SectionHeading eyebrow="DATA QUALITY" title="Missing values" detail="Top columns by missing percentage before preprocessing." /><MissingChart results={results} /></div>
                      <div className="panel chart-panel"><SectionHeading eyebrow="TARGET PROFILE" title="Target distribution" detail={results.task === 'classification' ? 'Counts by target class.' : 'Target values grouped into numeric bins.'} /><DistributionChart results={results} /></div>
                      <div className="panel chart-panel"><SectionHeading eyebrow="MODEL INTERPRETATION" title="Feature signals" detail="Relative importance from the winning estimator; not causal effect." /><FeatureChart results={results} /></div>
                      {results.task === 'classification' && <div className="panel chart-panel chart-wide"><SectionHeading eyebrow="ERROR ANALYSIS" title="Confusion matrix" detail={`Predictions from ${results.best_model} on the held-out test rows.`} /><ConfusionMatrix data={results.charts.confusion_matrix} /></div>}
                    </div>

                    <div className="panel leaderboard-panel"><SectionHeading eyebrow="ALL CANDIDATES" title="Full model leaderboard" detail="Dummy baseline is a reference point, not one of the three recommended models." /><div className="table-scroll"><table className="data-table leaderboard-table"><thead><tr><th>Model</th>{results.task === 'classification' ? <><th>Accuracy</th><th>Weighted F1</th><th>Balanced accuracy</th></> : <><th>R²</th><th>MAE ↓</th><th>RMSE ↓</th></>}<th>Fit time</th><th>Status</th></tr></thead><tbody>{results.leaderboard.map((model) => <tr key={model.model}><td><b>{model.model}</b>{model.is_baseline && <small className="baseline-note">Reference only</small>}</td>{results.task === 'classification' ? <><td>{percent(model.accuracy)}</td><td>{formatNumber(model.weighted_f1)}</td><td>{formatNumber(model.balanced_accuracy)}</td></> : <><td>{formatNumber(model.r2)}</td><td>{formatNumber(model.mae)}</td><td>{formatNumber(model.rmse)}</td></>}<td>{model.fit_seconds}s</td><td>{model.model === results.best_model ? <span className="tag tag-green">Top score</span> : model.is_baseline ? <span className="tag">Baseline</span> : <span className="tag tag-purple">Evaluated</span>}</td></tr>)}</tbody></table></div>{results.skipped_models?.length > 0 && <div className="skipped-note"><b>Some candidates were skipped:</b> {results.skipped_models.map((item) => `${item.model}: ${item.reason}`).join(' · ')}</div>}</div>

                    <div className="panel pipeline-panel"><SectionHeading eyebrow="WHAT THE PIPELINE DID" title="From raw data to tested models" detail="Preprocessing is fitted on training data only to reduce leakage." /><div className="pipeline-flow">{[{n:'01',title:'Inspect',desc:'Schema, types, missingness'},{n:'02',title:'Clean',desc:'Duplicates, empty cols, target gaps'},{n:'03',title:'Transform',desc:'Impute, scale, encode, TF-IDF'},{n:'04',title:'Split',desc:'Train and held-out test sets'},{n:'05',title:'Compare',desc:'Baseline + four candidates'}].map((step, index) => <div className="pipeline-step-wrap" key={step.n}><div className="pipeline-step"><span>{step.n}</span><b>{step.title}</b><small>{step.desc}</small></div>{index < 4 && <div className="flow-arrow">→</div>}</div>)}</div><div className="cleaning-summary"><h3><Icon name="check" size={16} /> Cleaning and preprocessing log</h3><ul>{results.preprocessing.steps.map((step, index) => <li key={index}>{step}</li>)}</ul><div className="preprocess-tags">{results.preprocessing.numeric_columns.length > 0 && <span className="type-pill numeric">Numeric · {results.preprocessing.numeric_columns.length}</span>}{results.preprocessing.categorical_columns.length > 0 && <span className="type-pill categorical">Categorical · {results.preprocessing.categorical_columns.length}</span>}{results.preprocessing.text_columns_tfidf.length > 0 && <span className="type-pill text">TF-IDF · {results.preprocessing.text_columns_tfidf.length}</span>}{summary.identifier_columns_excluded.length > 0 && <span className="type-pill excluded">IDs excluded · {summary.identifier_columns_excluded.length}</span>}</div></div></div>
                    <div className="closing-note"><span>✳</span><div><b>What does “best” mean here?</b><p>It means the top score among these candidate models on this one held-out split. For classification, compare weighted F1 and balanced accuracy too. For regression, choose based on your tolerance for error as well as R². Validate the winner on fresh data before relying on it.</p></div></div>
                  </div>
                )}
              </>
            )}
          </section>
        </section>

        <footer className="footer"><a className="brand footer-brand" href="#top"><span className="brand-icon"><Icon name="spark" size={17} /></span><span><b>Xplore</b><span className="brand-light">InnoQuest</span></span></a><p>Built for curious minds. Designed for transparent experimentation.</p><span className="footer-version">PROTOTYPE · v1.0</span></footer>
      </main>
    </div>
  );
}

export default App;
