'use strict';

const TOKEN_KEY = 'deepseek2api_admin_token';
const API_BASE = '/admin/api';

function getToken() { return localStorage.getItem(TOKEN_KEY) || ''; }
function setToken(t) { localStorage.setItem(TOKEN_KEY, t); }
function clearToken() { localStorage.removeItem(TOKEN_KEY); }

function showToast(msg, kind = 'info') {
  const el = document.getElementById('toast');
  el.className = `toast ${kind}`;
  el.textContent = msg;
  el.classList.remove('hidden');
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => el.classList.add('hidden'), 3500);
}

async function api(name, opts = {}) {
  const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
  const token = getToken();
  if (token) headers['X-Admin-Token'] = token;

  let resp;
  try {
    resp = await fetch(`${API_BASE}/${name}`, {
      method: opts.method || 'GET',
      headers,
      body: opts.body ? JSON.stringify(opts.body) : undefined,
    });
  } catch (e) {
    throw new Error('网络错误');
  }

  if (resp.status === 401) {
    clearToken();
    showLogin();
    throw new Error('unauthorized');
  }

  let data = null;
  try { data = await resp.json(); } catch (_) {}

  if (!resp.ok) {
    throw new Error((data && data.error) || `HTTP ${resp.status}`);
  }
  return data;
}

function fmtTime(ts) {
  if (!ts) return '-';
  const d = new Date(ts * 1000);
  return d.toLocaleString('zh-CN', { hour12: false });
}

function fmtDuration(sec) {
  if (sec == null) return '-';
  if (sec < 60) return `${Math.floor(sec)}秒`;
  if (sec < 3600) return `${Math.floor(sec / 60)}分${Math.floor(sec % 60)}秒`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}小时${Math.floor((sec % 3600) / 60)}分`;
  return `${Math.floor(sec / 86400)}天${Math.floor((sec % 86400) / 3600)}小时`;
}

function escapeHtml(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function showLogin() {
  document.getElementById('login-screen').classList.remove('hidden');
  document.getElementById('app').classList.add('hidden');
}

function showApp() {
  document.getElementById('login-screen').classList.add('hidden');
  document.getElementById('app').classList.remove('hidden');
  refresh();
}

async function handleLoginSubmit(e) {
  e.preventDefault();
  const pw = document.getElementById('login-password').value;
  const errEl = document.getElementById('login-error');
  errEl.hidden = true;
  try {
    const data = await api('login', { method: 'POST', body: { password: pw } });
    setToken(data.token);
    showApp();
    startAutoRefresh();
  } catch (err) {
    errEl.textContent = '登录失败：' + err.message;
    errEl.hidden = false;
  }
}

async function handleLogout() {
  try { await api('logout', { method: 'POST' }); } catch (_) {}
  stopAutoRefresh();
  clearToken();
  showLogin();
}

let currentView = 'dashboard';
const VIEW_TITLES = {
  dashboard: '仪表盘',
  logs: '请求日志',
  models: '模型列表',
  config: '配置查看',
};

function switchView(name) {
  currentView = name;
  document.querySelectorAll('.nav-item').forEach(a => {
    a.classList.toggle('active', a.dataset.view === name);
  });
  document.querySelectorAll('.view').forEach(s => {
    s.classList.toggle('active', s.id === `view-${name}`);
  });
  document.getElementById('topbar-title').textContent = VIEW_TITLES[name] || name;
  refresh();
}

let refreshTimer = null;

function startAutoRefresh() {
  stopAutoRefresh();
  if (document.getElementById('auto-refresh').checked) {
    refreshTimer = setInterval(() => refresh(), 5000);
  }
}

function stopAutoRefresh() {
  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = null;
}

async function refresh() {
  if (!getToken()) return;
  try {
    switch (currentView) {
      case 'dashboard': await refreshDashboard(); break;
      case 'logs': await refreshLogs(); break;
      case 'models': await refreshModels(); break;
      case 'config': await refreshConfig(); break;
    }
  } catch (err) {
    if (err.message !== 'unauthorized') console.error('refresh failed', err);
  }
}

async function refreshDashboard() {
  const d = await api('dashboard');
  const all = d.all_time;
  const r5 = d.recent_5m;
  const successRateColor = r5.success_rate >= 95 ? 'success' : r5.success_rate >= 80 ? 'warning' : 'error';

  let html = `
  <div class="kpi-grid">
    <div class="kpi-card info">
      <div class="kpi-label">总请求数</div>
      <div class="kpi-value">${(all.total || 0).toLocaleString()}</div>
      <div class="kpi-sub">成功 ${all.success} · 4xx ${all.client_errors} · 5xx ${all.server_errors}</div>
    </div>
    <div class="kpi-card ${successRateColor}">
      <div class="kpi-label">5分钟成功率</div>
      <div class="kpi-value">${(r5.success_rate || 0).toFixed(1)}%</div>
      <div class="kpi-sub">${r5.success}/${r5.total} 请求</div>
    </div>
    <div class="kpi-card warning">
      <div class="kpi-label">P50 延迟</div>
      <div class="kpi-value">${r5.p50_ms}ms</div>
      <div class="kpi-sub">P95 ${r5.p95_ms}ms</div>
    </div>
    <div class="kpi-card success">
      <div class="kpi-label">运行时长</div>
      <div class="kpi-value" style="font-size:18px;">${fmtDuration(d.uptime_seconds)}</div>
    </div>
  </div>
  <div class="panel">
    <div class="panel-header"><div class="panel-title">Top 模型</div></div>
    <div class="panel-body">
      ${(d.top_models || []).map(m => `<div class="flex-between" style="padding:6px 0;"><span class="mono">${escapeHtml(m.model)}</span><strong>${m.count}</strong></div>`).join('') || '<div class="empty-state">暂无数据</div>'}
    </div>
  </div>`;

  document.getElementById('view-dashboard').innerHTML = html;
}

async function refreshLogs() {
  const data = await api('logs?limit=100');
  const logs = data.logs || [];

  let html = `<div class="table-wrapper"><table class="data-table">
    <thead><tr><th>时间</th><th>协议</th><th>方法</th><th>路径</th><th>模型</th><th>状态</th><th>延迟</th><th>流式</th><th>错误</th></tr></thead>
    <tbody>`;

  if (logs.length === 0) {
    html += '<tr><td colspan="9" class="empty-state">暂无日志</td></tr>';
  } else {
    for (const l of logs) {
      const statusClass = l.status >= 500 ? 'text-error' : l.status >= 400 ? 'text-warning' : 'text-success';
      html += `<tr>
        <td class="mono text-muted" style="font-size:11px;white-space:nowrap;">${fmtTime(l.ts)}</td>
        <td>${escapeHtml(l.protocol)}</td>
        <td class="mono">${escapeHtml(l.method)}</td>
        <td class="mono" style="max-width:200px;overflow:hidden;text-overflow:ellipsis;">${escapeHtml(l.path)}</td>
        <td class="mono">${escapeHtml(l.model || '-')}</td>
        <td class="mono"><span class="${statusClass}">${l.status}</span></td>
        <td class="mono">${l.duration_ms}ms</td>
        <td>${l.stream ? 'stream' : 'sync'}</td>
        <td class="mono text-muted" style="font-size:11px;">${escapeHtml(l.error || '')}</td>
      </tr>`;
    }
  }

  html += '</tbody></table></div>';
  document.getElementById('view-logs').innerHTML = html;
}

async function refreshModels() {
  const data = await api('models');
  const models = data.models || [];

  let html = `<div class="panel"><div class="panel-header"><div class="panel-title">可用模型 (${models.length})</div></div><div class="panel-body">`;
  for (const m of models) {
    html += `<div class="flex-between" style="padding:8px 0;border-bottom:1px solid var(--border);"><span class="mono">${escapeHtml(m)}</span></div>`;
  }
  html += '</div></div>';
  document.getElementById('view-models').innerHTML = html;
}

async function refreshConfig() {
  const cfg = await api('config');

  let html = '<div class="panel"><div class="panel-header"><div class="panel-title">运行时配置</div></div><div class="panel-body"><div class="config-list">';
  for (const [k, v] of Object.entries(cfg)) {
    html += `<div class="config-row"><div class="config-key">${escapeHtml(k)}</div><div class="config-value">${escapeHtml(String(v))}</div></div>`;
  }
  html += '</div></div></div>';
  document.getElementById('view-config').innerHTML = html;
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('login-form').addEventListener('submit', handleLoginSubmit);
  document.getElementById('logout-btn').addEventListener('click', handleLogout);
  document.getElementById('auto-refresh').addEventListener('change', startAutoRefresh);
  document.getElementById('manual-refresh').addEventListener('click', () => refresh());

  document.querySelectorAll('.nav-item').forEach(a => {
    a.addEventListener('click', (e) => {
      e.preventDefault();
      switchView(a.dataset.view);
    });
  });

  if (getToken()) {
    api('dashboard').then(() => {
      showApp();
      startAutoRefresh();
    }).catch(() => {
      clearToken();
      showLogin();
    });
  } else {
    showLogin();
  }
});