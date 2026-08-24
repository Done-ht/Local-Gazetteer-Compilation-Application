/* PDF 工具 网页版前端逻辑
 * 双模式：本机访问 → 文件就地处理（路径浏览器选本机路径，无上传）
 *         远程访问 → 上传/下载
 * 进度通过 SSE 推送，支持取消；合并/合成/拼接/插入支持撤销重做与拖拽排序。
 */
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const ts = () => {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
};
const fmtSize = (n) => {
  if (!n) return "";
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / 1024 / 1024).toFixed(1) + " MB";
};
const toast = (msg) => {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), 2500);
};

/* =================== 全局状态 =================== */
const MODE = { local: "local", remote: "remote" };
const state = {
  mode: null,
  panels: {
    merge: { items: [], undo: [], redo: [], taskId: null, sse: null, busy: false },
    split: { item: null, taskId: null, sse: null, busy: false },
    convert: { item: null, taskId: null, sse: null, busy: false },
    compose: { items: [], undo: [], redo: [], taskId: null, sse: null, busy: false },
    append: { items: [], undo: [], redo: [], taskId: null, sse: null, busy: false },
    insert: { base: null, items: [], undo: [], redo: [], taskId: null, sse: null, busy: false },
  },
};

/* =================== API 封装 =================== */
function _handle401(r) {
  // 未登录或会话过期 → 跳登录页
  if (r.status === 401) { window.location.href = "/login"; return true; }
  return false;
}
async function apiPost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (_handle401(r)) return { error: "未登录" };
  return r.json();
}
async function uploadFiles(fileList) {
  const fd = new FormData();
  for (const f of fileList) fd.append("files", f);
  const r = await fetch("/api/upload", { method: "POST", body: fd });
  if (_handle401(r)) return [];
  return (await r.json()).files || [];
}
async function fetchLocalPages(path) {
  try {
    const d = await apiPost("/api/local/file_info", { path });
    return d.page_count || 0;
  } catch { return 0; }
}

/* =================== 路径浏览器（本机模式） =================== */
const PathBrowser = {
  cfg: null,
  current: "",
  parent: null,
  selected: new Set(),
  _quick: null,
  _canStore: (() => { try { localStorage.setItem("__t", "1"); localStorage.removeItem("__t"); return true; } catch { return false; } })(),

  async open(opts) {
    // opts: { mode: 'file'|'dir', filter: 'pdf'|'image'|'all', multi, title, onConfirm }
    this.cfg = opts;
    this.selected = new Set();
    $("pathModal-title").textContent = opts.title || "选择路径";
    $("pathModal-hint").textContent = opts.multi ? "可多选" : "选择后点击确定";
    $("pathModal-confirm").textContent = opts.mode === "dir" ? "选择此目录" : "确定";
    // 让弹窗主色跟随当前激活的功能面板（转换=橙、拆分=蓝…），
    // 这样“确定”按钮颜色与触发它的面板一致，视觉关联更清晰
    const activePanel = document.querySelector(".panel.active");
    if (activePanel) {
      const accent = getComputedStyle(activePanel).getPropertyValue("--accent").trim();
      if (accent) $("pathModal").style.setProperty("--accent", accent);
    }
    $("pathModal").hidden = false;
    await this._ensureQuick();
    this.load(opts.initialPath || this._lastDir() || "");
  },
  close() { $("pathModal").hidden = true; this.cfg = null; },

  _lastDir() { return this._canStore ? (localStorage.getItem("pdfpb:last") || "") : ""; },
  _readRecent() { try { return JSON.parse(localStorage.getItem("pdfpb:recent") || "[]"); } catch { return []; } },
  _addRecent(path) {
    if (!path || !this._canStore) return;
    let r = this._readRecent().filter(p => p !== path);
    r.unshift(path);
    if (r.length > 8) r.length = 8;
    localStorage.setItem("pdfpb:recent", JSON.stringify(r));
  },
  async _ensureQuick() {
    if (this._quick) return this._quick;
    const d = await apiPost("/api/local/quick_paths", {});
    this._quick = d.error ? [] : (d.paths || []);
    return this._quick;
  },

  async load(path) {
    const d = await apiPost("/api/local/list_dir", { path });
    if (d.error) { toast(d.error); return; }
    this.current = d.path;
    this.parent = d.parent;
    $("pathModal-path").value = d.path;
    if (this._canStore) localStorage.setItem("pdfpb:last", d.path);
    this.render(d);
  },

  render(d) {
    const ul = $("pathModal-list");
    ul.innerHTML = "";
    if (d.parent) {
      const li = document.createElement("li");
      li.innerHTML = `<span class="ic">⬆️</span><span>返回上一级</span>`;
      li.onclick = () => this.load(this.parent);
      ul.appendChild(li);
    }
    for (const name of d.dirs) {
      const li = document.createElement("li");
      li.className = "is-dir";
      li.innerHTML = `<span class="ic"></span><span class="name">${esc(name)}</span>`;
      li.onclick = () => this.load(d.path ? (d.path.replace(/\/?$/, "") + "/" + name) : name);
      ul.appendChild(li);
    }
    for (const f of d.files) {
      if (!this._match(f)) continue;
      const li = document.createElement("li");
      li.className = "is-file " + (f.is_pdf ? "is-pdf" : "is-img");
      li.dataset.path = f.path;
      li.innerHTML = `<span class="ic"></span><span class="name">${esc(f.name)}</span><span class="meta">${fmtSize(f.size)}</span>`;
      if (this.cfg.mode === "file") {
        li.onclick = () => this._toggle(li, f);
      }
      ul.appendChild(li);
    }
    this._renderCrumb();
    this._renderSide();
  },
  _renderCrumb() {
    const cur = this.current;
    const box = $("pathModal-crumb");
    box.innerHTML = "";
    if (!cur) return;
    // 统一分隔符后按级拆分，拼接时沿用原路径风格（Windows 用 \，POSIX 用 /）
    const parts = cur.replace(/\\/g, "/").split("/").filter(Boolean);
    const sep = cur.includes("\\") ? "\\" : "/";
    const leadingSlash = cur.startsWith("/");
    let cum = "";
    parts.forEach((seg, i) => {
      if (i === 0) {
        cum = /^[A-Za-z]:$/.test(seg) ? seg + "\\" : (leadingSlash ? "/" + seg : seg);
      } else {
        cum = cum.replace(/[\\/]+$/, "") + sep + seg;
      }
      if (i > 0) {
        const s = document.createElement("span");
        s.className = "crumb-sep";
        s.textContent = "/";
        box.appendChild(s);
      }
      const a = document.createElement("span");
      a.className = "crumb-item" + (i === parts.length - 1 ? " current" : "");
      a.textContent = seg;
      if (i < parts.length - 1) {
        const target = cum;  // 固定当前级别路径，避免闭包共享 cum 导致点击都跳到末级
        a.onclick = () => this.load(target);
      }
      box.appendChild(a);
    });
  },
  _renderSide() {
    const q = $("pathModal-quick");
    q.innerHTML = "";
    for (const e of (this._quick || [])) {
      const li = document.createElement("li");
      if (e.path === this.current) li.className = "active";
      li.title = e.path;
      li.innerHTML = `<span class="ic">${e.icon || "📁"}</span><span class="name">${esc(e.label)}</span>`;
      li.onclick = () => this.load(e.path);
      q.appendChild(li);
    }
    const r = $("pathModal-recent");
    r.innerHTML = "";
    const recent = this._readRecent();
    if (!recent.length) {
      r.innerHTML = `<li class="side-empty">暂无</li>`;
      return;
    }
    for (const p of recent) {
      const name = p.split(/[\\/]/).pop() || p;
      const li = document.createElement("li");
      if (p === this.current) li.className = "active";
      li.title = p;
      li.innerHTML = `<span class="ic">🕘</span><span class="name">${esc(name)}</span>`;
      li.onclick = () => this.load(p);
      r.appendChild(li);
    }
  },
  _match(f) {
    if (this.cfg.mode === "dir") return false;
    const flt = this.cfg.filter || "all";
    if (flt === "pdf") return f.is_pdf;
    if (flt === "image") return f.is_image;
    return true;
  },
  _toggle(li, f) {
    if (this.cfg.multi) {
      if (this.selected.has(f.path)) { this.selected.delete(f.path); li.classList.remove("selected"); }
      else { this.selected.add(f.path); li.classList.add("selected"); }
    } else {
      ul: for (const el of $("pathModal-list").children) el.classList.remove("selected");
      this.selected.clear();
      this.selected.add(f.path);
      li.classList.add("selected");
    }
  },
  confirm() {
    if (this.cfg.mode === "dir") {
      this._addRecent(this.current);
      this.cfg.onConfirm(this.current);
    } else if (this.cfg.mode === "file") {
      const paths = [...this.selected];
      if (!paths.length) { toast("请先选择文件"); return; }
      this._addRecent(this.current);
      this.cfg.onConfirm(paths);
    }
    this.close();
  },
};
$("pathModal-close").onclick = () => PathBrowser.close();
$("pathModal-cancel").onclick = () => PathBrowser.close();
$("pathModal-up").onclick = () => PathBrowser.parent && PathBrowser.load(PathBrowser.parent);
$("pathModal-go").onclick = () => PathBrowser.load($("pathModal-path").value);
$("pathModal-confirm").onclick = () => PathBrowser.confirm();

/* =================== 文件列表渲染（通用） =================== */
function renderList(panelId) {
  const st = state.panels[panelId];
  const ul = $(panelId + "-list");
  const items = panelId === "insert" ? st.items : st.items;
  ul.innerHTML = "";
  items.forEach((it, i) => {
    const li = document.createElement("li");
    li.dataset.idx = i;
    const meta = it.pageCount != null ? it.pageCount + " 页" : (it.size ? fmtSize(it.size) : "");
    li.innerHTML = `<span class="idx">${i + 1}</span><span class="name">${esc(it.name)}</span>` +
      `<span class="meta">${meta}</span><span class="del" title="移除">×</span>`;
    li.querySelector(".del").onclick = (e) => { e.stopPropagation(); pushUndo(panelId); items.splice(i, 1); renderList(panelId); };
    ul.appendChild(li);
  });
  if (ul.dataset.draggable === "true") enableDrag(ul, panelId);
  updateSummary(panelId);
}
function updateSummary(panelId) {
  const st = state.panels[panelId];
  const el = $(panelId + "-summary");
  if (!el) return;
  if (panelId === "merge" || panelId === "append") {
    const n = st.items.length;
    const pages = st.items.reduce((s, i) => s + (i.pageCount || 0), 0);
    el.textContent = `共 ${n} 个文件，${pages} 页`;
  } else if (panelId === "compose") {
    el.textContent = `共 ${st.items.length} 张图片`;
  } else if (panelId === "insert") {
    el.textContent = `共 ${st.items.length} 个插入文件`;
  }
}
/* 单文件面板（split/convert/insert-base）渲染 */
function renderSingle(panelId, item, suffix) {
  const ul = $(panelId + (suffix || "-list"));
  const sum = $(panelId + (suffix || "") + "-summary");
  ul.innerHTML = "";
  if (!item) { if (sum) sum.textContent = "未选择文件"; return; }
  const li = document.createElement("li");
  const meta = item.pageCount != null ? item.pageCount + " 页" : (item.size ? fmtSize(item.size) : "");
  li.innerHTML = `<span class="name">${esc(item.name)}</span><span class="meta">${meta}</span><span class="del" title="移除">×</span>`;
  li.querySelector(".del").onclick = () => { clearSingle(panelId, suffix); };
  ul.appendChild(li);
  if (sum) sum.textContent = `${item.name}`;
}

/* =================== 拖拽排序 =================== */
function enableDrag(ul, panelId) {
  let dragIdx = null;
  ul.querySelectorAll("li").forEach((li) => {
    li.draggable = true;
    li.addEventListener("dragstart", () => { dragIdx = +li.dataset.idx; li.classList.add("dragging"); });
    li.addEventListener("dragend", () => { li.classList.remove("dragging"); ul.querySelectorAll("li").forEach(x => x.classList.remove("drag-over")); });
    li.addEventListener("dragover", (e) => { e.preventDefault(); li.classList.add("drag-over"); });
    li.addEventListener("dragleave", () => li.classList.remove("drag-over"));
    li.addEventListener("drop", (e) => {
      e.preventDefault();
      const dropIdx = +li.dataset.idx;
      if (dragIdx === null || dragIdx === dropIdx) return;
      const items = state.panels[panelId].items;
      pushUndo(panelId);
      const [m] = items.splice(dragIdx, 1);
      items.splice(dropIdx, 0, m);
      renderList(panelId);
    });
  });
}

/* =================== 撤销 / 重做 =================== */
function pushUndo(panelId) {
  const st = state.panels[panelId];
  if (!st.undo) return;
  st.undo.push(JSON.parse(JSON.stringify(st.items)));
  st.redo = [];
}
function undo(panelId) {
  const st = state.panels[panelId];
  if (!st.undo || !st.undo.length) { toast("无可撤销"); return; }
  st.redo.push(JSON.parse(JSON.stringify(st.items)));
  st.items = st.undo.pop();
  renderList(panelId);
}
function redo(panelId) {
  const st = state.panels[panelId];
  if (!st.redo || !st.redo.length) { toast("无可重做"); return; }
  st.undo.push(JSON.parse(JSON.stringify(st.items)));
  st.items = st.redo.pop();
  renderList(panelId);
}

/* =================== 添加文件（本机/远程分流） =================== */
function addFilesTrigger(panelId, opts) {
  // opts: { filter: 'pdf'|'image'|'all', multi, target: 'items'|'single', suffix }
  if (state.mode === MODE.local) {
    PathBrowser.open({
      mode: "file", filter: opts.filter, multi: opts.multi !== false,
      title: opts.title || "选择文件",
      onConfirm: (paths) => addLocalPaths(panelId, paths, opts),
    });
  } else {
    const inp = $("hidden-file-input");
    inp.multiple = opts.multi !== false;
    inp.accept = opts.filter === "pdf" ? ".pdf" : (opts.filter === "image" ? ".jpg,.jpeg,.png,.bmp,.gif,.tiff,.tif,.webp" : ".pdf,.jpg,.jpeg,.png,.bmp,.gif,.tiff,.tif,.webp");
    inp.onchange = async () => {
      if (!inp.files.length) return;
      const files = await uploadFiles(inp.files);
      addUploadedItems(panelId, files, opts);
      inp.value = "";
    };
    inp.click();
  }
}
function pathToItem(p) {
  const name = p.split(/[\\/]/).pop();
  const ext = (name.split(".").pop() || "").toLowerCase();
  return { name, path: p, ext, isPdf: ext === "pdf", isImage: [".jpg",".jpeg",".png",".bmp",".gif",".tiff",".tif",".webp"].includes("." + ext), pageCount: null };
}
async function addLocalPaths(panelId, paths, opts) {
  const st = state.panels[panelId];
  if (opts.target === "single") {
    const it = pathToItem(paths[0]);
    if (opts.filter === "pdf" && !it.isPdf) { toast("请选择 PDF 文件"); return; }
    st[opts.singleKey || "item"] = it;
    renderSingle(panelId, it, opts.suffix);
    if (it.isPdf) { it.pageCount = await fetchLocalPages(it.path); renderSingle(panelId, it, opts.suffix); }
    onSingleAdded(panelId);
  } else {
    pushUndo(panelId);
    for (const p of paths) {
      const it = pathToItem(p);
      if (panelId === "merge" && !it.isPdf) continue;
      if (panelId === "compose" && !it.isImage) continue;
      st.items.push(it);
    }
    renderList(panelId);
    // 异步补全页数
    for (const it of st.items) {
      if (it.isPdf && it.pageCount == null) {
        fetchLocalPages(it.path).then(n => { it.pageCount = n; renderList(panelId); });
      }
    }
    onMultiAdded(panelId);
  }
}
function addUploadedItems(panelId, files, opts) {
  const st = state.panels[panelId];
  const toIt = (f) => {
    const ext = (f.name.split(".").pop() || "").toLowerCase();
    return { name: f.name, file_id: f.file_id, ext, isPdf: ext === "pdf",
      isImage: [".jpg",".jpeg",".png",".bmp",".gif",".tiff",".tif",".webp"].includes("." + ext), size: f.size, pageCount: null };
  };
  if (opts.target === "single") {
    st[opts.singleKey || "item"] = toIt(files[0]);
    renderSingle(panelId, st[opts.singleKey || "item"], opts.suffix);
    onSingleAdded(panelId);
  } else {
    pushUndo(panelId);
    for (const f of files) {
      const it = toIt(f);
      if (panelId === "merge" && !it.isPdf) continue;
      if (panelId === "compose" && !it.isImage) continue;
      st.items.push(it);
    }
    renderList(panelId);
    onMultiAdded(panelId);
  }
}
function onSingleAdded(panelId) {
  // 自动填默认输出路径（本机模式）
  if (state.mode !== MODE.local) return;
  const st = state.panels[panelId];
  if (panelId === "split") {
    if (!$("split-output").value && st.item) {
      const dir = st.item.path.split(/[\\/]/).slice(0, -1).join("\\");
      const base = st.item.name.replace(/\.pdf$/i, "");
      $("split-output").value = dir + "\\" + base + "_拆分结果";
    }
  } else if (panelId === "convert") {
    if (!$("convert-output").value && st.item) {
      const dir = st.item.path.split(/[\\/]/).slice(0, -1).join("\\");
      const base = st.item.name.replace(/\.pdf$/i, "");
      $("convert-output").value = dir + "\\_" + base + ".docx";
    }
  }
}
function onMultiAdded(panelId) {
  if (state.mode !== MODE.local) return;
  const st = state.panels[panelId];
  const fill = (id, val) => { if (!$(id).value) $(id).value = val; };
  if (panelId === "merge" && st.items[0]) {
    const dir = st.items[0].path.split(/[\\/]/).slice(0, -1).join("\\");
    fill("merge-output", dir + "\\_合并结果_" + ts() + ".pdf");
  } else if (panelId === "compose" && st.items[0]) {
    const dir = st.items[0].path.split(/[\\/]/).slice(0, -1).join("\\");
    fill("compose-output", dir + "\\合成结果_" + ts() + ".pdf");
  } else if (panelId === "append" && st.items[0]) {
    const dir = st.items[0].path.split(/[\\/]/).slice(0, -1).join("\\");
    fill("append-output", dir + "\\拼接结果_" + ts() + ".pdf");
  }
}

/* =================== 清空单文件 =================== */
function clearSingle(panelId, suffix) {
  const st = state.panels[panelId];
  if (panelId === "insert" && suffix === "-base") { st.base = null; renderSingle("insert", null, "-base"); $("insert-base-summary").textContent = "未选择基础 PDF"; $("insert-page-hint").textContent = ""; return; }
  st.item = null;
  renderSingle(panelId, null, suffix);
}

/* =================== 进度 / 任务 =================== */
function setBusy(panelId, busy) {
  const st = state.panels[panelId];
  st.busy = busy;
  const panel = $("panel-" + panelId);
  panel.querySelector('[data-action="start"]').disabled = busy;
  panel.querySelector('[data-action="cancel"]').disabled = !busy;
  if (!busy) {} 
}
function updateProgress(panelId, d) {
  const wrap = $(panelId + "-progress");
  wrap.hidden = false;
  const pct = d.total > 0 ? Math.round(d.current / d.total * 100) : 0;
  wrap.querySelector(".progress-bar").style.width = pct + "%";
  wrap.querySelector(".progress-text").textContent = `${pct}% · ${d.message || ""}`;
}
function showResult(panelId, type, html) {
  $(panelId + "-result").innerHTML = `<div class="alert alert-${type}">${html}</div>`;
}
function parsePages(text) {
  text = String(text).replace(/[，、]/g, ",").replace(/\s+/g, ",");
  const out = [];
  for (const part of text.split(",")) {
    const p = part.trim();
    if (!p) continue;
    if (p.includes("-")) {
      const [a, b] = p.split("-");
      for (let i = parseInt(a); i <= parseInt(b); i++) if (!isNaN(i) && i > 0) out.push(i);
    } else {
      const n = parseInt(p);
      if (!isNaN(n) && n > 0) out.push(n);
    }
  }
  return out;
}

function startTask(panelId, url, body, label) {
  const st = state.panels[panelId];
  apiPost(url, body).then((d) => {
    if (d.error) { showResult(panelId, "error", esc(d.error)); return; }
    const taskId = d.task_id;
    st.taskId = taskId;
    setBusy(panelId, true);
    $(panelId + "-progress").hidden = false;
    $(panelId + "-result").innerHTML = "";
    const es = new EventSource(`/api/task/${taskId}/progress`);
    st.sse = es;
    es.addEventListener("progress", (e) => updateProgress(panelId, JSON.parse(e.data)));
    es.addEventListener("done", (e) => {
      const data = JSON.parse(e.data);
      onComplete(panelId, data);
      es.close();
    });
    es.addEventListener("error", (e) => {
      if (e.data) {
        let msg = "任务出错";
        try { msg = JSON.parse(e.data).message || msg; } catch {}
        showResult(panelId, "error", esc(msg));
        es.close();
        setBusy(panelId, false);
      }
    });
    es.addEventListener("cancelled", () => {
      showResult(panelId, "error", "任务已取消");
      es.close();
      setBusy(panelId, false);
    });
    es.addEventListener("end", () => { es.close(); setBusy(panelId, false); });
  });
}
function onComplete(panelId, data) {
  const st = state.panels[panelId];
  setBusy(panelId, false);
  const wrap = $(panelId + "-progress");
  wrap.querySelector(".progress-bar").style.width = "100%";
  wrap.querySelector(".progress-text").textContent = "完成";
  const resultEl = $(panelId + "-result");
  if (state.mode === MODE.local) {
    // split 面板的 output_path 始终是目录（range/extract 也写入该目录下的单一 PDF）
    const isDirPath = data.output_is_dir || panelId === "split";
    const dir = isDirPath ? data.output_path : (data.output_path || "").split(/[\\/]/).slice(0, -1).join("\\");
    resultEl.innerHTML = `<div class="alert alert-success">完成！输出：<span class="meta">${esc(data.output_path || "")}</span> <a class="open-dir-link">打开输出目录</a></div>`;
    resultEl.querySelector(".open-dir-link").onclick = () => openDir(dir);
  } else {
    const isZip = data.output_is_dir;
    const label = isZip ? "下载结果（ZIP）" : "下载结果";
    resultEl.innerHTML = `<div class="alert alert-success">完成！<a href="/api/download/${st.taskId}" target="_blank">${label}</a> <a class="cleanup-link">清理临时文件</a></div>`;
    resultEl.querySelector(".cleanup-link").onclick = () => cleanupTask(st.taskId);
  }
}
function openDir(dir) { apiPost("/api/local/open_dir", { path: dir }).then(d => d.ok ? toast("已打开目录") : toast("打开失败")); }
function cleanupTask(id) { apiPost(`/api/task/${id}/cleanup`, {}).then(() => toast("已清理临时文件")); }

function cancelTask(panelId) {
  const st = state.panels[panelId];
  if (st.taskId) apiPost(`/api/task/${st.taskId}/cancel`, {}).then(() => toast("已请求取消"));
}

/* =================== 各功能启动 =================== */
function launchMerge() {
  const st = state.panels.merge;
  const sort = $("merge-sort").value;
  const body = { mode: state.mode, fast_mode: $("merge-fast").checked };
  if (state.mode === MODE.local) {
    body.sort_mode = sort;
    if (sort === "specified") body.files = st.items.map(i => i.path);
    else if (sort === "folder") body.folder_path = $("merge-folder").value;
    else body.root_path = $("merge-folder").value;
    body.output_path = $("merge-output").value;
    if (!body.output_path) { toast("请选择输出路径"); return; }
  } else {
    body.files = st.items.map(i => i.file_id);
  }
  if (sort === "specified" && !body.files.length) { toast("请先添加文件"); return; }
  if ((sort === "folder" || sort === "by_folder") && !body.folder_path) { toast("请选择文件夹"); return; }
  startTask("merge", "/api/merge/start", body, "合并");
}
function launchSplit() {
  const st = state.panels.split;
  if (!st.item) { toast("请选择 PDF"); return; }
  const smode = $("split-mode").value;
  // 模式与输入框不一致时给出提示，避免填了页码却被静默忽略
  if (smode === "range" && !$("split-range").value.trim()) { toast("请填写页码范围，如 1-3,5,8-10"); return; }
  if (smode === "extract" && !$("split-extract").value.trim()) { toast("请填写提取页码，如 1,3,5 或 1-3"); return; }
  const body = { mode: state.mode, split_mode: smode };
  if (state.mode === MODE.local) {
    body.input_path = st.item.path;
    body.output_dir = $("split-output").value;
    if (!body.output_dir) { toast("请选择输出目录"); return; }
  } else {
    body.files = [st.item.file_id];
  }
  if (smode === "range") body.range_text = $("split-range").value;
  else if (smode === "extract") body.extract_pages = parsePages($("split-extract").value);
  startTask("split", "/api/split/start", body, "拆分");
}
function launchConvert() {
  const st = state.panels.convert;
  if (!st.item) { toast("请选择 PDF"); return; }
  const body = { mode: state.mode, dpi: parseInt($("convert-dpi").value) || 150 };
  if (state.mode === MODE.local) {
    body.input_path = st.item.path;
    body.output_path = $("convert-output").value;
    if (!body.output_path) { toast("请选择输出路径"); return; }
  } else {
    body.files = [st.item.file_id];
  }
  startTask("convert", "/api/convert/start", body, "转换");
}
function launchCompose() {
  const st = state.panels.compose;
  if (!st.items.length) { toast("请添加图片"); return; }
  const body = { mode: state.mode, files: state.mode === MODE.local ? st.items.map(i => i.path) : st.items.map(i => i.file_id) };
  if (state.mode === MODE.local) {
    body.output_path = $("compose-output").value;
    if (!body.output_path) { toast("请选择输出路径"); return; }
  }
  startTask("compose", "/api/compose/start", body, "合成");
}
function launchAppend() {
  const st = state.panels.append;
  if (!st.items.length) { toast("请添加文件"); return; }
  const body = { mode: state.mode, files: state.mode === MODE.local ? st.items.map(i => i.path) : st.items.map(i => i.file_id) };
  if (state.mode === MODE.local) {
    body.output_path = $("append-output").value;
    if (!body.output_path) { toast("请选择输出路径"); return; }
  }
  startTask("append", "/api/append/start", body, "拼接");
}
function launchInsert() {
  const st = state.panels.insert;
  if (!st.base) { toast("请选择基础 PDF"); return; }
  if (!st.items.length) { toast("请添加插入内容"); return; }
  const body = {
    mode: state.mode,
    insert_page: parseInt($("insert-page").value) - 1,
    files: state.mode === MODE.local ? st.items.map(i => i.path) : st.items.map(i => i.file_id),
  };
  if (state.mode === MODE.local) {
    body.base_pdf = st.base.path;
    body.output_path = $("insert-output").value;
    if (!body.output_path) { toast("请选择输出路径"); return; }
  } else {
    body.base_pdf = st.base.file_id;
  }
  startTask("insert", "/api/insert/start", body, "插入");
}

/* =================== 事件绑定 =================== */
function bindPanel(panelId, cfg) {
  const panel = $("panel-" + panelId);
  panel.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-action]");
    if (!btn) return;
    const act = btn.dataset.action;
    const st = state.panels[panelId];
    if (st.busy && act !== "cancel") { toast("任务进行中，请等待或取消"); return; }
    switch (act) {
      case "add-files": addFilesTrigger(panelId, cfg.fileOpts); break;
      case "add-base":
        addFilesTrigger("insert", { filter: "pdf", multi: false, target: "single", singleKey: "base", suffix: "-base", title: "选择基础 PDF" });
        break;
      case "undo": undo(panelId); break;
      case "redo": redo(panelId); break;
      case "clear":
        if (panelId === "insert") { st.items = []; renderList(panelId); }
        else { st.items = []; st.undo = []; st.redo = []; renderList(panelId); }
        break;
      case "browse-folder":
        PathBrowser.open({ mode: "dir", title: "选择文件夹", onConfirm: (p) => { $("merge-folder").value = p; } });
        break;
      case "browse-output-file": browseOutput(panelId, "file"); break;
      case "browse-output-dir": browseOutput(panelId, "dir"); break;
      case "browse-output-docx": browseOutput(panelId, "docx"); break;
      case "start": cfg.launch(); break;
      case "cancel": cancelTask(panelId); break;
    }
  });
}
function browseOutput(panelId, kind) {
  if (state.mode !== MODE.local) { toast("远程模式由服务器自动生成输出路径"); return; }
  PathBrowser.open({
    mode: "dir", title: "选择输出目录",
    onConfirm: (dir) => {
      const setVal = (id, v) => { $(id).value = v; };
      if (panelId === "split") { setVal("split-output", dir); return; }
      const defaults = {
        merge: "_合并结果_" + ts() + ".pdf",
        compose: "合成结果_" + ts() + ".pdf",
        append: "拼接结果_" + ts() + ".pdf",
        convert: () => { const it = state.panels.convert.item; return it ? "_" + it.name.replace(/\.pdf$/i, "") + ".docx" : "输出.docx"; },
        insert: () => { const b = state.panels.insert.base; return b ? "_" + b.name.replace(/\.pdf$/i, "") + "_插入结果_" + ts() + ".pdf" : "插入结果.pdf"; },
      };
      const fn = defaults[panelId];
      const name = typeof fn === "function" ? fn() : fn;
      setVal(panelId + "-output", dir + "\\" + name);
    },
  });
}

/* =================== 初始化 =================== */
function applyMode(mode) {
  state.mode = mode;
  const badge = $("modeBadge");
  if (mode === MODE.local) {
    badge.textContent = "本机模式 · 文件就地处理";
    badge.classList.remove("remote");
  } else {
    badge.textContent = "远程模式 · 上传/下载";
    badge.classList.add("remote");
  }
  document.querySelectorAll(".output-block").forEach(b => b.style.display = mode === MODE.local ? "" : "none");
}
function bindTabs() {
  document.querySelectorAll(".tab").forEach(t => {
    t.onclick = () => {
      document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
      document.querySelectorAll(".panel").forEach(x => x.classList.remove("active"));
      t.classList.add("active");
      $("panel-" + t.dataset.tab).classList.add("active");
    };
  });
}
function bindControls() {
  // 合并排序模式切换
  $("merge-sort").onchange = () => {
    const v = $("merge-sort").value;
    $("merge-folder-row").hidden = v === "specified";
    const fileBlock = $("panel-merge").children[0];
    fileBlock.style.display = v === "specified" ? "" : "none";
  };
  // 拆分模式切换：两个输入行始终显示，未选中模式的输入框置灰禁用（不可填写）
  const updateSplitModeUI = () => {
    const v = $("split-mode").value;
    const rangeOn = v === "range";
    const extractOn = v === "extract";
    $("split-range").disabled = !rangeOn;
    $("split-extract").disabled = !extractOn;
    $("split-range-row").classList.toggle("row-disabled", !rangeOn);
    $("split-extract-row").classList.toggle("row-disabled", !extractOn);
  };
  $("split-mode").onchange = updateSplitModeUI;
  updateSplitModeUI();
  // insert 基础选择后获取页数提示
  const origAdd = addLocalPaths;
}

/* =================== 用户菜单 / 修改密码 / 用户管理 =================== */
function bindUserMenu() {
  const label = $("userLabel");
  const menu = $("userMenu");
  if (!label || !menu) return;
  // 双击用户名切换菜单（遵循"隐藏式注销"偏好）
  label.addEventListener("dblclick", () => { menu.hidden = !menu.hidden; });
  // 点击页面其他位置收起菜单
  document.addEventListener("click", (e) => {
    if (!menu.hidden && !e.target.closest(".user-area")) menu.hidden = true;
  });
  menu.querySelectorAll("button").forEach(btn => {
    btn.onclick = () => {
      menu.hidden = true;
      const act = btn.dataset.act;
      if (act === "logout") doLogout();
      else if (act === "change-pwd") openPwdModal();
      else if (act === "manage") openUserModal();
    };
  });
}

async function doLogout() {
  try { await fetch("/api/auth/logout", { method: "POST" }); } catch {}
  window.location.href = "/login";
}

function openPwdModal() {
  $("pwd-old").value = "";
  $("pwd-new").value = "";
  $("pwd-new2").value = "";
  $("pwd-result").innerHTML = "";
  $("pwdModal").hidden = false;
}
function closePwdModal() { $("pwdModal").hidden = true; }

async function submitPwd() {
  const oldp = $("pwd-old").value;
  const newp = $("pwd-new").value;
  const newp2 = $("pwd-new2").value;
  const result = $("pwd-result");
  if (!oldp || !newp) { result.innerHTML = `<div class="alert alert-error">请填写完整</div>`; return; }
  if (newp.length < 6) { result.innerHTML = `<div class="alert alert-error">新密码至少 6 位</div>`; return; }
  if (newp !== newp2) { result.innerHTML = `<div class="alert alert-error">两次新密码不一致</div>`; return; }
  const d = await apiPost("/api/auth/change_password", { old_password: oldp, new_password: newp });
  if (d.error) { result.innerHTML = `<div class="alert alert-error">${esc(d.error)}</div>`; return; }
  result.innerHTML = `<div class="alert alert-success">密码修改成功</div>`;
  setTimeout(closePwdModal, 1200);
}

async function openUserModal() {
  $("userModal").hidden = false;
  await loadUsers();
}
async function loadUsers() {
  const r = await fetch("/api/auth/users");
  if (_handle401(r)) return;
  const d = await r.json();
  if (d.error) { toast(d.error); return; }
  const ul = $("userList");
  ul.innerHTML = "";
  for (const u of d.users || []) {
    const li = document.createElement("li");
    li.className = "user-item";
    li.innerHTML = `
      <span class="name">${esc(u.username)}</span>
      <span class="role-badge ${u.role === "admin" ? "admin" : "user"}" data-user="${esc(u.username)}">${u.role}</span>
      <span class="created">${esc(u.created_at || "")}</span>
      <button class="btn btn-sm" data-act="reset" data-user="${esc(u.username)}">重置密码</button>
      <button class="btn btn-sm btn-danger" data-act="remove" data-user="${esc(u.username)}">删除</button>`;
    ul.appendChild(li);
  }
  // 角色切换
  ul.querySelectorAll(".role-badge").forEach(el => {
    el.onclick = async () => {
      const user = el.dataset.user;
      const newRole = el.textContent.trim() === "admin" ? "user" : "admin";
      const d = await apiPost("/api/auth/role", { username: user, role: newRole });
      if (d.error) { toast(d.error); return; }
      toast(`${user} 角色已改为 ${newRole}`);
      loadUsers();
    };
  });
  // 重置密码 / 删除
  ul.querySelectorAll("button[data-act]").forEach(btn => {
    btn.onclick = async () => {
      const user = btn.dataset.user;
      if (btn.dataset.act === "remove") {
        if (!confirm(`确认删除用户 ${user}？`)) return;
        const d = await apiPost("/api/auth/remove_user", { username: user });
        if (d.error) { toast(d.error); return; }
        toast(`已删除 ${user}`);
        loadUsers();
      } else {
        const np = prompt(`为 ${user} 设置新密码（至少 6 位）`);
        if (!np) return;
        if (np.length < 6) { toast("密码至少 6 位"); return; }
        const d = await apiPost("/api/auth/reset_password", { username: user, new_password: np });
        if (d.error) { toast(d.error); return; }
        toast(`${user} 密码已重置`);
      }
    };
  });
}

// 绑定修改密码弹窗按钮
document.addEventListener("DOMContentLoaded", () => {
  const pm = $("pwdModal");
  if (!pm) return;
  $("pwdModal-close").onclick = closePwdModal;
  $("pwdModal-cancel").onclick = closePwdModal;
  $("pwdModal-ok").onclick = submitPwd;
  const um = $("userModal");
  if (um) {
    $("userModal-close").onclick = () => { um.hidden = true; };
    $("userModal-close2").onclick = () => { um.hidden = true; };
  }
});

async function init() {
  bindTabs();
  // 绑定各面板
  bindPanel("merge", { fileOpts: { filter: "pdf", multi: true, target: "items" }, launch: launchMerge });
  bindPanel("split", { fileOpts: { filter: "pdf", multi: false, target: "single" }, launch: launchSplit });
  bindPanel("convert", { fileOpts: { filter: "pdf", multi: false, target: "single" }, launch: launchConvert });
  bindPanel("compose", { fileOpts: { filter: "image", multi: true, target: "items" }, launch: launchCompose });
  bindPanel("append", { fileOpts: { filter: "all", multi: true, target: "items" }, launch: launchAppend });
  bindPanel("insert", { fileOpts: { filter: "all", multi: true, target: "items" }, launch: launchInsert });
  bindControls();
  bindUserMenu();

  // 获取模式（/api/status 是 GET，不能用 apiPost）
  try {
    const r = await fetch("/api/status");
    if (_handle401(r)) return;
    const d = await r.json();
    applyMode(d.is_local ? MODE.local : MODE.remote);
  } catch {
    applyMode(MODE.remote);
  }
}
document.addEventListener("DOMContentLoaded", init);

/* insert 基础 PDF 选择后刷新页数提示（覆盖单文件渲染） */
const _origRenderSingle = renderSingle;
renderSingle = function (panelId, item, suffix) {
  _origRenderSingle(panelId, item, suffix);
  if (panelId === "insert" && suffix === "-base" && item && state.mode === MODE.local) {
    fetchLocalPages(item.path).then((n) => {
      $("insert-page-hint").textContent = `共 ${n} 页，可填 1 到 ${n + 1}`;
    });
  } else if (panelId === "insert" && suffix === "-base") {
    $("insert-page-hint").textContent = "1 = 最前";
  }
};
