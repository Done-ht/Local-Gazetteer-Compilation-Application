/* ============================================================
   server-paddle OCR 前端逻辑
   功能：文件选择→待提交清单→批次提交、轮询任务状态、进度更新、结果下载
   流程：选文件加入清单（可增删）→ 设置格式/预约时间 → 提交任务（一个批次）
   ============================================================ */
(function () {
    'use strict';

    // ---------------- 登录鉴权（多用户模式） ----------------
    // 统一为所有 fetch 请求附加会话 token；收到 401 时跳转登录页
    const AUTH_TOKEN_KEY = 'token';

    function getAuthHeaders() {
        const t = localStorage.getItem(AUTH_TOKEN_KEY);
        return t ? { 'X-Access-Token': t } : {};
    }

    function isAuthEndpoint(url) {
        const s = String(url);
        return s.indexOf('/api/login') >= 0 || s.indexOf('/api/register') >= 0;
    }

    const _origFetch = window.fetch;
    window.fetch = function (url, opts) {
        opts = opts || {};
        opts.headers = Object.assign({}, opts.headers || {}, getAuthHeaders());
        return _origFetch(url, opts).then(function (res) {
            if (res.status === 401 && !isAuthEndpoint(url)) {
                localStorage.removeItem(AUTH_TOKEN_KEY);
                location.href = '/login';
                throw new Error('unauthorized');
            }
            return res;
        });
    };

    // 页面加载：未登录跳登录页；已登录显示用户名/管理员入口/退出按钮
    (async function checkLogin() {
        try {
            const res = await _origFetch('/api/me', { headers: getAuthHeaders() });
            if (res.status === 401) { location.href = '/login'; return; }
            if (res.ok) {
                const data = await res.json();
                const u = data.user || {};
                const who = document.getElementById('userWho');
                const adminLink = document.getElementById('adminLink');
                const logoutBtn = document.getElementById('logoutBtn');
                if (who) who.textContent = (u.username || '') + (u.is_admin ? '（管理员）' : '');
                if (adminLink) adminLink.style.display = u.is_admin ? '' : 'none';
                if (logoutBtn) logoutBtn.style.display = '';
                if (logoutBtn) {
                    logoutBtn.addEventListener('click', async function () {
                        try {
                            await _origFetch('/api/logout', { method: 'POST', headers: getAuthHeaders() });
                        } catch (e) { /* 忽略 */ }
                        localStorage.removeItem(AUTH_TOKEN_KEY);
                        location.href = '/login';
                    });
                }
            }
        } catch (e) { /* 网络错误忽略，页面其余逻辑继续 */ }
    })();

    // ---------------- DOM 元素 ----------------
    const $ = (id) => document.getElementById(id);
    const dropzone = $('dropzone');
    const fileInput = $('fileInput');
    const folderInput = $('folderInput');
    const folderBtn = $('folderBtn');
    const draftInput = $('draftInput');
    const draftBtn = $('draftBtn');
    // 清单区
    const cartList = $('cartList');
    const cartEmpty = $('cartEmpty');
    const cartCount = $('cartCount');
    const cartTip = $('cartTip');
    const clearCartBtn = $('clearCartBtn');
    const submitTaskBtn = $('submitTaskBtn');
    const scheduleInput = $('scheduleInput');
    // 任务列表
    const taskList = $('taskList');
    const taskEmpty = $('taskEmpty');
    const clearDoneBtn = $('clearDoneBtn');
    const downloadZipBtn = $('downloadZipBtn');
    const logBox = $('logBox');
    const clearLogBtn = $('clearLogBtn');
    // 状态栏
    const statusPill = $('statusPill');
    const statusDot = $('statusDot');
    const statusText = $('statusText');
    const statusDetail = $('statusDetail');
    const engineName = $('engineName');
    const concInfo = $('concInfo');
    const queueInfo = $('queueInfo');
    const lanUrlEl = $('lanUrl');
    const lanUrlBig = $('lanUrlBig');
    const qrImg = $('qrImg');
    const heroCard = $('heroCard');

    // 设置区
const cfgDpi = $('cfgDpi');
const cfgConc = $('cfgConc');
const cfgLayer2 = $('cfgLayer2');
    const cfgLayout = $('cfgLayout');
    const cfgTableRec = $('cfgTableRec');
    const cfgTier = $('cfgTier');
    const saveConfigBtn = $('saveConfigBtn');
    const saveHint = $('saveHint');

    // 支持的文件扩展名
    const SUPPORTED_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.pdf', '.docx'];

    // ---------------- 状态 ----------------
    const state = {
        cart: [],             // 待提交清单：[{ name, size, file }]
        tasks: new Map(),     // task_id -> task dom 元素
        pollingIds: new Set(),// 正在轮询的 task_id
        lanUrl: '',
        qrLoaded: false,
        qrFailed: false,
        batches: new Map(),   // batch_id -> { batchId, batchNo, element, taskIds: Set }
        batchCounter: 0,      // 批次自增编号（任务01、任务02...）
        // 上传限制配置（从 /api/limits 获取）
        limits: {
            max_files_per_batch: 20,
            max_file_size_mb: 500,
            max_batch_size_mb: 2048,
            max_pending_tasks: 50,
            max_scheduled_tasks: 20,
            current_pending: 0,
            current_scheduled: 0,
        },
    };

    // ---------------- 跨标签页通信 ----------------
    // 通过 BroadcastChannel 通知其他标签页任务的增删变化。
    // 收到通知的标签页立即调 syncTasks 同步列表，无需等 visibilitychange。
    // 场景：标签页A删除任务 → 标签页B立即收到通知刷新列表，不再需要手动刷新。
    const _taskChannel = (typeof BroadcastChannel !== 'undefined')
        ? new BroadcastChannel('ocr-tasks') : null;
    if (_taskChannel) {
        _taskChannel.onmessage = (ev) => {
            const msg = ev.data || {};
            if (msg.type === 'delete' || msg.type === 'create' || msg.type === 'change') {
                syncTasks();
            }
        };
    }
    function _broadcastTaskChange(type, taskId) {
        if (_taskChannel) {
            try { _taskChannel.postMessage({ type: type, taskId: taskId }); } catch (e) {}
        }
    }

    // ---------------- Toast 提示（页面顶部显眼提醒） ----------------
    function showToast(msg, level) {
        level = level || 'info';
        let host = document.getElementById('toastHost');
        if (!host) {
            host = document.createElement('div');
            host.id = 'toastHost';
            host.className = 'toast-host';
            document.body.appendChild(host);
        }
        const t = document.createElement('div');
        t.className = 'toast toast-' + level;
        t.innerHTML = '<span class="toast-icon">' +
            (level === 'success' ? '✓' : level === 'error' ? '✗' : level === 'warn' ? '!' : 'i') +
            '</span><span class="toast-msg">' + escapeHtml(msg) + '</span>';
        host.appendChild(t);
        // 入场动画
        requestAnimationFrame(() => t.classList.add('show'));
        // 自动消失
        const duration = level === 'error' ? 5000 : 3000;
        setTimeout(() => {
            t.classList.remove('show');
            setTimeout(() => { if (t.parentNode) t.parentNode.removeChild(t); }, 300);
        }, duration);
    }

    // ---------------- 日志 ----------------
    // 日志防抖：连续相同消息在 2 秒内只显示一次，避免后端高频重复日志刷屏导致页面卡顿
    const _logDedup = new Map(); // msg -> lastTime
    function log(msg, level) {
        level = level || 'info';
        const key = level + ':' + msg;
        const now = Date.now();
        const last = _logDedup.get(key);
        if (last && now - last < 2000) return;
        _logDedup.set(key, now);
        // 清理过期的去重记录（避免 Map 无限增长）
        if (_logDedup.size > 200) {
            for (const [k, t] of _logDedup) {
                if (now - t > 5000) _logDedup.delete(k);
            }
        }
        const time = new Date().toLocaleTimeString();
        const line = document.createElement('div');
        line.className = 'log-line log-' + level;
        line.innerHTML = '<span class="log-time">[' + time + ']</span> ' + escapeHtml(msg);
        logBox.appendChild(line);
        logBox.scrollTop = logBox.scrollHeight;
        // 限制日志条数
        while (logBox.children.length > 300) {
            logBox.removeChild(logBox.firstChild);
        }
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, (c) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[c]));
    }

    // ---------------- 状态栏 ----------------
    function setStatus(ok, text) {
        statusPill.classList.remove('ok', 'err');
        if (ok === true) statusPill.classList.add('ok');
        if (ok === false) statusPill.classList.add('err');
        statusText.textContent = text;
    }

    // ---------------- 服务状态轮询 ----------------
    async function fetchStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            setStatus(true, '服务正常');
            engineName.textContent = data.engine.name +
                (data.engine.version ? ' v' + data.engine.version : '');
            const c = data.concurrency;
            concInfo.textContent = c.running + '/' + c.max_concurrent;
            queueInfo.textContent = c.queued;
            const newUrl = data.lan_url || '';
            const urlChanged = newUrl !== state.lanUrl;
            state.lanUrl = newUrl;
            lanUrlEl.textContent = newUrl || '本机';
            if (newUrl) {
                lanUrlBig.textContent = newUrl;
                // 只在 URL 变化或首次加载时才更新二维码 src，避免每 2 秒重载图片造成"刷新"感
                // 二维码加载失败后也不再重试（qrcode 库未装等场景）
                if (urlChanged && !state.qrFailed) {
                    state.qrLoaded = false;
                    qrImg.style.display = 'inline';
                    // <img> 标签无法携带请求头，改用带 token 的 fetch 获取二维码，
                    // 再以 objectURL 赋值给 img（否则 /api/qr 恒 401，二维码不显示）
                    fetch('/api/qr?' + encodeURIComponent(newUrl))
                        .then(function (r) {
                            if (!r.ok) throw new Error('qr ' + r.status);
                            return r.blob();
                        })
                        .then(function (blob) {
                            if (state.qrFailed) return;
                            if (qrImg.src && qrImg.src.indexOf('blob:') === 0) {
                                URL.revokeObjectURL(qrImg.src);
                            }
                            qrImg.src = URL.createObjectURL(blob);
                        })
                        .catch(function () {
                            state.qrFailed = true;
                            qrImg.style.display = 'none';
                        });
                }
            } else {
                lanUrlBig.textContent = 'http://127.0.0.1:8070';
                qrImg.style.display = 'none';
            }
        } catch (e) {
            setStatus(false, '服务连接失败');
            log('获取状态失败: ' + e.message, 'error');
        }
    }

    // ---------------- 获取上传限制 ----------------
    async function fetchLimits() {
        try {
            const res = await fetch('/api/limits');
            if (!res.ok) return;
            const data = await res.json();
            state.limits = data;
            // 更新可用槽位显示
            updateSlotsInfo();
        } catch (e) {
            // 获取失败保持默认值，不阻塞使用
        }
    }

    // ---------------- 更新可用槽位显示 ----------------
    function updateSlotsInfo() {
        const el = document.getElementById('slotsInfo');
        if (!el) return;
        const L = state.limits;
        if (!L || L.max_concurrent === undefined) {
            el.textContent = '';
            return;
        }
        const max = L.max_concurrent;
        const avail = L.available_slots !== undefined ? L.available_slots : max;
        const tcSelect = document.getElementById('taskConcurrencySelect');
        const tc = tcSelect ? parseInt(tcSelect.value, 10) || 1 : 1;
        if (tc > avail) {
            el.style.color = '#d1242f';
            el.textContent = `⚠ 当前可用 ${avail}/${max} 槽位，选择 ${tc} 进程将排队等待`;
        } else {
            el.style.color = '#0969da';
            el.textContent = `可用 ${avail}/${max} 槽位` + (tc > 1 ? `，本任务占用 ${tc} 个` : '');
        }
    }

    // ---------------- 提交前预检 ----------------
    // 返回错误消息字符串数组，空数组表示通过
    function validateCart(isScheduled) {
        const errors = [];
        const L = state.limits;
        const n = state.cart.length;
        const totalSize = state.cart.reduce((s, c) => s + c.size, 0);
        // 1. 单批次文件数
        if (n > L.max_files_per_batch) {
            errors.push('文件数 ' + n + ' 超过单批次上限 ' + L.max_files_per_batch + '，请分多次提交');
        }
        // 2. 单文件大小
        const maxFileBytes = L.max_file_size_mb * 1024 * 1024;
        const oversize = state.cart.filter(c => c.size > maxFileBytes);
        if (oversize.length > 0) {
            errors.push('以下文件超过单文件上限 ' + L.max_file_size_mb + ' MB：' +
                oversize.map(c => c.name + ' (' + formatSize(c.size) + ')').join('、'));
        }
        // 3. 批次总大小
        if (totalSize > L.max_batch_size_mb * 1024 * 1024) {
            errors.push('批次总大小 ' + formatSize(totalSize) + ' 超过上限 ' +
                L.max_batch_size_mb + ' MB，请分批提交');
        }
        // 4. 待处理任务总数
        if (L.current_pending + n > L.max_pending_tasks) {
            errors.push('当前待处理任务 ' + L.current_pending + ' 个 + 本次 ' + n +
                ' 个将超过上限 ' + L.max_pending_tasks + '，请等待部分任务完成');
        }
        // 5. 预约任务数（仅预约提交时检查）
        if (isScheduled && L.current_scheduled + n > L.max_scheduled_tasks) {
            errors.push('当前预约任务 ' + L.current_scheduled + ' 个 + 本次 ' + n +
                ' 个将超过上限 ' + L.max_scheduled_tasks + '，请减少预约任务');
        }
        return errors;
    }

    // ---------------- 文件选择 → 加入清单 ----------------
    // 文件选择后不直接上传，而是加入"待提交清单"，支持多次添加、去重、删除
    function addToCart(files) {
        const arr = Array.from(files || []);
        // 过滤支持的扩展名
        const valid = arr.filter(f => {
            const dotIdx = f.name.lastIndexOf('.');
            if (dotIdx < 0) return false;
            const ext = f.name.slice(dotIdx).toLowerCase();
            return SUPPORTED_EXTS.indexOf(ext) >= 0;
        });
        const rejected = arr.length - valid.length;
        if (rejected > 0) log('已忽略 ' + rejected + ' 个不支持的文件', 'warn');
        if (valid.length === 0) return;
        let added = 0, duplicated = 0;
        valid.forEach(f => {
            // 去重：同名 + 同大小 + 同修改时间视为同一文件
            const exists = state.cart.some(c =>
                c.name === f.name && c.size === f.size && c.file.lastModified === f.lastModified
            );
            if (exists) { duplicated++; return; }
            state.cart.push({ name: f.name, size: f.size, file: f });
            added++;
        });
        if (added > 0) {
            log('已加入清单 ' + added + ' 个文件' +
                (duplicated > 0 ? '（跳过 ' + duplicated + ' 个重复）' : ''));
        } else if (duplicated > 0) {
            log('清单中已存在这 ' + duplicated + ' 个文件', 'warn');
        }
        renderCart();
    }

    // 从清单移除指定项
    function removeFromCart(index) {
        const item = state.cart[index];
        if (!item) return;
        state.cart.splice(index, 1);
        renderCart();
        log('已从清单移除: ' + item.name);
    }

    // 清空清单
    function clearCart() {
        if (state.cart.length === 0) return;
        const n = state.cart.length;
        state.cart = [];
        renderCart();
        log('已清空清单（' + n + ' 个文件）');
    }

    // 格式化文件大小
    function formatSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1024 / 1024).toFixed(2) + ' MB';
    }

    // 渲染待提交清单
    function renderCart() {
        const n = state.cart.length;
        cartCount.textContent = n + ' 个文件';
        // 清空列表（含超限警告）
        cartList.innerHTML = '';
        if (n === 0) {
            cartList.appendChild(cartEmpty);
            cartEmpty.style.display = '';
            submitTaskBtn.disabled = true;
            clearCartBtn.disabled = true;
            updateCartTip();
            return;
        }
        clearCartBtn.disabled = false;
        submitTaskBtn.disabled = false;
        // 渲染每个文件条目
        const totalSize = state.cart.reduce((s, c) => s + c.size, 0);
        state.cart.forEach((item, idx) => {
            const row = document.createElement('div');
            row.className = 'cart-row';
            row.innerHTML =
                '<span class="cart-row-name" title="' + escapeHtml(item.name) + '">' +
                    escapeHtml(item.name) +
                '</span>' +
                '<span class="cart-row-size">' + formatSize(item.size) + '</span>' +
                '<button class="cart-row-del" title="移除" data-idx="' + idx + '">×</button>';
            row.querySelector('.cart-row-del').addEventListener('click', (e) => {
                e.stopPropagation();
                removeFromCart(idx);
            });
            cartList.appendChild(row);
        });
        // 汇总行
        const summary = document.createElement('div');
        summary.className = 'cart-summary';
        summary.textContent = '共 ' + n + ' 个文件，合计 ' + formatSize(totalSize);
        cartList.appendChild(summary);
        updateCartTip();
    }

    // 更新清单提示（显示预约信息）
    function updateCartTip() {
        const schedTime = scheduleInput.value;
        if (state.cart.length === 0) {
            cartTip.textContent = '';
            return;
        }
        if (schedTime) {
            cartTip.textContent = '⏰ 将预约于 ' + schedTime + ' 执行（错峰），点击"提交任务"创建批次';
        } else {
            cartTip.textContent = '点击"提交任务"将清单作为批次立即提交识别';
        }
    }

    // 预约时间变化时更新提示
    scheduleInput.addEventListener('change', updateCartTip);
    scheduleInput.addEventListener('input', updateCartTip);

    // 任务并发数变化时更新槽位显示
    const tcSelect = document.getElementById('taskConcurrencySelect');
    if (tcSelect) {
        tcSelect.addEventListener('change', () => {
            updateSlotsInfo();
            updateCartTip();
        });
    }

    // ---------------- 拖拽 / 选择 ----------------
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInput.click();
        }
    });
    fileInput.addEventListener('change', (e) => {
        addToCart(e.target.files);
        // 清空 input.value 允许重复选择同一文件
        e.target.value = '';
    });

    ['dragenter', 'dragover'].forEach(ev => {
        dropzone.addEventListener(ev, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });
    ['dragleave', 'drop'].forEach(ev => {
        dropzone.addEventListener(ev, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });
    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer && e.dataTransfer.files;
        if (files && files.length) addToCart(files);
    });

    // ---------------- 清空清单 ----------------
    clearCartBtn.addEventListener('click', clearCart);

    // ---------------- 提交任务（清单作为批次上传） ----------------
    submitTaskBtn.addEventListener('click', () => {
        if (state.cart.length === 0) return;
        const isScheduled = !!scheduleInput.value;
        // 提交前预检：刷新限制数据后校验
        // 先同步拉取最新限制（确保 current_pending 等实时数据准确）
        fetchLimits().then(() => {
            const errors = validateCart(isScheduled);
            if (errors.length > 0) {
                // 超限：显示错误提示，不提交
                errors.forEach(e => log(e, 'error'));
                showToast(errors[0], 'error');
                // 在清单区显示详细的超限提示
                showCartWarning(errors);
                return;
            }
            doSubmit(isScheduled);
        });
    });

    // 实际提交逻辑（预检通过后执行）
    function doSubmit(isScheduled) {
        const schedTime = scheduleInput.value;
        const fileCount = state.cart.length;
        const fileNames = state.cart.map(c => c.name);
        submitTaskBtn.disabled = true;
        clearCartBtn.disabled = true;
        const sorted = state.cart.slice().sort((a, b) => a.name.localeCompare(b.name, 'zh'));
        log('提交任务：' + fileCount + ' 个文件' +
            (isScheduled ? '（预约 ' + schedTime + ' 执行）' : '（立即执行）'));

        // 立即创建临时批次和 uploading 状态的任务条目
        // 这样用户提交后马上能在任务列表看到，不用等 HTTP 传完
        const tempBatchId = 'temp_' + Date.now();
        const tempTaskIds = [];
        sorted.forEach((item, idx) => {
            const tempId = 'temp_' + Date.now() + '_' + idx;
            tempTaskIds.push({ tempId, name: item.name });
            addTaskItem(tempId, item.name, 'uploading', 0,
                isScheduled ? '预约于 ' + schedTime + ' 执行' : '', tempBatchId);
        });

        // 立即清空清单（任务已在列表中，清单可以清了）
        state.cart = [];
        renderCart();
        scheduleInput.value = '';
        updateCartTip();
        submitTaskBtn.disabled = false;
        clearCartBtn.disabled = true;

        // 构建 FormData
        // 注意：schedule_time/task_concurrency 必须作为 FormData 字段发送，
        // 后端用 Form(None) 注解接收（File() 路由下 Query 参数不会被 Form 读取）。
        // 之前用 URLSearchParams 放在 URL query string 里，后端始终收到 None，
        // 导致 task_concurrency 永远为 1（单文件并发功能失效）。
        // 输出格式在识别后选择，提交时不发送。
        const fd = new FormData();
        sorted.forEach(item => fd.append('files', item.file, item.name));
        if (schedTime) fd.append('schedule_time', schedTime);
        // 任务并发数：大于1时PDF按页切分并行处理
        const tcSelect = document.getElementById('taskConcurrencySelect');
        const tc = tcSelect ? parseInt(tcSelect.value, 10) || 1 : 1;
        if (tc > 1) fd.append('task_concurrency', String(tc));
        const url = '/api/upload';

        // 用 XHR 上传以获取上传进度（fetch 无法获取上传进度）
        const xhr = new XMLHttpRequest();
        xhr.open('POST', url);
        // XHR 不走上方 window.fetch 封装，必须手动附带会话 token，否则后端返回 401
        const _uploadToken = localStorage.getItem(AUTH_TOKEN_KEY);
        if (_uploadToken) xhr.setRequestHeader('X-Access-Token', _uploadToken);
        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const pct = Math.round(e.loaded / e.total * 100);
                tempTaskIds.forEach(t => updateUploadProgress(t.tempId, pct));
            }
        };
        xhr.onload = () => {
            // 上传完成，解析响应
            let data;
            try {
                data = JSON.parse(xhr.responseText);
            } catch (e) {
                log('响应解析失败: ' + e.message, 'error');
                tempTaskIds.forEach(t => updateTaskItem(t.tempId, {
                    status: 'error', error: '响应解析失败', message: '上传失败',
                }));
                return;
            }
            if (xhr.status < 200 || xhr.status >= 300) {
                const errMsg = data.detail || '上传失败 (' + xhr.status + ')';
                log('提交失败: ' + errMsg, 'error');
                showToast('提交失败：' + errMsg, 'error');
                tempTaskIds.forEach(t => updateTaskItem(t.tempId, {
                    status: 'error', error: errMsg, message: '上传失败',
                }));
                return;
            }
            // 上传成功：把临时 task_id 替换为真实 task_id
            const realBatchId = data.batch_id;
            const batch = state.batches.get(tempBatchId);
            if (batch) {
                state.batches.delete(tempBatchId);
                batch.batchId = realBatchId;
                batch.element.dataset.batchId = realBatchId;
                state.batches.set(realBatchId, batch);
            }
            let successCount = 0;
            let failCount = 0;
            (data.tasks || []).forEach((t, idx) => {
                const tempEntry = tempTaskIds[idx];
                if (!tempEntry) return;
                const tempId = tempEntry.tempId;
                const item = state.tasks.get(tempId);
                if (t.task_id && item) {
                    // 替换 task_id：更新 DOM、state.tasks、batch.taskIds
                    item.dataset.taskId = t.task_id;
                    state.tasks.delete(tempId);
                    state.tasks.set(t.task_id, item);
                    if (batch) {
                        batch.taskIds.delete(tempId);
                        batch.taskIds.add(t.task_id);
                    }
                    // 更新状态为后端返回的（queued/scheduled）
                    updateTaskItem(t.task_id, {
                        status: t.status,
                        source_name: t.filename,
                        queue_position: t.queue_position,
                        queue_reason: t.queue_reason,
                    });
                    startPolling(t.task_id);
                    log('任务已创建: ' + t.filename + ' (' + t.task_id + ')');
                    successCount++;
                    if (t.queue_position > 0) {
                        log(t.filename + ' 排队中，位置 ' + t.queue_position, 'warn');
                    }
                } else if (t.error && item) {
                    updateTaskItem(tempId, {
                        status: 'error', error: t.error, message: '提交失败: ' + t.error,
                    });
                    log(t.filename + ' 提交失败: ' + t.error, 'error');
                    failCount++;
                }
            });
            // Toast 提醒
            if (successCount > 0 && failCount === 0) {
                const msg = isScheduled
                    ? '已预约 ' + successCount + ' 个文件于 ' + schedTime + ' 执行'
                    : '已提交 ' + successCount + ' 个文件，开始识别：' +
                        fileNames.slice(0, 2).join('、') +
                        (fileNames.length > 2 ? ' 等' : '');
                showToast(msg, isScheduled ? 'info' : 'success');
            } else if (successCount > 0 && failCount > 0) {
                showToast('部分提交成功：' + successCount + ' 成功，' + failCount + ' 失败', 'warn');
            } else if (failCount > 0) {
                showToast('提交失败 ' + failCount + ' 个文件', 'error');
            }
            // 更新批次头部
            if (batch) updateBatchHeader(realBatchId);
            // 提交后刷新限制数据（current_pending 等已变化）
            fetchLimits();
            // 广播给其他标签页：有新任务创建
            if (successCount > 0) {
                _broadcastTaskChange('create', '');
            }
        };
        xhr.onerror = () => {
            log('上传失败：网络错误', 'error');
            showToast('上传失败：网络错误', 'error');
            tempTaskIds.forEach(t => updateTaskItem(t.tempId, {
                status: 'error', error: '网络错误', message: '上传失败',
            }));
        };
        xhr.send(fd);
    }

    // 在清单区显示超限警告
    function showCartWarning(errors) {
        // 移除已有警告
        const old = cartList.querySelector('.cart-warning');
        if (old) old.remove();
        const div = document.createElement('div');
        div.className = 'cart-warning';
        div.innerHTML = '<div class="cart-warning-title">⚠ 无法提交，请先解决以下问题：</div>' +
            errors.map(e => '<div class="cart-warning-item">• ' + escapeHtml(e) + '</div>').join('');
        cartList.appendChild(div);
    }

    // ---------------- 批次管理 ----------------
    // 同一次上传的所有文件共享一个 batch_id，前端按批次分组展示为"任务01/任务02"
    // 每个批次是一个可折叠面板，展开后看到具体文件（独立任务）
    function getOrCreateBatch(batchId) {
        if (!batchId) batchId = '_default';
        if (state.batches.has(batchId)) {
            return state.batches.get(batchId);
        }
        state.batchCounter++;
        const batchNo = String(state.batchCounter).padStart(2, '0');
        const el = document.createElement('div');
        el.className = 'batch-item';
        el.dataset.batchId = batchId;
        el.innerHTML =
            '<div class="batch-header">' +
                '<span class="batch-arrow" title="点击展开/折叠">▼</span>' +
                '<span class="batch-icon">📂</span>' +
                '<span class="batch-title">任务' + batchNo + '</span>' +
                '<span class="batch-count"></span>' +
                '<span class="batch-badge"></span>' +
                '<div class="batch-progress-wrap"><div class="batch-progress-bar"></div></div>' +
                '<button class="btn btn-ghost btn-sm batch-download" disabled>打包下载</button>' +
            '</div>' +
            '<div class="batch-files"></div>';
        // 点击头部切换展开/折叠（按钮区域不触发）
        const header = el.querySelector('.batch-header');
        header.addEventListener('click', (e) => {
            if (e.target.closest('button')) return;
            el.classList.toggle('collapsed');
            const arrow = el.querySelector('.batch-arrow');
            arrow.textContent = el.classList.contains('collapsed') ? '▶' : '▼';
        });
        // 打包下载按钮
        const dlBtn = el.querySelector('.batch-download');
        dlBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            downloadBatch(batchId);
        });
        const batch = {
            batchId: batchId,
            batchNo: batchNo,
            element: el,
            taskIds: new Set(),
        };
        state.batches.set(batchId, batch);
        // 新批次插入到列表顶部（最新在前），便于用户即时查看上传结果
        if (taskEmpty) taskEmpty.style.display = 'none';
        const firstBatch = taskList.querySelector('.batch-item');
        if (firstBatch) {
            taskList.insertBefore(el, firstBatch);
        } else {
            taskList.appendChild(el);
        }
        return batch;
    }

    // 计算批次聚合状态：从批次内所有任务的状态聚合
    // 优先级：running > uploading > queued > scheduled > error 部分完成 > done 全部完成
    function computeBatchStatus(batchId) {
        const batch = state.batches.get(batchId);
        if (!batch || batch.taskIds.size === 0) {
            return { status: 'queued', label: '等待中', doneCount: 0, totalCount: 0, progressPct: 0 };
        }
        let hasRunning = false, hasUploading = false, hasQueued = false, hasScheduled = false, hasError = false, hasDone = false;
        let doneCount = 0, totalCount = 0;
        let sumProgress = 0, sumTotal = 0;
        batch.taskIds.forEach(taskId => {
            const item = state.tasks.get(taskId);
            if (!item) return;
            totalCount++;
            if (item.classList.contains('running')) hasRunning = true;
            else if (item.classList.contains('uploading')) hasUploading = true;
            else if (item.classList.contains('queued')) hasQueued = true;
            else if (item.classList.contains('scheduled')) hasScheduled = true;
            else if (item.classList.contains('done')) { hasDone = true; doneCount++; }
            else if (item.classList.contains('error')) hasError = true;
            // 累计进度（仅 running/uploading/done 任务有有效进度）
            const bar = item.querySelector('.progress-bar');
            if (bar) {
                const w = parseFloat(bar.style.width) || 0;
                sumProgress += w;
                sumTotal += 100;
            }
        });
        let status, label;
        if (hasRunning) {
            status = 'running';
            label = '进行中 ' + doneCount + '/' + totalCount;
        } else if (hasUploading) {
            status = 'uploading';
            label = '上传中';
        } else if (hasQueued) {
            status = 'queued';
            label = doneCount > 0 ? '等待中 ' + doneCount + '/' + totalCount : '等待中';
        } else if (hasScheduled) {
            status = 'scheduled';
            label = doneCount > 0 ? '已预约 ' + doneCount + '/' + totalCount : '已预约';
        } else if (hasError && hasDone) {
            status = 'error';
            label = '部分失败 ' + doneCount + '/' + totalCount;
        } else if (hasError) {
            status = 'error';
            label = '失败';
        } else if (hasDone) {
            status = 'done';
            label = '已完成 ' + doneCount + '/' + totalCount;
        } else {
            status = 'queued';
            label = '等待中';
        }
        const progressPct = sumTotal > 0 ? Math.round(sumProgress / sumTotal * 100) : 0;
        return { status, label, doneCount, totalCount, progressPct };
    }

    // 更新批次头部显示：状态徽章、文件数、进度条、打包下载按钮
    function updateBatchHeader(batchId) {
        const batch = state.batches.get(batchId);
        if (!batch) return;
        const info = computeBatchStatus(batchId);
        // 更新状态类名时保留 collapsed 折叠状态
        // 避免轮询更新（预约任务长时间停留在 scheduled）反复撤销用户折叠
        const wasCollapsed = batch.element.classList.contains('collapsed');
        batch.element.className = 'batch-item ' + info.status;
        if (wasCollapsed) batch.element.classList.add('collapsed');
        const badgeMap = {
            uploading: ['上传中', 'badge-uploading'],
            queued: ['等待中', 'badge-queued'],
            scheduled: ['已预约', 'badge-scheduled'],
            running: ['进行中', 'badge-running'],
            paused: ['已暂停', 'badge-paused'],
            done: ['已完成', 'badge-done'],
            error: ['部分失败', 'badge-error'],
        };
        const bd = badgeMap[info.status] || [info.label, 'badge-queued'];
        const badge = batch.element.querySelector('.batch-badge');
        badge.textContent = info.label;
        badge.className = 'batch-badge ' + bd[1];
        batch.element.querySelector('.batch-count').textContent = info.totalCount + ' 个文件';
        // 整体进度条
        const bar = batch.element.querySelector('.batch-progress-bar');
        if (info.status === 'running' || info.status === 'uploading') {
            bar.style.width = info.progressPct + '%';
            bar.classList.remove('indeterminate');
        } else if (info.status === 'done') {
            bar.style.width = '100%';
            bar.classList.remove('indeterminate');
        } else if (info.status === 'queued' && info.doneCount > 0) {
            // 部分完成排队中：显示已完成比例
            bar.style.width = info.progressPct + '%';
            bar.classList.remove('indeterminate');
        } else {
            bar.style.width = '0%';
            bar.classList.remove('indeterminate');
        }
        // 打包下载按钮：有已完成任务时启用
        const dlBtn = batch.element.querySelector('.batch-download');
        dlBtn.disabled = info.doneCount === 0;
        dlBtn.textContent = info.doneCount > 0
            ? '打包下载 (' + info.doneCount + ')'
            : '打包下载';
    }

    // ---------------- 鉴权下载工具 ----------------
    // 下载接口需要登录鉴权，而 <a> 导航 / window.open 无法携带 token（会 401 卡住），
    // 统一用带 token 的 fetch 拉取 blob，再触发浏览器下载。
    function downloadViaFetch(url, fallbackName) {
        return fetch(url)
            .then(function (res) {
                if (!res.ok) {
                    // 401 时 fetch 封装已清理 token 并跳转登录页，此处只提示其他错误
                    return res.json().then(function (d) {
                        showToast('下载失败：' + (d.detail || res.status), 'error');
                    }).catch(function () {
                        showToast('下载失败：' + res.status, 'error');
                    });
                }
                return res.blob().then(function (blob) {
                    // 从 Content-Disposition 解析文件名（优先 filename*=UTF-8''）
                    let name = fallbackName || 'download';
                    const cd = res.headers.get('Content-Disposition') || '';
                    const m1 = cd.match(/filename\*=UTF-8''([^;]+)/i);
                    if (m1) {
                        try { name = decodeURIComponent(m1[1]); } catch (e) { name = m1[1]; }
                    } else {
                        const m2 = cd.match(/filename="?([^";]+)"?/i);
                        if (m2 && m2[1]) name = m2[1];
                    }
                    const objUrl = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = objUrl;
                    a.download = name;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    setTimeout(function () { URL.revokeObjectURL(objUrl); }, 1000);
                });
            })
            .catch(function (e) {
                if (e && e.message === 'unauthorized') return; // fetch 封装已跳登录页
                showToast('下载失败：' + ((e && e.message) || '网络错误'), 'error');
            });
    }

    // 整批打包下载
    function downloadBatch(batchId) {
        const batch = state.batches.get(batchId);
        if (!batch) return;
        const doneIds = [];
        batch.taskIds.forEach(taskId => {
            const item = state.tasks.get(taskId);
            if (item && item.classList.contains('done')) doneIds.push(taskId);
        });
        if (doneIds.length === 0) {
            showToast('该批次没有已完成的文件', 'warn');
            return;
        }
        const url = '/api/download_zip?task_ids=' + encodeURIComponent(doneIds.join(','));
        log('打包下载任务' + batch.batchNo + ' 的 ' + doneIds.length + ' 个文件');
        showToast('正在打包任务' + batch.batchNo + ' 的 ' + doneIds.length + ' 个文件...', 'info');
        downloadViaFetch(url, 'OCR结果.zip');
    }

    // ---------------- 任务列表渲染 ----------------
    function addTaskItem(taskId, filename, status, queuePos, queueReason, batchId) {
        if (state.tasks.has(taskId)) return;
        const batch = getOrCreateBatch(batchId);
        batch.taskIds.add(taskId);
        const item = document.createElement('div');
        item.className = 'task-item ' + (status || 'queued');
        item.dataset.taskId = taskId;
        item.dataset.batchId = batch.batchId;
        item.dataset.status = status || 'queued';
        item.innerHTML =
            '<div class="task-head">' +
                '<span class="task-name">' + escapeHtml(filename) + '</span>' +
                '<span class="task-concurrency-badge" style="display:none"></span>' +
                '<span class="task-badge"></span>' +
            '</div>' +
            '<div class="task-msg"></div>' +
            '<div class="task-queue-reason" style="display:none"></div>' +
            '<div class="progress-wrap"><div class="progress-bar"></div></div>' +
            '<div class="task-actions"></div>' +
            '<details class="task-result" style="display:none">' +
                '<summary>查看识别文字</summary>' +
                '<div class="task-result-text"></div>' +
            '</details>';
        batch.element.querySelector('.batch-files').appendChild(item);
        state.tasks.set(taskId, item);
        updateTaskItem(taskId, {
            status: status, source_name: filename,
            queue_position: queuePos, queue_reason: queueReason
        });
    }

    // 更新上传进度（XHR upload.onprogress 回调）
    function updateUploadProgress(taskId, pct) {
        const item = state.tasks.get(taskId);
        if (!item) return;
        const bar = item.querySelector('.progress-bar');
        bar.style.width = pct + '%';
        bar.classList.remove('indeterminate');
        const msg = item.querySelector('.task-msg');
        msg.textContent = '上传中 ' + pct + '%';
    }

    function updateTaskItem(taskId, info) {
        const item = state.tasks.get(taskId);
        if (!item) return;
        const status = info.status || 'queued';
        const prevStatus = item.dataset.lastStatus || '';
        item.className = 'task-item ' + status;
        item.dataset.status = status;
        const badge = item.querySelector('.task-badge');
        const badgeMap = {
            uploading: ['上传中', 'badge-uploading'],
            scheduled: ['已预约', 'badge-scheduled'],
            queued: ['排队中', 'badge-queued'],
            running: ['处理中', 'badge-running'],
            paused: ['已暂停', 'badge-paused'],
            done: ['完成', 'badge-done'],
            error: ['失败', 'badge-error'],
        };
        const bd = badgeMap[status] || ['未知', 'badge-queued'];
        badge.textContent = bd[0];
        badge.className = 'task-badge ' + bd[1];

        // 进程数标识（仅 running 显示，便于核对并发数是否正确传递）
        const tcBadge = item.querySelector('.task-concurrency-badge');
        const tc = info.task_concurrency || 1;
        if (status === 'running' && tc > 1) {
            tcBadge.textContent = tc + '进程';
            tcBadge.className = 'task-concurrency-badge';
            tcBadge.style.display = 'inline-block';
        } else {
            tcBadge.style.display = 'none';
        }

        // 消息
        // 优先使用新的页级进度字段（completed_pages / total_pages）
        // 旧的 progress / total 字段作为回退
        const completedPages = (typeof info.completed_pages === 'number') ? info.completed_pages : null;
        const totalPages = (typeof info.total_pages === 'number' && info.total_pages > 0) ? info.total_pages : null;
        const msg = item.querySelector('.task-msg');
        if (totalPages !== null && completedPages !== null &&
            (status === 'running' || status === 'paused')) {
            // 新进度方式：显示已识别页数
            msg.textContent = '已识别 ' + completedPages + '/' + totalPages + ' 页';
        } else {
            msg.textContent = info.message || '';
        }

        // 排队/预约原因
        const qr = item.querySelector('.task-queue-reason');
        if ((status === 'queued' || status === 'scheduled') && info.queue_reason) {
            qr.textContent = info.queue_reason;
            qr.style.display = 'block';
        } else {
            qr.style.display = 'none';
        }

        // 进度条
        const bar = item.querySelector('.progress-bar');
        if (status === 'uploading') {
            // 上传进度由 updateUploadProgress 单独更新，这里只确保样式正确
            bar.classList.remove('indeterminate');
        } else if (status === 'running') {
            if (totalPages !== null && completedPages !== null) {
                // 新进度方式：基于 ocr_pages 已完成页数计算
                const pct = Math.min(100, Math.round(completedPages / totalPages * 100));
                bar.style.width = pct + '%';
                bar.classList.remove('indeterminate');
            } else if (info.total && info.total > 0 && info.progress >= 0) {
                // 旧进度方式回退：progress/total
                const pct = Math.min(100, Math.round(info.progress / info.total * 100));
                bar.style.width = pct + '%';
                bar.classList.remove('indeterminate');
            } else {
                bar.classList.add('indeterminate');
            }
        } else if (status === 'done') {
            bar.style.width = '100%';
            bar.classList.remove('indeterminate');
        } else if (status === 'paused') {
            // 暂停时保留当前进度（不动画），用户能看到已处理比例
            if (totalPages !== null && completedPages !== null) {
                const pct = Math.min(100, Math.round(completedPages / totalPages * 100));
                bar.style.width = pct + '%';
            } else if (info.total && info.total > 0 && info.progress >= 0) {
                const pct = Math.min(100, Math.round(info.progress / info.total * 100));
                bar.style.width = pct + '%';
            }
            bar.classList.remove('indeterminate');
        } else if (status === 'error') {
            bar.style.width = '0%';
            bar.classList.remove('indeterminate');
        }

        // 仅在状态变化或关键派生状态变化时才重建 actions 区域，
        // 避免轮询时每秒重建按钮、重复绑定事件监听器造成页面卡顿。
        const hasProgress = (status === 'running') &&
            ((completedPages !== null && completedPages > 0) ||
             (info.progress && info.progress > 0));
        const hasDoneText = (status === 'done') && !!info.text;
        const prevHasProgress = item.dataset.lastHasProgress === '1';
        const prevHasDoneText = item.dataset.lastHasDoneText === '1';
        const needRebuildActions =
            status !== prevStatus ||
            (status === 'running' && hasProgress !== prevHasProgress) ||
            (status === 'done' && hasDoneText !== prevHasDoneText);
        item.dataset.lastStatus = status;
        item.dataset.lastHasProgress = hasProgress ? '1' : '0';
        item.dataset.lastHasDoneText = hasDoneText ? '1' : '0';
        if (!needRebuildActions) {
            // 对于 error 状态，错误文本可能变化，直接更新已有 span
            if (status === 'error') {
                const errSpan = item.querySelector('.task-actions .task-error-msg');
                if (errSpan) errSpan.textContent = info.error || '处理失败';
            }
            // 更新所属批次头部与全局打包按钮
            const batchId = item.dataset.batchId;
            if (batchId) updateBatchHeader(batchId);
            updateDownloadZipBtn();
            return;
        }

        // 完成后显示结果与下载按钮
        const actions = item.querySelector('.task-actions');
        const resultBox = item.querySelector('.task-result');
        actions.innerHTML = '';
        if (status === 'done' && info.has_result !== false) {
            // 识别后选择导出格式并下载
            const exportRow = document.createElement('div');
            exportRow.className = 'task-export';
            const sel = document.createElement('select');
            sel.className = 'export-format-select';
            sel.title = '选择导出的文件类型';
            const exportOpts = [
                ['searchable_pdf', '可搜索 PDF'],
                ['docx', 'Word 文档 DOCX'],
                ['txt', '纯文本 TXT'],
                ['markdown', 'Markdown'],
                ['json', 'JSON（含坐标）'],
            ];
            exportOpts.forEach((pair) => {
                const o = document.createElement('option');
                o.value = pair[0];
                o.textContent = pair[1];
                sel.appendChild(o);
            });
            // 默认选中与任务原格式一致的项（original 对 PDF 归一化为可搜索 PDF）
            const savedFmt = info.output_format || '';
            const normFmt = savedFmt === 'original' ? 'searchable_pdf' : savedFmt;
            if (Array.prototype.some.call(sel.options, (o) => o.value === normFmt)) {
                sel.value = normFmt;
            }
            exportRow.appendChild(sel);
            // 下载按钮
            const dl = document.createElement('button');
            dl.className = 'btn btn-success btn-sm';
            dl.textContent = '导出下载';
            dl.addEventListener('click', () => downloadResult(taskId, sel.value, info.output_name));
            exportRow.appendChild(dl);
            actions.appendChild(exportRow);
            // 复制文字按钮
            if (info.text) {
                const cp = document.createElement('button');
                cp.className = 'btn btn-ghost btn-sm';
                cp.textContent = '复制文字';
                cp.addEventListener('click', () => copyText(info.text, cp));
                actions.appendChild(cp);
            }
            // 识别文字展示
            resultBox.style.display = '';
            const txt = resultBox.querySelector('.task-result-text');
            txt.textContent = info.text || '(无文字)';
        }
        if (status === 'error') {
            const err = document.createElement('span');
            err.className = 'task-error-msg';
            err.style.color = 'var(--danger)';
            err.style.fontSize = '12px';
            err.textContent = info.error || '处理失败';
            actions.appendChild(err);
            // 重试按钮：中断/失败的任务可重新提交
            // 前提是源文件仍在磁盘上（后端 retry_task 会检查）
            const retry = document.createElement('button');
            retry.className = 'btn btn-warn btn-sm';
            retry.textContent = '重试';
            retry.title = '重新排队执行（源文件需仍在磁盘上）';
            retry.addEventListener('click', () => retryTask(taskId, retry));
            actions.appendChild(retry);
            // 失败状态也可能已有部分页处理完成，允许导出半成品
            // 用于备份已识别内容或跨机器续作
            const expDraft2 = document.createElement('button');
            expDraft2.className = 'btn btn-ghost btn-sm';
            expDraft2.textContent = '导出半成品';
            expDraft2.title = '导出已处理部分（含源文件，可跨机器续作）';
            expDraft2.addEventListener('click', () => exportDraft(taskId, true, expDraft2));
            actions.appendChild(expDraft2);
            // 提前导出 PDF：合并已处理页为最终 PDF
            const finPartial2 = document.createElement('button');
            finPartial2.className = 'btn btn-ghost btn-sm';
            finPartial2.textContent = '导出已识别 PDF';
            finPartial2.title = '把已处理页合并为最终 PDF（不影响重试）';
            finPartial2.addEventListener('click', () => finalizePartial(taskId, finPartial2));
            actions.appendChild(finPartial2);
        }

        // 运行中状态：添加「导出半成品」和「提前导出 PDF」按钮
        // 用于长时间任务提前备份进度或查看部分结果
        if (status === 'running') {
            // 仅当进度 > 0 时显示（有页已处理），使用外部已计算的 hasProgress
            if (hasProgress) {
                const expDraft = document.createElement('button');
                expDraft.className = 'btn btn-ghost btn-sm';
                expDraft.textContent = '导出半成品';
                expDraft.title = '导出当前进度（含源文件，可跨机器续作，不影响任务运行）';
                expDraft.addEventListener('click', () => exportDraft(taskId, true, expDraft));
                actions.appendChild(expDraft);

                const finPartial = document.createElement('button');
                finPartial.className = 'btn btn-ghost btn-sm';
                finPartial.textContent = '导出已识别 PDF';
                finPartial.title = '把已处理页合并为最终 PDF（任务继续运行，可多次导出）';
                finPartial.addEventListener('click', () => finalizePartial(taskId, finPartial));
                actions.appendChild(finPartial);
            }
        }

        // 暂停/恢复按钮
        // running/queued/scheduled → 显示「暂停」
        // paused → 显示「恢复」
        if (status === 'running' || status === 'queued' || status === 'scheduled') {
            const pause = document.createElement('button');
            pause.className = 'btn btn-warn btn-sm';
            pause.textContent = '暂停';
            if (status === 'running') {
                pause.title = '暂停任务：切断 OCR 推理并释放槽位，已处理页保留（断点续传）';
            } else {
                pause.title = '暂停排队中的任务';
            }
            pause.addEventListener('click', () => pauseTask(taskId, pause));
            actions.appendChild(pause);
        } else if (status === 'paused') {
            const resume = document.createElement('button');
            resume.className = 'btn btn-success btn-sm';
            resume.textContent = '恢复';
            resume.title = '恢复任务：重新排队处理（pipeline 自动断点续传）';
            resume.addEventListener('click', () => resumeTask(taskId, resume));
            actions.appendChild(resume);
        }

        // 删除按钮：所有状态都显示（uploading 禁用）
        // running 状态强制删除：切断 OCR 推理 + 取消协程 + 释放槽位
        const del = document.createElement('button');
        del.className = 'btn btn-ghost btn-sm task-del-btn';
        del.textContent = '删除';
        del.title = '删除任务及其文件';
        if (status === 'running') {
            del.title = '强制删除：将中断 OCR 推理并释放资源';
        } else if (status === 'uploading') {
            del.disabled = true;
            del.title = '上传中的任务不能删除';
        } else {
            del.title = '删除任务及其文件';
        }
        if (status !== 'uploading') {
            del.addEventListener('click', () => deleteTask(taskId, del));
        }
        actions.appendChild(del);

        // 更新所属批次头部与全局打包按钮
        const batchId = item.dataset.batchId;
        if (batchId) updateBatchHeader(batchId);
        updateDownloadZipBtn();
    }

    function downloadResult(taskId, fmt, fallbackName) {
        let url = '/api/download/' + taskId;
        if (fmt) url += '?format=' + encodeURIComponent(fmt);
        log('下载结果: ' + taskId + (fmt ? ' (' + fmt + ')' : ''));
        downloadViaFetch(url, fallbackName || ('结果_' + taskId + (fmt ? '_' + fmt : '')));
    }

    // 重试中断/失败的任务
    function retryTask(taskId, btn) {
        if (btn) {
            btn.disabled = true;
            btn.textContent = '提交中...';
        }
        fetch('/api/tasks/' + taskId + '/retry', { method: 'POST' })
            .then(res => {
                if (!res.ok) {
                    return res.json().then(d => {
                        throw new Error(d.detail || '重试失败');
                    });
                }
                return res.json();
            })
            .then(data => {
                log('任务已重新提交: ' + taskId, 'info');
                showToast('已重新提交，排队中', 'success');
                // 重新开始轮询
                startPolling(taskId);
                // 立即查询一次更新 UI
                _pollOnce(taskId);
            })
            .catch(e => {
                log('重试失败: ' + e.message, 'error');
                showToast('重试失败：' + e.message, 'error');
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = '重试';
                }
            });
    }

    // 暂停任务（running/queued/scheduled 可暂停）
    function pauseTask(taskId, btn) {
        if (btn) {
            btn.disabled = true;
            btn.textContent = '暂停中...';
        }
        fetch('/api/tasks/' + taskId + '/pause', { method: 'POST' })
            .then(res => {
                if (!res.ok) {
                    return res.json().then(d => {
                        throw new Error(d.detail || '暂停失败');
                    });
                }
                return res.json();
            })
            .then(data => {
                log('任务已暂停: ' + taskId, 'info');
                showToast('已暂停，可点击「恢复」继续处理（断点续传）', 'success');
                _pollOnce(taskId);
                _broadcastTaskChange('change', taskId);
            })
            .catch(e => {
                log('暂停失败: ' + e.message, 'error');
                showToast('暂停失败：' + e.message, 'error');
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = '暂停';
                }
            });
    }

    // 恢复暂停的任务
    function resumeTask(taskId, btn) {
        if (btn) {
            btn.disabled = true;
            btn.textContent = '恢复中...';
        }
        fetch('/api/tasks/' + taskId + '/resume', { method: 'POST' })
            .then(res => {
                if (!res.ok) {
                    return res.json().then(d => {
                        throw new Error(d.detail || '恢复失败');
                    });
                }
                return res.json();
            })
            .then(data => {
                log('任务已恢复: ' + taskId, 'info');
                showToast('已恢复，重新排队处理（断点续传）', 'success');
                startPolling(taskId);
                _pollOnce(taskId);
                _broadcastTaskChange('change', taskId);
            })
            .catch(e => {
                log('恢复失败: ' + e.message, 'error');
                showToast('恢复失败：' + e.message, 'error');
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = '恢复';
                }
            });
    }

    // 导出半成品（运行中/失败状态都可调用）
    // 打包当前进度为 .ocr_draft ZIP，包含源文件 + 已处理单页
    // 可用于跨机器续作或备份进度
    function exportDraft(taskId, includeSource, btn) {
        if (btn) {
            btn.disabled = true;
            btn.textContent = '导出中...';
        }
        const url = '/api/tasks/' + taskId + '/export_draft?include_source=' +
            (includeSource ? 'true' : 'false');
        fetch(url)
            .then(res => {
                if (!res.ok) {
                    return res.json().then(d => {
                        throw new Error(d.detail || '导出失败');
                    });
                }
                // 从响应头读取进度信息
                const completed = res.headers.get('X-Completed-Pages') || '?';
                const total = res.headers.get('X-Total-Pages') || '?';
                return res.blob().then(blob => ({ blob, completed, total }));
            })
            .then(({ blob, completed, total }) => {
                // 触发下载
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = `draft_${taskId}.ocr_draft`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(a.href);
                log(`导出半成品: ${taskId} (已完成 ${completed}/${total} 页)`, 'info');
                showToast(`已导出半成品（${completed}/${total} 页）`, 'success');
            })
            .catch(e => {
                log('导出半成品失败: ' + e.message, 'error');
                showToast('导出失败：' + e.message, 'error');
            })
            .finally(() => {
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = '导出半成品';
                }
            });
    }

    // 提前导出 PDF（运行中/失败状态都可调用）
    // 把当前已处理页合并为最终 PDF，不影响任务继续运行
    // 可多次调用，每次生成独立的下载链接
    function finalizePartial(taskId, btn) {
        if (btn) {
            btn.disabled = true;
            btn.textContent = '合并中...';
        }
        fetch('/api/tasks/' + taskId + '/finalize_partial', { method: 'POST' })
            .then(res => {
                if (!res.ok) {
                    return res.json().then(d => {
                        throw new Error(d.detail || '导出失败');
                    });
                }
                return res.json();
            })
            .then(data => {
                // 用带 token 的 fetch 下载（window.open 无法携带 X-Access-Token，会 401）
                downloadViaFetch(data.download_url, data.output_name || ('导出_' + taskId + '.pdf'));
                log(
                    `提前导出 PDF: ${taskId} (${data.page_count} 页)`,
                    'info'
                );
                showToast(`已导出 ${data.page_count} 页 PDF，正在下载...`, 'success');
            })
            .catch(e => {
                log('提前导出 PDF 失败: ' + e.message, 'error');
                showToast('导出失败：' + e.message, 'error');
            })
            .finally(() => {
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = '导出已识别 PDF';
                }
            });
    }

    // 删除任务（含工作目录、结果文件）
    // 从 DOM 和 state 中移除任务（供 deleteTask 和轮询 404 复用）
    function removeTaskFromDom(taskId) {
        const item = state.tasks.get(taskId);
        if (!item) return;
        const batchId = item.dataset.batchId;
        item.remove();
        state.tasks.delete(taskId);
        // 从批次的 taskIds 中移除
        if (batchId) {
            const batch = state.batches.get(batchId);
            if (batch) {
                batch.taskIds.delete(taskId);
                // 批次为空则移除整个批次
                if (batch.taskIds.size === 0) {
                    batch.element.remove();
                    state.batches.delete(batchId);
                } else {
                    updateBatchHeader(batchId);
                }
            }
        }
        // 所有任务都删空时显示空提示
        if (state.batches.size === 0 && taskEmpty) taskEmpty.style.display = '';
        updateDownloadZipBtn();
    }

    function deleteTask(taskId, btn) {
        // running 任务强制删除需要更强确认
        const item = state.tasks.get(taskId);
        const isRunning = item && item.dataset.status === 'running';
        const promptMsg = isRunning
            ? '该任务正在处理中！强制删除将中断 OCR 推理并丢弃已处理内容。\n\n确认强制删除？'
            : '确认删除此任务？相关文件将一并删除。';
        if (!confirm(promptMsg)) return;
        if (btn) {
            btn.disabled = true;
            btn.textContent = '删除中...';
        }
        fetch('/api/tasks/' + taskId, { method: 'DELETE' })
            .then(res => {
                if (!res.ok) {
                    return res.json().then(d => {
                        throw new Error(d.detail || '删除失败');
                    });
                }
                return res.json();
            })
            .then(data => {
                log('任务已删除: ' + taskId, 'info');
                showToast('任务已删除', 'success');
                // 停止轮询
                stopPolling(taskId);
                // 从 DOM 移除任务条目
                removeTaskFromDom(taskId);
                // 删除后刷新限制数据（current_pending 已变化）
                fetchLimits();
                // 广播删除事件给其他标签页
                _broadcastTaskChange('delete', taskId);
            })
            .catch(e => {
                log('删除失败: ' + e.message, 'error');
                showToast('删除失败：' + e.message, 'error');
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = '删除';
                }
            });
    }

    // 停止轮询指定任务
    function stopPolling(taskId) {
        state.pollingIds.delete(taskId);
    }

    function copyText(text, btn) {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(() => {
                const old = btn.textContent;
                btn.textContent = '已复制';
                setTimeout(() => { btn.textContent = old; }, 1500);
            });
        } else {
            log('当前浏览器不支持自动复制，请手动选择文字复制', 'warn');
        }
    }

    // ---------------- 轮询任务状态 ----------------
    // 自适应轮询：根据任务状态动态调整间隔，避免无意义的高频请求
    //   running   → 3 秒（需要及时看到进度更新）
    //   queued    → 6 秒（排队状态变化慢）
    //   scheduled → 20 秒（预约任务可能等待数小时，没必要频繁查询）
    // 用 setTimeout 递归而非 setInterval，避免请求耗时叠加导致请求堆积
    //
    // 页面可见性检测：标签页隐藏时暂停所有轮询，可见时恢复。
    // 避免多标签页同时轮询导致请求量爆炸（15个标签页 × N个任务 = 大量重复请求）。
    // 只有用户正在看的那个标签页才发请求。
    state.visible = !document.hidden;
    state.pollingPaused = new Set(); // 暂停的 taskId（页面隐藏时记录，恢复时重启）
    document.addEventListener('visibilitychange', () => {
        const nowVisible = !document.hidden;
        state.visible = nowVisible;
        if (nowVisible) {
            // 页面恢复可见：全量同步任务列表
            // 旧逻辑只对 pollingPaused 中的 taskId 调 _pollOnce，无法感知其他标签页的
            // 增删操作。改为调 syncTasks 全量拉取列表，同步新增/删除的任务。
            syncTasks();
            // 同时恢复已有非终态任务的轮询
            const toResume = Array.from(state.pollingPaused);
            state.pollingPaused.clear();
            toResume.forEach(tid => {
                if (state.pollingIds.has(tid)) {
                    _pollOnce(tid);
                }
            });
        }
    });

    // 单次查询（供恢复轮询时使用）
    async function _pollOnce(taskId) {
        try {
            const res = await fetch('/api/tasks/' + taskId);
            if (!res.ok) {
                if (res.status === 404) {
                    // 任务在后端已不存在（被其他标签页删除或服务重启清理）
                    // 直接从 DOM 移除，而非标记为 error（旧逻辑会留下空壳任务行）
                    log('任务 ' + taskId + ' 不存在（已被删除或服务重启），从列表移除', 'warn');
                    stopPolling(taskId);
                    removeTaskFromDom(taskId);
                    fetchLimits();
                } else {
                    log('查询任务 ' + taskId + ' 失败: ' + res.status, 'error');
                    stopPolling(taskId);
                }
                return;
            }
            const info = await res.json();
            updateTaskItem(taskId, info);
            if (info.status === 'done') {
                stopPolling(taskId);
                log('任务完成: ' + (info.source_name || taskId) +
                    (info.pages ? ' (' + info.pages + ' 页)' : ''), 'info');
                showToast('识别完成：' + (info.source_name || taskId) +
                    (info.pages ? ' (' + info.pages + ' 页)' : ''), 'success');
                return;
            }
            if (info.status === 'error') {
                stopPolling(taskId);
                log('任务失败: ' + (info.source_name || taskId) + ' - ' + (info.error || ''), 'error');
                showToast('识别失败：' + (info.source_name || taskId), 'error');
                return;
            }
            // 安排下次轮询
            _scheduleNext(taskId, info.status);
        } catch (e) {
            log('轮询任务 ' + taskId + ' 异常: ' + e.message, 'error');
            _scheduleNext(taskId, 'error');
        }
    }

    function _scheduleNext(taskId, status) {
        if (!state.pollingIds.has(taskId)) return;
        // 页面不可见时暂停，等恢复后再查
        if (!state.visible) {
            state.pollingPaused.add(taskId);
            return;
        }
        let interval;
        if (status === 'running') interval = 3000;
        else if (status === 'queued') interval = 6000;
        else if (status === 'scheduled') interval = 20000;
        else if (status === 'paused') interval = 30000;  // 暂停状态低频轮询
        else interval = 5000;
        const timer = setTimeout(() => _pollOnce(taskId), interval);
        state.pollingIds[taskId] = timer;
    }

    function startPolling(taskId) {
        if (state.pollingIds.has(taskId)) return;
        state.pollingIds.add(taskId);
        // 页面不可见时不立即查询，等恢复后再查
        if (!state.visible) {
            state.pollingPaused.add(taskId);
            return;
        }
        _pollOnce(taskId);
    }

    function stopPolling(taskId) {
        state.pollingIds.delete(taskId);
        state.pollingPaused.delete(taskId);
        const timer = state.pollingIds[taskId];
        if (timer) {
            clearTimeout(timer);
            delete state.pollingIds[taskId];
        }
    }

    // ---------------- 清理已完成 ----------------
    // 整批清理：当批次内所有任务都已完成（done/error）时，移除整个批次
    // 关键：必须调用后端 DELETE 接口，否则刷新页面后任务还会从 _tasks_state.json 恢复
    clearDoneBtn.addEventListener('click', async () => {
        const batchesToRemove = [];
        const tasksToDelete = [];  // [{taskId, batchId}]
        state.batches.forEach((batch, batchId) => {
            let allDone = true;
            let anyTask = false;
            batch.taskIds.forEach(taskId => {
                const item = state.tasks.get(taskId);
                if (item) {
                    anyTask = true;
                    if (!item.classList.contains('done') && !item.classList.contains('error')) {
                        allDone = false;
                    }
                }
            });
            if (allDone && anyTask) {
                batchesToRemove.push(batchId);
                batch.taskIds.forEach(taskId => tasksToDelete.push({ taskId, batchId }));
            }
        });

        if (tasksToDelete.length === 0) {
            showToast('没有可清理的已完成批次', 'info');
            return;
        }

        // 禁用按钮防止重复点击
        clearDoneBtn.disabled = true;
        const originalText = clearDoneBtn.textContent;
        clearDoneBtn.textContent = '清理中...';

        // 并发调用 DELETE 接口（每个任务独立删除，失败的不影响其他）
        const results = await Promise.allSettled(
            tasksToDelete.map(({ taskId }) =>
                fetch('/api/tasks/' + taskId, { method: 'DELETE' })
                    .then(res => res.ok ? res.json() : res.json().then(d => Promise.reject(new Error(d.detail || '删除失败'))))
            )
        );

        let successCount = 0;
        let failCount = 0;
        const successTaskIds = new Set();
        results.forEach((r, i) => {
            const { taskId, batchId } = tasksToDelete[i];
            if (r.status === 'fulfilled') {
                successCount++;
                successTaskIds.add(taskId);
                stopPolling(taskId);
                state.tasks.delete(taskId);
            } else {
                failCount++;
                log('删除任务 ' + taskId + ' 失败: ' + (r.reason?.message || r.reason), 'error');
            }
        });

        // 移除完全成功的批次（批次内所有任务都已删除）
        batchesToRemove.forEach(batchId => {
            const batch = state.batches.get(batchId);
            if (!batch) return;
            // 检查批次内是否所有任务都已成功删除
            const allDeleted = Array.from(batch.taskIds).every(tid => successTaskIds.has(tid));
            if (allDeleted) {
                batch.element.remove();
                state.batches.delete(batchId);
                // 从成功列表中移除该批次的 taskIds
                batch.taskIds.forEach(tid => successTaskIds.delete(tid));
            } else {
                // 部分删除：只移除成功的任务条目
                successTaskIds.forEach(tid => {
                    if (batch.taskIds.has(tid)) {
                        const item = state.tasks.get(tid);
                        if (item) item.remove();
                        batch.taskIds.delete(tid);
                    }
                });
                if (batch.taskIds.size === 0) {
                    batch.element.remove();
                    state.batches.delete(batchId);
                } else {
                    updateBatchHeader(batchId);
                }
                successTaskIds.clear();
            }
        });

        // 移除剩余的成功任务条目（防止漏掉）
        successTaskIds.forEach(taskId => {
            const item = state.tasks.get(taskId);
            if (item) item.remove();
        });

        if (state.batches.size === 0 && taskEmpty) taskEmpty.style.display = '';
        updateDownloadZipBtn();
        // 刷新限制数据（current_pending 已变化）
        fetchLimits();

        // 恢复按钮
        clearDoneBtn.disabled = false;
        clearDoneBtn.textContent = originalText;

        if (successCount > 0) {
            log('已清理 ' + successCount + ' 个已完成文件' +
                (failCount > 0 ? '（' + failCount + ' 个失败）' : ''));
            showToast(
                '已清理 ' + successCount + ' 个已完成文件' +
                (failCount > 0 ? '（' + failCount + ' 个失败）' : ''),
                failCount > 0 ? 'warn' : 'success'
            );
        } else {
            showToast('清理失败：' + (results[0]?.reason?.message || '未知错误'), 'error');
        }
    });

    // 更新全局打包下载按钮的启用状态：有任意已完成任务时启用
    function updateDownloadZipBtn() {
        let doneCount = 0;
        state.tasks.forEach((item) => {
            if (item.classList.contains('done')) doneCount++;
        });
        downloadZipBtn.disabled = doneCount === 0;
        downloadZipBtn.textContent = doneCount > 0
            ? '打包全部 (' + doneCount + ')'
            : '打包全部';
    }

    // ---------------- 清空日志 ----------------
    clearLogBtn.addEventListener('click', () => { logBox.innerHTML = ''; });

    // ---------------- 文件夹选择 → 加入清单 ----------------
    // webkitdirectory：浏览器返回文件夹内所有文件（含子目录）的 FileList
    // 过滤支持的扩展名后加入清单（不直接上传）
    folderBtn.addEventListener('click', () => folderInput.click());
    folderInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (!files || !files.length) return;
        const total = files.length;
        const before = state.cart.length;
        addToCart(files);
        const added = state.cart.length - before;
        const skipped = total - added;
        if (added === 0) {
            showToast('文件夹内没有新增的支持文件（共扫描 ' + total + ' 个）', 'warn');
        } else {
            showToast('已加入清单 ' + added + ' 个文件（来自文件夹，跳过 ' + skipped + ' 个）', 'info');
        }
        // 清空 input.value 允许重复选择同一文件夹
        e.target.value = '';
    });

    // ---------------- 导入半成品续作 ----------------
    // 上传 .ocr_draft 文件，后端解压后从断点继续处理
    // 前提：半成品必须包含源文件（include_source=true 导出的）
    draftBtn.addEventListener('click', () => draftInput.click());
    draftInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        // 校验扩展名（浏览器 accept 已过滤，这里二次校验防止拖入）
        if (!file.name.toLowerCase().endsWith('.ocr_draft')) {
            showToast('请选择 .ocr_draft 格式的半成品文件', 'warn');
            e.target.value = '';
            return;
        }
        // 文件大小预检（与后端限制一致）
        const limits = await fetch('/api/limits').then(r => r.json()).catch(() => null);
        if (limits && limits.max_file_size_mb &&
            file.size > limits.max_file_size_mb * 1024 * 1024) {
            showToast(
                `文件过大：${(file.size / 1024 / 1024).toFixed(1)} MB，上限 ${limits.max_file_size_mb} MB`,
                'error'
            );
            e.target.value = '';
            return;
        }

        // 确认导入
        if (!confirm(
            `确认导入半成品续作？\n\n` +
            `文件名：${file.name}\n` +
            `大小：${(file.size / 1024 / 1024).toFixed(1)} MB\n\n` +
            `将解压并从断点继续处理，已识别的页不会重复处理。`
        )) {
            e.target.value = '';
            return;
        }

        // 上传到 /api/upload_draft
        draftBtn.disabled = true;
        const originalText = draftBtn.textContent;
        draftBtn.textContent = '导入中...';

        const formData = new FormData();
        formData.append('file', file);
        // 输出格式在识别后选择，导入时使用服务端默认格式
        const outputFormat = null;

        try {
            const res = await fetch('/api/upload_draft', {
                method: 'POST',
                body: formData,
            });
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.detail || '导入失败');
            }
            // 导入成功，添加任务到列表
            log(
                `导入半成品成功：${data.source_name} ` +
                `(已完成 ${data.completed_pages}/${data.total_pages} 页，从第 ${data.completed_pages + 1} 页继续)`,
                'info'
            );
            showToast(
                `已导入：${data.source_name}（从第 ${data.completed_pages + 1} 页继续）`,
                'success'
            );
            // 添加任务条目并开始轮询
            addTaskItem(
                data.task_id,
                data.source_name,
                data.status || 'queued',
                null,
                null,
                null  // 导入的任务没有 batch_id
            );
            startPolling(data.task_id);
            // 刷新限制数据
            fetchLimits();
        } catch (err) {
            log('导入半成品失败：' + err.message, 'error');
            showToast('导入失败：' + err.message, 'error');
        } finally {
            draftBtn.disabled = false;
            draftBtn.textContent = originalText;
            // 清空 input 允许重复选择同一文件
            e.target.value = '';
        }
    });

    // ---------------- 打包下载（zip） ----------------
    downloadZipBtn.addEventListener('click', () => {
        // 收集所有已完成（done）的任务 ID
        const doneIds = [];
        state.tasks.forEach((item, taskId) => {
            if (item.classList.contains('done')) {
                doneIds.push(taskId);
            }
        });
        if (doneIds.length === 0) {
            showToast('没有已完成的任务可下载', 'warn');
            return;
        }
        const url = '/api/download_zip?task_ids=' + encodeURIComponent(doneIds.join(','));
        log('打包下载 ' + doneIds.length + ' 个结果文件...');
        showToast('正在打包 ' + doneIds.length + ' 个文件...', 'info');
        downloadViaFetch(url, 'OCR结果.zip');
    });

    // ---------------- 配置加载 / 保存 ----------------
    async function loadConfig() {
        try {
            const res = await fetch('/api/config');
            const cfg = await res.json();
            cfgDpi.value = cfg.render_dpi || 200;
            cfgConc.value = cfg.max_concurrent || 3;
            cfgLayer2.checked = !!(cfg.filter && cfg.filter.enable_layer2);
            cfgLayout.checked = !!(cfg.paddle && cfg.paddle.enable_layout);
            cfgTableRec.checked = !!(cfg.paddle && cfg.paddle.use_table_recognition);
            cfgTier.value = (cfg.paddle && cfg.paddle.ocr_model_tier) || 'small';
        } catch (e) {
            log('加载配置失败: ' + e.message, 'error');
        }
    }

    saveConfigBtn.addEventListener('click', async () => {
        const body = {
            render_dpi: parseInt(cfgDpi.value, 10) || 200,
            max_concurrent: parseInt(cfgConc.value, 10) || 3,
            enable_layer2: cfgLayer2.checked,
            enable_layout: cfgLayout.checked,
            use_table_recognition: cfgTableRec.checked,
            ocr_model_tier: cfgTier.value,
        };
        saveHint.textContent = '保存中…';
        saveHint.classList.remove('err');
        try {
            const res = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || '保存失败');
            // 检测是否有需要重启才生效的配置变更
            if (data.needs_restart && data.restart_fields && data.restart_fields.length > 0) {
                const fields = data.restart_fields.join('、');
                saveHint.textContent = '已保存，需重启生效';
                saveHint.classList.add('err');
                log('配置已保存，以下项目需重启服务才生效：' + fields, 'warn');
                // 提示用户并自动关闭服务
                const ok = confirm(
                    '配置已保存。\n\n' +
                    '以下项目需要重启服务才能生效：\n' +
                    fields + '\n\n' +
                    '点击"确定"立即关闭服务（请手动重新启动 exe），' +
                    '点击"取消"稍后手动重启。'
                );
                if (ok) {
                    saveHint.textContent = '服务关闭中…';
                    try {
                        await fetch('/api/shutdown', { method: 'POST' });
                        saveHint.textContent = '服务已关闭，请重新启动 exe';
                        log('服务已关闭，请重新启动 exe');
                        // 停止所有轮询，避免连接错误刷屏
                        state.polling = {};
                    } catch (e) {
                        // 服务关闭后连接会失败，这是正常的
                        saveHint.textContent = '服务已关闭，请重新启动 exe';
                        log('服务已关闭，请重新启动 exe');
                    }
                } else {
                    log('未关闭服务，请稍后手动重启以使配置生效', 'warn');
                }
            } else {
                saveHint.textContent = '已保存';
                log('配置已保存');
                setTimeout(() => { saveHint.textContent = ''; }, 2000);
            }
        } catch (e) {
            saveHint.textContent = '保存失败: ' + e.message;
            saveHint.classList.add('err');
            log('保存配置失败: ' + e.message, 'error');
        }
    });

    // ---------------- 加载已有任务列表 ----------------
    // 全量同步任务列表：拉取后端列表，与本地 state.tasks 对比，
    // 移除后端已不存在的任务，新增本地没有的任务，更新已有任务状态。
    // 用于标签页恢复可见时同步其他标签页的增删操作。
    async function syncTasks() {
        try {
            const res = await fetch('/api/tasks');
            if (!res.ok) return;
            const data = await res.json();
            const remoteTasks = data.tasks || [];
            const remoteIds = new Set(remoteTasks.map(t => t.task_id).filter(Boolean));

            // 1. 移除本地有但后端已不存在的任务（被其他标签页删除）
            for (const localId of Array.from(state.tasks.keys())) {
                if (!remoteIds.has(localId)) {
                    stopPolling(localId);
                    removeTaskFromDom(localId);
                }
            }

            // 2. 新增后端有但本地没有的任务（其他标签页提交的新任务）
            remoteTasks.sort((a, b) => (a.created_at || 0) - (b.created_at || 0));
            remoteTasks.forEach(t => {
                if (!t.task_id) return;
                if (!state.tasks.has(t.task_id)) {
                    addTaskItem(
                        t.task_id,
                        t.source_name || t.task_id,
                        t.status,
                        t.queue_position,
                        t.queue_reason,
                        t.batch_id
                    );
                }
                // 更新已有任务的状态
                updateTaskItem(t.task_id, t);
                // 非终态且未在轮询的任务启动轮询
                if (t.status !== 'done' && t.status !== 'error' &&
                    !state.pollingIds.has(t.task_id)) {
                    startPolling(t.task_id);
                }
            });

            // 3. 所有任务都空时显示空提示
            if (remoteTasks.length === 0 && state.batches.size === 0 && taskEmpty) {
                taskEmpty.style.display = '';
            }
        } catch (e) {
            log('同步任务列表失败: ' + e.message, 'warn');
        }
    }

    async function loadTasks() {
        try {
            const res = await fetch('/api/tasks');
            if (!res.ok) return;
            const data = await res.json();
            const tasks = data.tasks || [];
            if (tasks.length === 0) return;
            // 按创建时间正序（旧→新），然后依次 addTaskItem（新的会插到顶部）
            // 这样最终顺序就是最新在前
            tasks.sort((a, b) => (a.created_at || 0) - (b.created_at || 0));
            tasks.forEach(t => {
                if (!t.task_id) return;
                addTaskItem(
                    t.task_id,
                    t.source_name || t.task_id,
                    t.status,
                    t.queue_position,
                    t.queue_reason,
                    t.batch_id
                );
                // 非终态任务开启轮询
                if (t.status !== 'done' && t.status !== 'error') {
                    startPolling(t.task_id);
                }
            });
            log('已加载 ' + tasks.length + ' 个历史任务');
        } catch (e) {
            log('加载任务列表失败: ' + e.message, 'warn');
        }
    }

    // ---------------- 初始化 ----------------
    function init() {
        log('server-paddle OCR 前端已加载');
        // 二维码加载成功/失败事件
        qrImg.addEventListener('load', () => {
            // 检测是否为 1x1 透明占位图（qrcode 库未装时后端返回）
            if (qrImg.naturalWidth <= 1) {
                state.qrFailed = true;
                qrImg.style.display = 'none';
                log('二维码库未安装，二维码已隐藏（不影响 OCR 功能）', 'warn');
            } else {
                state.qrLoaded = true;
            }
        });
        qrImg.addEventListener('error', () => {
            state.qrFailed = true;
            qrImg.style.display = 'none';
            log('二维码加载失败，已隐藏', 'warn');
        });
        fetchStatus();
        fetchLimits();
        loadConfig();
        loadTasks();
        // 设置预约时间选择器的最小值为当前时间（避免选过去时间）
        const now = new Date();
        now.setMinutes(now.getMinutes() - now.getTimezoneOffset());  // 转本地时区
        scheduleInput.min = now.toISOString().slice(0, 16);
        // 状态轮询：每 3 秒，但页面不可见时暂停
        // 避免多标签页同时轮询导致请求量爆炸
        let statusTimer = null;
        const scheduleStatusPoll = () => {
            if (statusTimer) clearTimeout(statusTimer);
            if (!state.visible) return;  // 不可见时不安排下次轮询
            statusTimer = setTimeout(async () => {
                await fetchStatus();
                scheduleStatusPoll();
            }, 3000);
        };
        // 页面可见性变化时恢复/暂停 status 轮询
        document.addEventListener('visibilitychange', () => {
            if (state.visible) {
                fetchStatus();  // 恢复可见时立即查一次
                scheduleStatusPoll();
            }
        });
        scheduleStatusPoll();
    }

    init();
})();
