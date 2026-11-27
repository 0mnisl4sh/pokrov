(function () {
  "use strict";

  var THEME_KEY = "pokrov_ui_theme";
  var API_KEY_STORAGE = "pokrov_ui_api_key";

  var state = {
    presets: [],
    statsLabels: {
      categories: {},
      actions: {},
      controlPoints: {}
    },
    autoRefreshTimer: null
  };

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined && text !== null) {
      node.textContent = String(text);
    }
    return node;
  }

  function clearNode(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function getApiKey() {
    return window.sessionStorage.getItem(API_KEY_STORAGE) || "";
  }

  function setApiKey(val) {
    if (val) {
      window.sessionStorage.setItem(API_KEY_STORAGE, val);
    } else {
      window.sessionStorage.removeItem(API_KEY_STORAGE);
    }
  }

  function showAuthBar(message) {
    var bar = document.getElementById("auth-bar");
    var msg = document.getElementById("auth-msg");
    if (message) {
      msg.textContent = message;
    }
    bar.classList.remove("hidden");
  }

  function apiHeaders(extra) {
    var headers = Object.assign({}, extra || {});
    var key = getApiKey();
    if (key) {
      headers["X-Pokrov-Api-Key"] = key;
    }
    return headers;
  }

  async function apiFetch(url, options) {
    var opts = Object.assign({}, options || {});
    opts.headers = apiHeaders(opts.headers);
    var resp = await window.fetch(url, opts);
    if (resp.status === 401) {
      showAuthBar("Шлюз вернул 401: введите X-Pokrov-Api-Key для продолжения работы.");
      throw new Error("Unauthorized (401)");
    }
    if (!resp.ok) {
      var errText = "HTTP " + resp.status;
      try {
        var errJson = await resp.json();
        if (errJson && errJson.detail) {
          errText = String(errJson.detail);
        }
      } catch (_e) {
        /* ignore non-json error */
      }
      throw new Error(errText);
    }
    return resp;
  }

  function applyTheme(theme) {
    var t = theme === "light" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", t);
    var btn = document.getElementById("btn-theme-toggle");
    btn.textContent = t === "dark" ? "Светлая тема" : "Тёмная тема";
    window.localStorage.setItem(THEME_KEY, t);
  }

  function actionPillClass(action) {
    if (action === "block" || action === "require_approval") {
      return "pill pill-crit";
    }
    if (action === "route_local") {
      return "pill pill-high";
    }
    if (action === "mask" || action === "mask_reversible") {
      return "pill pill-warn";
    }
    if (action === "flag") {
      return "pill pill-info";
    }
    return "pill pill-ok";
  }

  function severityPillClass(sev) {
    if (sev === "critical") {
      return "pill pill-crit";
    }
    if (sev === "high") {
      return "pill pill-high";
    }
    if (sev === "medium") {
      return "pill pill-warn";
    }
    return "pill pill-info";
  }

  function routerPillClass(decision) {
    if (decision === "block") {
      return "pill pill-crit";
    }
    if (decision === "local") {
      return "pill pill-ok";
    }
    if (decision === "ru_cloud") {
      return "pill pill-info";
    }
    return "pill pill-warn";
  }

  function renderHighlightedText(container, rawText, spans) {
    clearNode(container);
    if (!rawText) {
      return;
    }
    var sorted = (spans || []).slice().sort(function (a, b) {
      if (a.start !== b.start) {
        return a.start - b.start;
      }
      return b.end - a.end;
    });

    var cursor = 0;
    for (var i = 0; i < sorted.length; i++) {
      var sp = sorted[i];
      var s = Math.max(cursor, Math.min(rawText.length, sp.start));
      var e = Math.max(s, Math.min(rawText.length, sp.end));
      if (e <= s) {
        continue;
      }
      if (s > cursor) {
        container.appendChild(document.createTextNode(rawText.slice(cursor, s)));
      }
      var mark = el("mark", "hl-span hl-sev-" + sp.severity, rawText.slice(s, e));
      mark.title = sp.category_label_ru + " (" + sp.action + ")";
      container.appendChild(mark);
      cursor = e;
    }
    if (cursor < rawText.length) {
      container.appendChild(document.createTextNode(rawText.slice(cursor)));
    }
  }

  function renderPlaygroundResult(rawText, data) {
    document.getElementById("pg-empty-hint").classList.add("hidden");
    document.getElementById("pg-results").classList.remove("hidden");

    var actionBadge = document.getElementById("res-action-badge");
    clearNode(actionBadge);
    actionBadge.appendChild(el("span", actionPillClass(data.action), data.action + " — " + data.action_label_ru));

    var routerBadge = document.getElementById("res-router-badge");
    clearNode(routerBadge);
    routerBadge.appendChild(el("span", routerPillClass(data.router.decision), data.router.label_ru));
    document.getElementById("res-router-sub").textContent =
      data.router.upstream === "none"
        ? "Передача во внешние и локальные модели отменена"
        : "Upstream: " + data.router.upstream + " (" + data.router.model + ")";

    document.getElementById("res-latency").textContent = data.latency_ms.toFixed(2) + " ms";
    document.getElementById("res-layers").textContent = "Слои: " + data.layers_executed.join(", ");

    document.getElementById("res-det-count").textContent = String(data.detections_count);
    document.getElementById("res-evasion").textContent = data.evasion_detected
      ? "Обнаружена попытка обхода (анти-обход)"
      : "Признаки обхода не выявлены";

    document.getElementById("res-cp-label").textContent = data.control_point_label_ru;
    renderHighlightedText(document.getElementById("res-highlighted"), rawText, data.spans);

    var maskedNote = document.getElementById("res-masked-note");
    var maskedPre = document.getElementById("res-masked");
    maskedNote.classList.add("hidden");
    if (data.blocked) {
      maskedPre.textContent = "[ЗАБЛОКИРОВАНО ПОЛИТИКОЙ] " + data.reason;
    } else {
      maskedPre.textContent = data.masked_text;
      if (data.router.decision === "local" && !data.masking_applied) {
        maskedNote.textContent =
          "Политика route_local: запрос целиком перенаправлен в доверенный локальный контур без отправки в облако.";
        maskedNote.classList.remove("hidden");
      }
    }

    var tbody = document.getElementById("res-spans-body");
    clearNode(tbody);
    if (!data.spans.length) {
      var emptyRow = el("tr");
      var emptyCell = el("td", "muted", "Угрозы и персональные данные не обнаружены.");
      emptyCell.colSpan = 7;
      emptyRow.appendChild(emptyCell);
      tbody.appendChild(emptyRow);
    } else {
      for (var i = 0; i < data.spans.length; i++) {
        var sp = data.spans[i];
        var tr = el("tr");
        tr.appendChild(el("td", "", sp.category_label_ru + " (" + sp.category + ")"));

        var tdSev = el("td");
        tdSev.appendChild(el("span", severityPillClass(sp.severity), sp.severity_label_ru));
        tr.appendChild(tdSev);

        var tdAct = el("td");
        tdAct.appendChild(el("span", actionPillClass(sp.action), sp.action));
        tr.appendChild(tdAct);

        tr.appendChild(el("td", "mono", sp.detector));
        tr.appendChild(el("td", "mono", String(sp.confidence)));

        var evasionInfo = sp.evasion_techniques.length
          ? sp.span_source + " (" + sp.evasion_techniques.join(", ") + ")"
          : sp.span_source;
        tr.appendChild(el("td", "", evasionInfo));

        var preview = sp.masked_preview || "—";
        var hmacShort = sp.value_hmac ? sp.value_hmac.slice(0, 12) + "…" : "—";
        tr.appendChild(el("td", "mono", preview + " | " + hmacShort));
        tbody.appendChild(tr);
      }
    }

    var compGrid = document.getElementById("res-compliance-list");
    clearNode(compGrid);
    if (!data.compliance_refs.length) {
      compGrid.appendChild(el("p", "muted", "Нарушения нормативных требований не зафиксированы."));
    } else {
      for (var c = 0; c < data.compliance_refs.length; c++) {
        var ref = data.compliance_refs[c];
        var card = el("div", "compliance-card");
        card.appendChild(el("div", "compliance-title", ref.framework + " — " + ref.article));
        card.appendChild(el("div", "", ref.title));
        if (ref.risk_hint) {
          card.appendChild(el("div", "compliance-hint", ref.risk_hint));
        }
        compGrid.appendChild(card);
      }
    }
  }

  async function loadPresets() {
    try {
      var resp = await apiFetch("/api/playground/presets");
      var data = await resp.json();
      state.presets = data.presets || [];
      var container = document.getElementById("presets-container");
      clearNode(container);
      for (var i = 0; i < state.presets.length; i++) {
        (function (preset) {
          var btn = el("button", "preset-btn", preset.title);
          btn.type = "button";
          btn.title = preset.description;
          btn.addEventListener("click", function () {
            document.getElementById("pg-input").value = preset.text;
            runPlayground();
          });
          container.appendChild(btn);
        })(state.presets[i]);
      }
    } catch (_e) {
      /* handled by auth bar if 401 */
    }
  }

  async function runPlayground() {
    var text = document.getElementById("pg-input").value;
    if (!text.trim()) {
      return;
    }
    var btn = document.getElementById("btn-pg-run");
    btn.disabled = true;
    try {
      var resp = await apiFetch("/api/playground", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          profile: document.getElementById("pg-profile").value,
          control_point: document.getElementById("pg-control-point").value,
          record_audit: document.getElementById("pg-record-audit").checked
        })
      });
      var data = await resp.json();
      renderPlaygroundResult(text, data);
    } catch (err) {
      window.alert("Ошибка инспекции: " + err.message);
    } finally {
      btn.disabled = false;
    }
  }

  function renderCategoryBars(byCategory) {
    var container = document.getElementById("category-bars");
    clearNode(container);
    var entries = Object.keys(byCategory || {}).map(function (k) {
      return [k, byCategory[k]];
    });
    entries.sort(function (a, b) {
      return b[1] - a[1];
    });
    if (!entries.length) {
      container.appendChild(el("p", "muted", "События обнаружений пока отсутствуют."));
      return;
    }
    var maxVal = entries[0][1] || 1;
    for (var i = 0; i < entries.length; i++) {
      var cat = entries[i][0];
      var cnt = entries[i][1];
      var label = state.statsLabels.categories[cat] || cat;
      var row = el("div", "bar-row");
      row.appendChild(el("div", "", label));
      var track = el("div", "bar-track");
      var fill = el("div", "bar-fill");
      fill.style.width = Math.max(4, Math.round((cnt / maxVal) * 100)) + "%";
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el("div", "mono", String(cnt)));
      container.appendChild(row);
    }
  }

  function populateCategoryFilter(catLabels) {
    var sel = document.getElementById("flt-category");
    var current = sel.value;
    if (sel.options.length > 1) {
      return;
    }
    var keys = Object.keys(catLabels || {});
    for (var i = 0; i < keys.length; i++) {
      var opt = el("option", "", catLabels[keys[i]] + " (" + keys[i] + ")");
      opt.value = keys[i];
      sel.appendChild(opt);
    }
    sel.value = current;
  }

  async function loadAuditStatsAndEvents() {
    try {
      var statsResp = await apiFetch("/api/stats");
      var stats = await statsResp.json();
      state.statsLabels.categories = stats.category_labels_ru || {};
      state.statsLabels.actions = stats.action_labels_ru || {};
      state.statsLabels.controlPoints = stats.control_point_labels_ru || {};

      document.getElementById("hdr-profile").textContent = stats.active_profile || "—";
      document.getElementById("kpi-total").textContent = String(stats.total_requests || 0);
      document.getElementById("kpi-pii").textContent = String(stats.pii_leaks_stopped || 0);
      document.getElementById("kpi-blocked").textContent = String(stats.blocked_count || 0);
      document.getElementById("kpi-masked").textContent = String(stats.masked_count || 0);
      document.getElementById("kpi-local").textContent = String(stats.routed_local_count || 0);
      document.getElementById("kpi-p95").textContent = Number(stats.p95_latency_ms || 0).toFixed(2);
      document.getElementById("audit-status-text").textContent = stats.audit_enabled
        ? "SQLite Zero-Retention (очередь: " + stats.queue_size + ")"
        : "Аудит в БД отключён";

      renderCategoryBars(stats.by_category);
      populateCategoryFilter(state.statsLabels.categories);

      var params = new window.URLSearchParams();
      params.set("limit", "50");
      var fltAction = document.getElementById("flt-action").value;
      var fltCat = document.getElementById("flt-category").value;
      if (fltAction) {
        params.set("action", fltAction);
      }
      if (fltCat) {
        params.set("category", fltCat);
      }

      var evResp = await apiFetch("/api/events?" + params.toString());
      var evData = await evResp.json();
      renderAuditEvents(evData.events || []);
    } catch (_e) {
      /* handled */
    }
  }

  function renderAuditEvents(events) {
    var tbody = document.getElementById("audit-events-body");
    clearNode(tbody);
    if (!events.length) {
      var emptyTr = el("tr");
      var emptyTd = el("td", "muted", "События аудита по выбранным фильтрам отсутствуют.");
      emptyTd.colSpan = 9;
      emptyTr.appendChild(emptyTd);
      tbody.appendChild(emptyTr);
      return;
    }

    for (var i = 0; i < events.length; i++) {
      (function (ev) {
        var tr = el("tr");
        tr.appendChild(el("td", "mono", String(ev.timestamp || "").replace("T", " ").slice(0, 19)));

        var tdAct = el("td");
        tdAct.appendChild(el("span", actionPillClass(ev.action), ev.action));
        tr.appendChild(tdAct);

        var tdSev = el("td");
        tdSev.appendChild(el("span", severityPillClass(ev.severity), ev.severity));
        tr.appendChild(tdSev);

        tr.appendChild(el("td", "mono", ev.control_point));
        tr.appendChild(el("td", "mono", (ev.categories || []).join(", ") || "—"));
        tr.appendChild(el("td", "mono", ev.upstream || "—"));
        tr.appendChild(el("td", "", ev.policy_profile || "—"));
        tr.appendChild(el("td", "mono", Number(ev.latency_ms || 0).toFixed(2)));

        var tdBtn = el("td");
        var toggleBtn = el("button", "btn btn-ghost btn-sm", "Подробнее");
        toggleBtn.type = "button";
        tdBtn.appendChild(toggleBtn);
        tr.appendChild(tdBtn);

        var detailTr = el("tr", "detail-row hidden");
        var detailTd = el("td");
        detailTd.colSpan = 9;
        var hmacs = (ev.value_hmacs || []).join(", ") || "—";
        var refs = (ev.compliance_refs || [])
          .map(function (r) {
            return r.framework + " " + r.article;
          })
          .join("; ");
        detailTd.textContent =
          "event_id=" + ev.event_id +
          " | request_id=" + ev.request_id +
          " | principal=" + ev.principal +
          " | evasion=" + String(Boolean(ev.evasion_detected)) +
          " | HMAC=" + hmacs +
          (refs ? " | Нормы: " + refs : "") +
          (ev.reason ? " | Причина: " + ev.reason : "");
        detailTr.appendChild(detailTd);

        toggleBtn.addEventListener("click", function () {
          detailTr.classList.toggle("hidden");
        });

        tbody.appendChild(tr);
        tbody.appendChild(detailTr);
      })(events[i]);
    }
  }

  async function downloadReport(fmt) {
    try {
      var resp = await apiFetch("/api/reports/" + encodeURIComponent(fmt));
      var blob = await resp.blob();
      var url = window.URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url;
      var ext = fmt === "docx" ? "docx" : fmt === "html" ? "html" : fmt === "cef" ? "cef" : "jsonl";
      a.download = "pokrov-report." + ext;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      window.alert("Ошибка экспорта отчёта: " + err.message);
    }
  }

  async function loadPolicy() {
    try {
      var resp = await apiFetch("/api/policy");
      var data = await resp.json();

      document.getElementById("hdr-profile").textContent = data.active_profile;
      document.getElementById("policy-hash-val").textContent = String(data.policy_hash || "").slice(0, 16) + "…";
      document.getElementById("policy-mode-note").textContent = data.demo_mode
        ? "Демо-режим / разрешено переключение профилей: доступно свободное сравнение всех профилей."
        : "Боевой режим: понижение строгости профиля (downgrade) заблокировано политикой безопасности.";
      document.getElementById("policy-yaml-view").textContent = data.policy_yaml || "";
      document.getElementById("legal-disclaimer").textContent = data.legal_disclaimer || "";

      var pgSel = document.getElementById("pg-profile");
      var prevSel = pgSel.value;
      clearNode(pgSel);
      var profiles = data.profiles || [];
      for (var i = 0; i < profiles.length; i++) {
        var opt = el("option", "", profiles[i].name);
        opt.value = profiles[i].name;
        pgSel.appendChild(opt);
      }
      pgSel.value = data.profiles.some(function (p) { return p.name === prevSel; })
        ? prevSel
        : data.active_profile;

      var grid = document.getElementById("profiles-grid");
      clearNode(grid);
      for (var j = 0; j < profiles.length; j++) {
        (function (prof) {
          var card = el("div", prof.is_active ? "profile-card active" : "profile-card");
          var head = el("div", "panel-subheader");
          head.appendChild(el("strong", "", prof.name));
          if (prof.is_active) {
            head.appendChild(el("span", "pill pill-ok", "Активен"));
          } else {
            head.appendChild(el("span", "badge-mini", "Ранг " + prof.strictness_rank));
          }
          card.appendChild(head);
          card.appendChild(el("p", "muted", prof.description || "Без описания"));

          var meta = el("div", "profile-meta");
          meta.appendChild(
            el("div", "", "Облачные LLM: " + (prof.allow_cloud_upstreams ? "разрешены (с фильтрацией)" : "запрещены (только локальный контур)"))
          );
          meta.appendChild(
            el("div", "", "Квоты: RPM=" + prof.quotas.rpm + ", max_chars=" + prof.quotas.max_prompt_chars)
          );
          meta.appendChild(
            el("div", "", "Правил категорий: " + prof.rules_count + ", кастомных: " + prof.custom_patterns_count)
          );
          card.appendChild(meta);

          var btn = el(
            "button",
            prof.is_active ? "btn btn-ghost" : "btn btn-primary",
            prof.is_active ? "Текущий профиль" : "Активировать профиль"
          );
          btn.type = "button";
          btn.disabled = prof.is_active || !prof.can_switch_to;
          if (!prof.can_switch_to && !prof.is_active) {
            btn.title = "Понижение строгости профиля запрещено в боевом режиме";
          }
          btn.addEventListener("click", function () {
            switchProfile(prof.name);
          });
          card.appendChild(btn);
          grid.appendChild(card);
        })(profiles[j]);
      }

      var upBody = document.getElementById("upstreams-body");
      clearNode(upBody);
      var upstreams = data.upstreams || [];
      for (var u = 0; u < upstreams.length; u++) {
        var up = upstreams[u];
        var tr = el("tr");
        tr.appendChild(el("td", "mono", up.name));
        tr.appendChild(el("td", "mono", up.provider));
        tr.appendChild(el("td", "mono", up.model));
        tr.appendChild(el("td", "", up.residency));

        var roles = [];
        if (up.is_default) {
          roles.push("default");
        }
        if (up.is_local_target) {
          roles.push("local_target");
        }
        if (up.is_cloud_target) {
          roles.push("cloud_target");
        }
        tr.appendChild(el("td", "mono", roles.join(", ") || "—"));

        var tdSt = el("td");
        tdSt.appendChild(
          el(
            "span",
            up.status === "ready" ? "pill pill-ok" : "pill pill-warn",
            up.status === "ready" ? "Готов" : "Нет API-ключа"
          )
        );
        tr.appendChild(tdSt);
        upBody.appendChild(tr);
      }
    } catch (_e) {
      /* handled */
    }
  }

  async function switchProfile(profileName) {
    try {
      await apiFetch("/api/policy/profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile: profileName })
      });
      await loadPolicy();
      await loadAuditStatsAndEvents();
    } catch (err) {
      window.alert("Не удалось переключить профиль: " + err.message);
    }
  }

  function setupTabs() {
    var buttons = document.querySelectorAll(".tab-btn");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].addEventListener("click", function () {
        var target = this.getAttribute("data-tab");
        for (var j = 0; j < buttons.length; j++) {
          var isAct = buttons[j].getAttribute("data-tab") === target;
          buttons[j].classList.toggle("active", isAct);
          buttons[j].setAttribute("aria-selected", isAct ? "true" : "false");
        }
        var panels = document.querySelectorAll(".tab-panel");
        for (var k = 0; k < panels.length; k++) {
          panels[k].classList.toggle("active", panels[k].id === "tab-" + target);
        }
        if (target === "audit") {
          loadAuditStatsAndEvents();
        } else if (target === "policy") {
          loadPolicy();
        }
      });
    }
  }

  function bindEvents() {
    document.getElementById("btn-theme-toggle").addEventListener("click", function () {
      var cur = document.documentElement.getAttribute("data-theme");
      applyTheme(cur === "dark" ? "light" : "dark");
    });

    document.getElementById("btn-auth-toggle").addEventListener("click", function () {
      document.getElementById("auth-bar").classList.toggle("hidden");
    });

    document.getElementById("btn-api-key-save").addEventListener("click", function () {
      setApiKey(document.getElementById("api-key-input").value.trim());
      document.getElementById("auth-bar").classList.add("hidden");
      loadPresets();
      loadPolicy();
      loadAuditStatsAndEvents();
    });

    document.getElementById("btn-api-key-clear").addEventListener("click", function () {
      setApiKey("");
      document.getElementById("api-key-input").value = "";
    });

    document.getElementById("btn-pg-run").addEventListener("click", runPlayground);
    document.getElementById("pg-input").addEventListener("keydown", function (ev) {
      if ((ev.ctrlKey || ev.metaKey) && ev.key === "Enter") {
        ev.preventDefault();
        runPlayground();
      }
    });

    document.getElementById("btn-copy-masked").addEventListener("click", function () {
      var txt = document.getElementById("res-masked").textContent || "";
      if (navigator.clipboard && txt) {
        navigator.clipboard.writeText(txt);
      }
    });

    document.getElementById("btn-copy-yaml").addEventListener("click", function () {
      var txt = document.getElementById("policy-yaml-view").textContent || "";
      if (navigator.clipboard && txt) {
        navigator.clipboard.writeText(txt);
      }
    });

    document.getElementById("btn-audit-refresh").addEventListener("click", loadAuditStatsAndEvents);
    document.getElementById("flt-action").addEventListener("change", loadAuditStatsAndEvents);
    document.getElementById("flt-category").addEventListener("change", loadAuditStatsAndEvents);

    document.getElementById("flt-autorefresh").addEventListener("change", function () {
      if (state.autoRefreshTimer !== null) {
        window.clearInterval(state.autoRefreshTimer);
        state.autoRefreshTimer = null;
      }
      if (this.checked) {
        state.autoRefreshTimer = window.setInterval(loadAuditStatsAndEvents, 5000);
      }
    });

    var exportButtons = document.querySelectorAll(".btn-export");
    for (var i = 0; i < exportButtons.length; i++) {
      exportButtons[i].addEventListener("click", function () {
        downloadReport(this.getAttribute("data-fmt"));
      });
    }
  }

  applyTheme(window.localStorage.getItem(THEME_KEY) || "dark");
  document.getElementById("api-key-input").value = getApiKey();
  setupTabs();
  bindEvents();
  loadPresets();
  loadPolicy();
  loadAuditStatsAndEvents();
})();
