(function () {
  "use strict";

  // ---------------------------------------------------------------------------
  // 설정: config.js + 주소창 ?api=... / ?mock=1 덮어쓰기
  // useMock: "auto" 이면 init() 에서 /api/v1/health 응답 여부로 정한다.
  // ---------------------------------------------------------------------------
  const config = Object.assign({ useMock: "auto", apiBaseUrl: "" }, window.BUSAN_METRO_CONFIG);
  const urlParams = new URLSearchParams(location.search);
  if (urlParams.get("api")) {
    config.useMock = false;
    config.apiBaseUrl = urlParams.get("api");
  } else if (urlParams.get("mock") === "1") {
    config.useMock = true;
  }

  async function resolveMode() {
    if (config.useMock !== "auto") return;
    const base = config.apiBaseUrl.replace(/\/+$/, "");
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 1500);
    try {
      const res = await fetch(`${base}/api/v1/health`, { signal: controller.signal });
      config.useMock = !res.ok;
    } catch (e) {
      config.useMock = true;
    } finally {
      clearTimeout(timer);
    }
  }

  const TRANSFER_NAMES = new Set(["서면", "연산", "덕천", "수영", "미남", "동래"]);

  const ERROR_TEXT = {
    STATION_NOT_FOUND: "선택한 역을 찾을 수 없습니다. 페이지를 새로고침해 역 목록을 다시 불러온 뒤 선택해 주세요.",
    DATE_OUT_OF_RANGE: "예측 가능 기간 밖의 날짜입니다. 기간 안의 날짜를 골라 주세요.",
    INVALID_HOUR: "요청한 시간대가 올바르지 않습니다. 시간대는 1~24(24는 다음 날 00~01시)만 가능합니다.",
    INVALID_REQUEST: "요청 형식이 올바르지 않습니다. 호선·역·날짜를 다시 선택해 주세요.",
    PREDICTION_NOT_READY: "이 날짜의 예측이 아직 만들어지지 않았습니다. 예측은 전날 17시에 생성됩니다.",
    NETWORK: "예측 서버에 연결할 수 없습니다. 네트워크 상태나 API 주소를 확인해 주세요.",
    SERVER: "예측 서버에서 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
    UNKNOWN: "알 수 없는 오류가 발생했습니다.",
  };

  // ---------------------------------------------------------------------------
  // API 클라이언트
  // ---------------------------------------------------------------------------
  class ApiError extends Error {
    constructor(code, message, status) {
      super(message || code);
      this.code = code;
      this.status = status;
    }
  }

  function getScenario() {
    const el = document.getElementById("mock-scenario");
    return el && el.value ? el.value : "normal";
  }

  async function apiGet(path, params) {
    const qs = new URLSearchParams();
    for (const [key, value] of params || []) qs.append(key, value);
    const query = qs.toString();
    const base = config.apiBaseUrl.replace(/\/+$/, "");
    const url = `${base}${path}${query ? `?${query}` : ""}`;

    let res;
    try {
      res = config.useMock ? await window.MockApi.fetch(url, getScenario()) : await fetch(url);
    } catch (e) {
      throw new ApiError("NETWORK", e.message);
    }

    let body = null;
    try {
      body = await res.json();
    } catch (e) {
      body = null;
    }

    if (!res.ok) {
      if (body && typeof body.code === "string") throw new ApiError(body.code, body.message, res.status);
      throw new ApiError(res.status >= 500 ? "SERVER" : "UNKNOWN", `HTTP ${res.status}`, res.status);
    }
    if (body === null) throw new ApiError("SERVER", "응답이 JSON이 아닙니다.", res.status);
    return body;
  }

  // 응답 정규화: API 초안이 바뀌어도 이 두 함수만 고치면 된다.
  function normalizeStations(body) {
    const list = Array.isArray(body) ? body : body.stations || [];
    return list.map((s) => ({
      station_id: Number(s.station_id),
      station_name: String(s.station_name),
      line_num: Number(s.line_num),
    }));
  }

  function normalizePrediction(body) {
    const rows = body.predictions || body.hours || [];
    const station = body.station || {
      station_id: body.station_id,
      station_name: body.station_name,
      line_num: body.line_num,
    };
    return {
      station,
      date: body.date,
      model_version: body.model_version,
      is_template: body.is_template === true,
      generated_at: body.generated_at,
      rows: rows
        .map((r) => ({
          hour: Number(r.hour),
          predicted: r.predicted_passengers == null ? null : Number(r.predicted_passengers),
          typical: r.typical_passengers == null ? null : Number(r.typical_passengers),
        }))
        .sort((a, b) => a.hour - b.hour),
    };
  }

  // ---------------------------------------------------------------------------
  // 표시 도우미
  // ---------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const fmt = new Intl.NumberFormat("ko-KR");
  const pad = (n) => String(n).padStart(2, "0");

  // hour=h 는 h시~h+1시, hour=24 는 date 다음 날 00~01시
  function hourLabel(h, long) {
    const start = h % 24;
    const label = `${pad(start)}~${pad((start + 1) % 24 || 24)}시`;
    if (h === 24) return long ? "00~01시 (다음 날)" : "00~01시+1";
    return label;
  }

  function formatDate(dateStr) {
    const d = new Date(`${dateStr}T00:00:00`);
    const w = "일월화수목금토"[d.getDay()];
    return `${dateStr} (${w})`;
  }

  function formatDateTime(iso) {
    if (!iso) return "-";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }

  function showStatus(kind, html) {
    const el = $("status");
    el.className = `status ${kind}`;
    el.innerHTML = html;
    el.hidden = false;
  }

  function hideStatus() {
    $("status").hidden = true;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }

  function showError(err, context) {
    const code = err instanceof ApiError ? err.code : "UNKNOWN";
    let text = ERROR_TEXT[code] || ERROR_TEXT.UNKNOWN;
    if (code === "DATE_OUT_OF_RANGE" && hasRange(state.modelInfo)) {
      text += ` (예측 가능 기간: ${state.modelInfo.prediction_start} ~ ${state.modelInfo.prediction_end})`;
    }
    const detail = err.message && err.message !== code ? ` · ${escapeHtml(err.message)}` : "";
    showStatus(
      "error",
      `<strong>${escapeHtml(context)}</strong> ${escapeHtml(text)}<br><span class="code">code: ${escapeHtml(code)}${detail}</span>`,
    );
  }

  // ---------------------------------------------------------------------------
  // 상태와 화면
  // ---------------------------------------------------------------------------
  const state = { stations: [], modelInfo: null, chart: null, requestSeq: 0 };

  function fillLines() {
    const lines = Array.from(new Set(state.stations.map((s) => s.line_num))).sort((a, b) => a - b);
    const sel = $("line-select");
    sel.innerHTML = `<option value="">호선 선택</option>` + lines.map((n) => `<option value="${n}">${n}호선</option>`).join("");
    sel.disabled = false;
  }

  function fillStations(lineNum) {
    const sel = $("station-select");
    if (!lineNum) {
      sel.innerHTML = `<option value="">호선을 먼저 고르세요</option>`;
      sel.disabled = true;
      return;
    }
    const list = state.stations.filter((s) => s.line_num === Number(lineNum)).sort((a, b) => a.station_id - b.station_id);
    sel.innerHTML =
      `<option value="">역 선택 (${list.length}개)</option>` +
      list
        .map((s) => {
          const mark = TRANSFER_NAMES.has(s.station_name) ? " · 환승" : "";
          return `<option value="${s.station_id}">${escapeHtml(s.station_name)} (${s.station_id})${mark}</option>`;
        })
        .join("");
    sel.disabled = false;
  }

  function hasRange(info) {
    return Boolean(info && info.prediction_start && info.prediction_end);
  }

  // 예측 가능 기간이 있으면 그 안에서만 고르게 하고, 없으면 내일을 기본값으로 둔다.
  function setupDate(info) {
    const input = $("date-input");
    if (hasRange(info)) {
      input.min = info.prediction_start;
      input.max = info.prediction_end;
      input.value = info.prediction_end;
      $("date-hint").textContent = `예측 가능: ${info.prediction_start} ~ ${info.prediction_end}`;
    } else {
      const t = new Date();
      t.setDate(t.getDate() + 1);
      input.value = `${t.getFullYear()}-${pad(t.getMonth() + 1)}-${pad(t.getDate())}`;
      $("date-hint").textContent = "예측 가능 기간 정보가 없습니다.";
    }
    input.disabled = false;
  }

  function updateSubmitState() {
    $("submit-btn").disabled = !($("station-select").value && $("date-input").value);
  }

  // 한산한 시간은 운행 시간대(예측값과 과거 대표값이 모두 0보다 큰 시간) 중에서 고른다.
  function pickExtremes(rows) {
    const valid = rows.filter((r) => r.predicted != null);
    if (valid.length === 0) return { peak: null, quiet: null };
    const operating = valid.filter((r) => r.predicted > 0 && (r.typical == null || r.typical > 0));
    const pool = operating.length > 0 ? operating : valid;
    const peak = valid.reduce((a, b) => (b.predicted > a.predicted ? b : a));
    const quiet = pool.reduce((a, b) => (b.predicted < a.predicted ? b : a));
    return { peak, quiet };
  }

  function renderResult(pred) {
    const station = state.stations.find((s) => s.station_id === Number(pred.station.station_id)) || pred.station;
    const name = pred.station.station_name || station.station_name;
    const line = pred.station.line_num || station.line_num;

    $("template-banner").hidden = !pred.is_template;
    $("result-title").textContent = `${name}역 (${line}호선 · ${pred.station.station_id}) · ${formatDate(pred.date)}`;
    $("result-meta").textContent = `모델 ${pred.model_version || "-"} · 생성 ${formatDateTime(pred.generated_at)}${
      config.useMock ? " · 목업 데이터" : ""
    }`;

    const transferNote = $("transfer-note");
    if (TRANSFER_NAMES.has(name)) {
      const others = state.stations.filter((s) => s.station_name === name && s.line_num !== Number(line));
      transferNote.textContent = `환승역입니다. 이 값은 ${line}호선 역(${pred.station.station_id})만의 인원입니다. ${others
        .map((s) => `${s.line_num}호선(${s.station_id})`)
        .join(", ")}은 따로 조회하세요.`;
      transferNote.hidden = false;
    } else {
      transferNote.hidden = true;
    }

    const { peak, quiet } = pickExtremes(pred.rows);
    $("peak-hour").textContent = peak ? hourLabel(peak.hour, true) : "-";
    $("peak-value").textContent = peak ? `${fmt.format(peak.predicted)}명 예측` : "";
    $("quiet-hour").textContent = quiet ? hourLabel(quiet.hour, true) : "-";
    $("quiet-value").textContent = quiet ? `${fmt.format(quiet.predicted)}명 예측` : "";

    const total = pred.rows.reduce((s, r) => s + (r.predicted || 0), 0);
    const typicalTotal = pred.rows.reduce((s, r) => s + (r.typical || 0), 0);
    $("total-value").textContent = `${fmt.format(total)}명`;
    $("total-sub").textContent = typicalTotal ? `과거 대표값 합계 ${fmt.format(typicalTotal)}명` : "";

    renderChart(pred.rows, peak, quiet, pred.is_template);
    renderTable(pred.rows, peak, quiet);
    $("result").hidden = false;
  }

  function renderChart(rows, peak, quiet, isTemplate) {
    const css = getComputedStyle(document.documentElement);
    const color = (v) => css.getPropertyValue(v).trim();
    const barColors = rows.map((r) => {
      if (peak && r.hour === peak.hour) return color("--peak");
      if (quiet && r.hour === quiet.hour) return color("--quiet");
      return color("--bar");
    });

    const data = {
      labels: rows.map((r) => hourLabel(r.hour, false)),
      datasets: [
        {
          type: "bar",
          label: isTemplate ? "예측 인원 (template·임의값)" : "예측 인원",
          data: rows.map((r) => r.predicted),
          backgroundColor: barColors,
          borderRadius: 3,
          order: 2,
        },
        {
          type: "line",
          label: "과거 대표값",
          data: rows.map((r) => r.typical),
          borderColor: color("--line"),
          backgroundColor: color("--line"),
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.25,
          order: 1,
        },
      ],
    };

    if (typeof window.Chart !== "function") {
      showStatus("error", "그래프 라이브러리(Chart.js)를 불러오지 못했습니다. 아래 표에서 수치를 확인하세요.");
      $("result").querySelector(".table-wrap").open = true;
      return;
    }

    if (state.chart) state.chart.destroy();
    state.chart = new window.Chart($("chart"), {
      data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          y: { beginAtZero: true, title: { display: true, text: "시간당 승하차 인원(명)" } },
          x: { title: { display: true, text: "시간대" } },
        },
        plugins: {
          tooltip: {
            callbacks: {
              title: (items) => hourLabel(rows[items[0].dataIndex].hour, true),
              label: (item) => `${item.dataset.label}: ${item.raw == null ? "-" : fmt.format(item.raw)}명`,
            },
          },
        },
      },
    });
  }

  function renderTable(rows, peak, quiet) {
    $("table-body").innerHTML = rows
      .map((r) => {
        const cls = peak && r.hour === peak.hour ? "peak-row" : quiet && r.hour === quiet.hour ? "quiet-row" : "";
        const diff = r.predicted != null && r.typical != null ? r.predicted - r.typical : null;
        const diffText = diff == null ? "-" : `${diff > 0 ? "+" : ""}${fmt.format(diff)}`;
        return `<tr class="${cls}"><td>${hourLabel(r.hour, true)}</td><td>${
          r.predicted == null ? "-" : fmt.format(r.predicted)
        }</td><td>${r.typical == null ? "-" : fmt.format(r.typical)}</td><td>${diffText}</td></tr>`;
      })
      .join("");
  }

  // ---------------------------------------------------------------------------
  // 이벤트
  // ---------------------------------------------------------------------------
  async function query(event) {
    event.preventDefault();
    const stationId = $("station-select").value;
    const date = $("date-input").value;
    if (!stationId || !date) return;

    const seq = ++state.requestSeq;
    $("submit-btn").disabled = true;
    $("result").hidden = true;
    showStatus("loading", "예측을 불러오는 중…");
    try {
      const body = await apiGet("/api/v1/predictions", [
        ["station_id", stationId],
        ["date", date],
      ]);
      if (seq !== state.requestSeq) return;
      hideStatus();
      renderResult(normalizePrediction(body));
    } catch (err) {
      if (seq !== state.requestSeq) return;
      showError(err, "예측을 불러오지 못했습니다.");
    } finally {
      if (seq === state.requestSeq) updateSubmitState();
    }
  }

  async function init() {
    await resolveMode();
    if (config.useMock) {
      const panel = $("mock-panel");
      $("mock-scenario").innerHTML = window.MockApi.SCENARIOS.map(
        (s) => `<option value="${s.value}">${escapeHtml(s.label)}</option>`,
      ).join("");
      panel.hidden = false;
    }

    $("line-select").addEventListener("change", (e) => {
      fillStations(e.target.value);
      updateSubmitState();
    });
    $("station-select").addEventListener("change", updateSubmitState);
    $("date-input").addEventListener("input", updateSubmitState);
    $("query-form").addEventListener("submit", query);

    showStatus("loading", "역 목록과 모델 정보를 불러오는 중…");
    try {
      const [infoBody, stationsBody] = await Promise.all([apiGet("/api/v1/model-info"), apiGet("/api/v1/stations")]);
      state.modelInfo = infoBody;
      state.stations = normalizeStations(stationsBody);
      const range = hasRange(infoBody) ? ` · 예측 가능 기간 ${infoBody.prediction_start} ~ ${infoBody.prediction_end}` : "";
      $("model-info").textContent = `모델 ${infoBody.model_version || "-"}${range}${
        config.useMock ? " · 목업 모드" : ` · API ${config.apiBaseUrl || "(같은 출처)"}`
      }`;
      fillLines();
      setupDate(infoBody);
      hideStatus();
    } catch (err) {
      $("model-info").textContent = "모델 정보를 불러오지 못했습니다.";
      showError(err, "초기 정보를 불러오지 못했습니다.");
    }
  }

  // 경로 조회(route.js)가 같은 API 클라이언트와 표시 도우미를 쓰도록 공개한다.
  window.BusanMetroApp = {
    ready: init(),
    state,
    config,
    apiGet,
    normalizePrediction,
    ApiError,
    ERROR_TEXT,
    escapeHtml,
    hourLabel,
    formatDate,
    fmt,
    pad,
    TRANSFER_NAMES,
  };
})();
