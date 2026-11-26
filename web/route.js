// 경로 조회: 출발역에서 출발 시각 이후 가장 빨리 도착역에 닿는 경로(환승 2회까지)를 찾아
// 각 역의 도착 시각과 그 시간대의 예측 인원을 보여준다.
// 계산 규칙은 busan_metro_congestion/service/route_timeline.py(plan_trip, plan_journey)와 같다.
//   - 한 호선 구간은 같은 호선 시각표 안에서 같은 열차번호를 따라간다(열차번호는 호선끼리 겹칠 수 있다).
//   - 출발역보다 늦게 도착역에 닿는 열차만 고르므로 방향과 단축 운행이 자동으로 걸러진다.
//   - 환승역에서는 환승 도보 시간이 지난 뒤 처음 오는 열차를 탄다.
//   - 도착 시각이 가장 이른 경로를 고르고, 같으면 환승이 적은 경로를 고른다.
//   - 운행일은 03시에 시작하고, 00~01시 도착은 hour=24(다음 날 00~01시)로 본다.
(function () {
  "use strict";

  const app = window.BusanMetroApp;
  const timetable = window.TRAIN_TIMETABLE;
  const $ = (id) => document.getElementById(id);
  const DAY_LABELS = { 1: "평일", 2: "토요일", 3: "일요일·공휴일" };
  const SERVICE_DAY_START = 3 * 3600;
  const MAX_TRANSFERS = 2;
  const holidays = new Set((timetable && timetable.holidays) || []);
  const walks = new Map(((timetable && timetable.transfers) || []).map(([a, b, s]) => [`${a}-${b}`, s]));

  // ---------------------------------------------------------------------------
  // 경로 계산
  // ---------------------------------------------------------------------------
  function dayTypeFor(dateStr) {
    const dow = new Date(`${dateStr}T00:00:00`).getDay();
    if (holidays.has(dateStr) || dow === 0) return 3;
    if (dow === 6) return 2;
    return 1;
  }

  function clockToSeconds(hhmm) {
    const [h, m] = hhmm.split(":").map(Number);
    const sec = h * 3600 + m * 60;
    return sec < SERVICE_DAY_START ? sec + 24 * 3600 : sec;
  }

  function secondsToClock(sec) {
    const s = sec % (24 * 3600);
    return `${app.pad(Math.floor(s / 3600))}:${app.pad(Math.floor((s % 3600) / 60))}`;
  }

  function hourBucket(sec) {
    const h = Math.floor(sec / 3600) % 24;
    return h === 0 ? 24 : h;
  }

  function lineOf(stationId) {
    const s = app.state.stations.find((x) => x.station_id === stationId);
    return s ? s.line_num : null;
  }

  // 한 호선 구간: 출발 시각 이후 출발역에 처음 오는 열차 중 도착역까지 가는 열차
  function planTrip(dayType, originId, destId, departSec) {
    const line = lineOf(originId);
    if (line === null || line !== lineOf(destId)) return null;
    const trains = (timetable.lines[String(line)] || {})[String(dayType)] || {};
    let best = null;
    for (const [trainNo, stops] of Object.entries(trains)) {
      const o = stops.find((s) => s[0] === originId);
      const d = stops.find((s) => s[0] === destId);
      if (!o || !d || o[1] < departSec || d[1] <= o[1]) continue;
      if (!best || o[1] < best.depart) best = { trainNo, depart: o[1], arrive: d[1], stops };
    }
    if (!best) return null;
    return {
      line,
      trainNo: best.trainNo,
      stops: best.stops
        .filter((s) => s[1] >= best.depart && s[1] <= best.arrive)
        .sort((a, b) => a[1] - b[1])
        .map(([stationId, sec]) => ({ stationId, sec, clock: secondsToClock(sec), hour: hourBucket(sec) })),
    };
  }

  // (승차역, 하차역) 구간 목록으로 된 경로 후보. 지난 호선은 다시 타지 않고 도착역 호선에 오르면 멈춘다.
  function candidateRoutes(originId, destId) {
    const routes = [];
    const destLine = lineOf(destId);
    const edges = Array.from(walks.keys()).map((k) => k.split("-").map(Number));
    function extend(board, segments, used) {
      const line = lineOf(board);
      if (line === destLine) {
        if (board !== destId) routes.push([...segments, [board, destId]]);
        return;
      }
      if (segments.length >= MAX_TRANSFERS) return;
      for (const [src, dst] of edges) {
        if (lineOf(src) !== line || src === board || used.has(lineOf(dst))) continue;
        extend(dst, [...segments, [board, src]], new Set([...used, lineOf(dst)]));
      }
    }
    if (lineOf(originId) !== null && destLine !== null) extend(originId, [], new Set([lineOf(originId)]));
    return routes;
  }

  function planJourney(dayType, originIds, destIds, departSec) {
    let best = null;
    for (const o of originIds) {
      for (const d of destIds) {
        if (o === d) continue;
        for (const route of candidateRoutes(o, d)) {
          const legs = [];
          let ready = departSec;
          let ok = true;
          for (let i = 0; i < route.length; i++) {
            const [board, alight] = route[i];
            let walk = null;
            if (i > 0) {
              walk = walks.get(`${route[i - 1][1]}-${board}`);
              const prev = legs[legs.length - 1].stops;
              ready = prev[prev.length - 1].sec + walk;
            }
            const leg = planTrip(dayType, board, alight, ready);
            if (!leg) {
              ok = false;
              break;
            }
            leg.walk = walk;
            legs.push(leg);
          }
          if (!ok) continue;
          const last = legs[legs.length - 1].stops;
          const score = [last[last.length - 1].sec, legs.length];
          if (!best || score[0] < best.score[0] || (score[0] === best.score[0] && score[1] < best.score[1])) {
            best = { score, legs };
          }
        }
      }
    }
    return best ? best.legs : null;
  }

  // ---------------------------------------------------------------------------
  // 화면
  // ---------------------------------------------------------------------------
  function showStatus(kind, html) {
    const el = $("route-status");
    el.className = `status ${kind}`;
    el.innerHTML = html;
    el.hidden = false;
  }

  function stationName(id) {
    const s = app.state.stations.find((x) => x.station_id === id);
    return s ? s.station_name : String(id);
  }

  function idsForName(name) {
    return app.state.stations.filter((s) => s.station_name === name).map((s) => s.station_id);
  }

  // 역 이름 하나에 옵션 하나. 환승역은 호선을 모두 표시하고 계산할 때 빠른 호선을 고른다.
  function fillStations() {
    const byName = new Map();
    for (const s of app.state.stations) {
      if (!byName.has(s.station_name)) byName.set(s.station_name, { first: s.station_id, lines: [] });
      const entry = byName.get(s.station_name);
      entry.lines.push(s.line_num);
      entry.first = Math.min(entry.first, s.station_id);
    }
    const lines = Array.from(new Set(app.state.stations.map((s) => s.line_num))).sort((a, b) => a - b);
    const html = lines
      .map((line) => {
        const names = Array.from(byName.entries())
          .filter(([, e]) => Math.min(...e.lines) === line)
          .sort((a, b) => a[1].first - b[1].first);
        const options = names
          .map(([name, e]) => {
            const label = e.lines.length > 1 ? `${name} (${e.lines.sort().join("·")}호선 환승)` : name;
            return `<option value="${app.escapeHtml(name)}">${app.escapeHtml(label)}</option>`;
          })
          .join("");
        return `<optgroup label="${line}호선">${options}</optgroup>`;
      })
      .join("");
    for (const id of ["route-from", "route-to"]) {
      $(id).innerHTML = `<option value="">${id === "route-from" ? "출발역" : "도착역"} 선택</option>${html}`;
      $(id).disabled = false;
    }
  }

  function updateSubmit() {
    const ok = ["route-from", "route-to", "route-date", "route-time"].every((id) => $(id).value);
    $("route-submit").disabled = !ok || $("route-from").value === $("route-to").value;
  }

  async function fetchStopValues(stationIds, dateStr) {
    const results = await Promise.all(
      stationIds.map((id) =>
        app
          .apiGet("/api/v1/predictions", [
            ["station_id", id],
            ["date", dateStr],
          ])
          .then((body) => [id, app.normalizePrediction(body)]),
      ),
    );
    return new Map(results);
  }

  function stopRow(stop, values) {
    const pred = values.get(stop.stationId);
    const row = pred ? pred.rows.find((r) => r.hour === stop.hour) : null;
    return { ...stop, name: stationName(stop.stationId), predicted: row ? row.predicted : null, typical: row ? row.typical : null };
  }

  function renderTimeline(legs, values, dateStr, dayType) {
    const legRows = legs.map((leg) => leg.stops.map((s) => stopRow(s, values)));
    const all = legRows.flat();
    const max = Math.max(1, ...all.map((r) => r.predicted || 0));
    const peak = all.reduce((a, b) => ((b.predicted || 0) > (a.predicted || 0) ? b : a), all[0]);
    const first = all[0];
    const last = all[all.length - 1];
    const minutes = Math.floor((last.sec - first.sec) / 60);
    const transfers = legs.length - 1;

    $("route-template-banner").hidden = !Array.from(values.values()).some((p) => p.is_template);
    $("route-title").textContent = `${first.name} → ${last.name} · ${app.formatDate(dateStr)}`;
    $("route-meta").textContent =
      `${DAY_LABELS[dayType]} 시각표 · ${first.clock} 출발 · ${last.clock} 도착 · 약 ${minutes}분 · ` +
      `${transfers === 0 ? "환승 없음" : `환승 ${transfers}회`}` +
      (app.config.useMock ? " · 인원은 목업 데이터" : "");

    const html = [];
    legs.forEach((leg, i) => {
      const rows = legRows[i];
      if (i > 0) {
        const prev = legRows[i - 1][legRows[i - 1].length - 1];
        const walkMin = Math.floor((leg.walk || 0) / 60);
        const waitMin = Math.max(0, Math.floor((rows[0].sec - prev.sec - (leg.walk || 0)) / 60));
        html.push(
          `<li class="tl-transfer">↪ <strong>${app.escapeHtml(prev.name)}</strong>에서 ${leg.line}호선으로 환승 · 도보 ${walkMin}분 + 대기 ${waitMin}분</li>`,
        );
      }
      html.push(`<li class="tl-leg">${leg.line}호선 · ${app.escapeHtml(leg.trainNo)}번 열차</li>`);
      rows.forEach((r, j) => {
        const isFirst = i === 0 && j === 0;
        const isLast = i === legs.length - 1 && j === rows.length - 1;
        const role = isFirst ? "출발" : isLast ? "도착" : j === rows.length - 1 ? "하차" : j === 0 ? "승차" : "";
        const width = r.predicted == null ? 0 : Math.round((r.predicted / max) * 100);
        const isPeak = r === peak && r.predicted != null;
        const value =
          r.predicted == null
            ? `<span class="tl-missing">예측 없음</span>`
            : `<strong>${app.fmt.format(r.predicted)}명</strong>` +
              (r.typical == null ? "" : ` <span class="tl-typical">(평소 ${app.fmt.format(r.typical)}명)</span>`);
        html.push(`<li class="tl-stop line-${leg.line}${isPeak ? " tl-peak" : ""}">
          <span class="tl-time">${r.clock}</span>
          <span class="tl-dot" aria-hidden="true"></span>
          <span class="tl-name">${app.escapeHtml(r.name)}${role ? ` <em>${role}</em>` : ""}
            <span class="tl-hour">${app.hourLabel(r.hour, true)}</span></span>
          <span class="tl-value">${value}<span class="tl-bar"><span style="width:${width}%"></span></span></span>
        </li>`);
      });
    });
    $("route-timeline").innerHTML = html.join("");
    $("route-result").hidden = false;
  }

  async function query(event) {
    event.preventDefault();
    const dateStr = $("route-date").value;
    const dayType = dayTypeFor(dateStr);
    const originIds = idsForName($("route-from").value);
    const destIds = idsForName($("route-to").value);
    $("route-result").hidden = true;

    const legs = planJourney(dayType, originIds, destIds, clockToSeconds($("route-time").value));
    if (!legs) {
      showStatus(
        "error",
        `<strong>경로를 찾지 못했습니다.</strong> ${DAY_LABELS[dayType]} 시각표에 이 시각 이후 도착역까지 가는 열차가 없습니다.`,
      );
      return;
    }
    showStatus("loading", "역별 예측을 불러오는 중…");
    $("route-submit").disabled = true;
    try {
      const ids = Array.from(new Set(legs.flatMap((leg) => leg.stops.map((s) => s.stationId))));
      const values = await fetchStopValues(ids, dateStr);
      $("route-status").hidden = true;
      renderTimeline(legs, values, dateStr, dayType);
    } catch (err) {
      const code = err instanceof app.ApiError ? err.code : "UNKNOWN";
      showStatus(
        "error",
        `<strong>예측을 불러오지 못했습니다.</strong> ${app.escapeHtml(app.ERROR_TEXT[code] || app.ERROR_TEXT.UNKNOWN)}<br><span class="code">code: ${app.escapeHtml(code)}</span>`,
      );
    } finally {
      updateSubmit();
    }
  }

  function switchTab(route) {
    $("station-view").hidden = route;
    $("route-view").hidden = !route;
    $("tab-station").classList.toggle("active", !route);
    $("tab-route").classList.toggle("active", route);
    $("tab-station").setAttribute("aria-selected", String(!route));
    $("tab-route").setAttribute("aria-selected", String(route));
  }

  async function init() {
    $("tab-station").addEventListener("click", () => switchTab(false));
    $("tab-route").addEventListener("click", () => switchTab(true));
    if (!timetable || !timetable.lines) {
      showStatus("error", "열차시각표(web/data/timetable.js)가 없습니다. scripts/export_web_timetable.py를 먼저 실행하세요.");
      return;
    }
    await app.ready;
    if (!app.state.stations.length) return;
    fillStations();
    const dateInput = $("route-date");
    const main = $("date-input");
    dateInput.min = main.min;
    dateInput.max = main.max;
    dateInput.value = main.value;
    dateInput.disabled = false;
    $("route-time").disabled = false;

    for (const id of ["route-from", "route-to", "route-date", "route-time"]) {
      $(id).addEventListener("input", updateSubmit);
      $(id).addEventListener("change", updateSubmit);
    }
    $("route-form").addEventListener("submit", query);
  }

  init();
})();
