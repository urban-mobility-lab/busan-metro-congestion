// API 초안(A 전달 문서) 형식을 흉내 내는 목업 서버.
// 실제 API와 같은 경로·쿼리를 받아 { status, body } 를 돌려준다.
// 시나리오를 바꾸면 template 데이터, 오류 code, 네트워크 실패, 지연을 재현할 수 있다.
(function () {
  "use strict";

  const MODEL_VERSION = "mock-day-ahead-0.1";
  const VALID_HOURS = Array.from({ length: 24 }, (_, i) => i + 1);

  // B 전달 문서의 상태 코드
  const ERROR_STATUS = {
    STATION_NOT_FOUND: 404,
    DATE_OUT_OF_RANGE: 422,
    INVALID_HOUR: 422,
    INVALID_REQUEST: 422,
    PREDICTION_NOT_READY: 503,
  };

  const ERROR_MESSAGES = {
    STATION_NOT_FOUND: "station_id에 해당하는 역이 없습니다.",
    DATE_OUT_OF_RANGE: "예측 가능 기간 밖의 날짜입니다.",
    INVALID_HOUR: "hour는 1~24 사이의 정수여야 합니다.",
    INVALID_REQUEST: "필수 파라미터가 없거나 형식이 올바르지 않습니다.",
    PREDICTION_NOT_READY: "해당 날짜의 예측이 아직 생성되지 않았습니다.",
  };

  const SCENARIOS = [
    { value: "normal", label: "정상 응답" },
    { value: "template", label: "template 데이터 (is_template=true)" },
    { value: "slow", label: "느린 응답 (2초 지연)" },
    { value: "STATION_NOT_FOUND", label: "오류: STATION_NOT_FOUND" },
    { value: "DATE_OUT_OF_RANGE", label: "오류: DATE_OUT_OF_RANGE" },
    { value: "INVALID_HOUR", label: "오류: INVALID_HOUR" },
    { value: "INVALID_REQUEST", label: "오류: INVALID_REQUEST" },
    { value: "PREDICTION_NOT_READY", label: "오류: PREDICTION_NOT_READY" },
    { value: "server_error", label: "서버 오류 (HTTP 500)" },
    { value: "network", label: "네트워크 실패 (연결 불가)" },
  ];

  // 승객 규모가 큰 역(목업 가중치). 실제 수치와 무관하다.
  const BIG_STATIONS = {
    119: 3.2, 219: 2.6, 113: 2.4, 206: 2.0, 123: 1.8, 305: 1.3, 111: 1.8,
    227: 1.8, 128: 1.7, 203: 1.6, 125: 1.3, 402: 0.9, 233: 1.4, 313: 1.0,
    208: 1.3, 301: 0.9, 122: 1.4, 102: 1.5, 134: 1.3, 314: 1.2, 307: 1.1,
  };

  function pad(n) {
    return String(n).padStart(2, "0");
  }

  function toIsoDate(d) {
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  }

  function addDays(d, n) {
    const r = new Date(d.getFullYear(), d.getMonth(), d.getDate());
    r.setDate(r.getDate() + n);
    return r;
  }

  // 전날 17시에 다음 날 예측을 만든다는 규칙을 흉내 낸다.
  function predictionWindow(now) {
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const end = now.getHours() >= 17 ? addDays(today, 1) : today;
    return {
      start: toIsoDate(addDays(today, -7)),
      end: toIsoDate(end),
      issuedAt: `${toIsoDate(addDays(end, -1))}T17:00:00+09:00`,
    };
  }

  // 시드 기반 의사 난수 (같은 역·날짜는 항상 같은 값을 준다)
  function seededRandom(seed) {
    let s = seed % 2147483647;
    if (s <= 0) s += 2147483646;
    return function () {
      s = (s * 16807) % 2147483647;
      return (s - 1) / 2147483646;
    };
  }

  // 시간대별 상대 수요 (hour=h 는 h시~h+1시, hour=24 는 다음 날 00~01시)
  const WEEKDAY_PROFILE = {
    1: 0, 2: 0, 3: 0, 4: 0, 5: 0.08, 6: 0.35, 7: 0.85, 8: 1.0, 9: 0.62,
    10: 0.45, 11: 0.47, 12: 0.52, 13: 0.53, 14: 0.52, 15: 0.56, 16: 0.64,
    17: 0.82, 18: 0.95, 19: 0.7, 20: 0.5, 21: 0.43, 22: 0.36, 23: 0.24,
    24: 0.08,
  };
  const WEEKEND_PROFILE = {
    1: 0, 2: 0, 3: 0, 4: 0, 5: 0.05, 6: 0.15, 7: 0.3, 8: 0.42, 9: 0.5,
    10: 0.62, 11: 0.72, 12: 0.8, 13: 0.85, 14: 0.88, 15: 0.9, 16: 0.88,
    17: 0.85, 18: 0.8, 19: 0.7, 20: 0.55, 21: 0.45, 22: 0.36, 23: 0.24,
    24: 0.08,
  };

  function buildPredictions(station, dateStr, hours, isTemplate) {
    const date = new Date(`${dateStr}T00:00:00`);
    const dow = date.getDay();
    const weekend = dow === 0 || dow === 6;
    const profile = weekend ? WEEKEND_PROFILE : WEEKDAY_PROFILE;
    const scale = 1400 * (BIG_STATIONS[station.station_id] || 0.6 + (station.station_id % 7) / 10);
    const rand = seededRandom(station.station_id * 100003 + Number(dateStr.replace(/-/g, "")) % 100000);

    return hours.map((h) => {
      const typical = Math.round(profile[h] * scale);
      if (isTemplate) {
        return { hour: h, predicted_passengers: Math.round(rand() * 3000), typical_passengers: typical };
      }
      const noise = 0.85 + rand() * 0.3;
      return { hour: h, predicted_passengers: Math.round(typical * noise), typical_passengers: typical };
    });
  }

  function error(code) {
    return { status: ERROR_STATUS[code], body: { code, message: ERROR_MESSAGES[code] } };
  }

  function handle(path, params, scenario, now) {
    const win = predictionWindow(now);

    if (path === "/api/v1/health") {
      return { status: 200, body: { status: "ok" } };
    }

    if (path === "/api/v1/model-info") {
      return {
        status: 200,
        body: {
          model_version: MODEL_VERSION,
          prediction_start: win.start,
          prediction_end: win.end,
          generated_at: win.issuedAt,
          target: "역의 시간당 승차+하차 합계 인원",
        },
      };
    }

    if (path === "/api/v1/stations") {
      return { status: 200, body: window.MOCK_STATIONS.slice() };
    }

    if (path === "/api/v1/predictions") {
      const forced = ERROR_MESSAGES[scenario] ? scenario : null;
      if (forced) {
        return error(forced);
      }

      const stationId = params.get("station_id");
      const dateStr = params.get("date");
      if (!stationId || !dateStr || !/^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
        return error("INVALID_REQUEST");
      }
      const station = window.MOCK_STATIONS.find((s) => String(s.station_id) === stationId);
      if (!station) return error("STATION_NOT_FOUND");
      if (dateStr < win.start || dateStr > win.end) return error("DATE_OUT_OF_RANGE");

      const rawHours = params.getAll("hour");
      let hours = VALID_HOURS;
      if (rawHours.length > 0) {
        hours = rawHours.map(Number);
        if (hours.some((h) => !Number.isInteger(h) || h < 1 || h > 24)) {
          return error("INVALID_HOUR");
        }
        hours = Array.from(new Set(hours)).sort((a, b) => a - b);
      }

      const isTemplate = scenario === "template";
      return {
        status: 200,
        body: {
          station: {
            station_id: station.station_id,
            station_name: station.station_name,
            line_num: station.line_num,
          },
          date: dateStr,
          model_version: MODEL_VERSION,
          is_template: isTemplate,
          generated_at: `${toIsoDate(addDays(new Date(`${dateStr}T00:00:00`), -1))}T17:00:00+09:00`,
          predictions: buildPredictions(station, dateStr, hours, isTemplate),
        },
      };
    }

    return error("INVALID_REQUEST");
  }

  // fetch 와 같은 모양의 응답(Promise<{ ok, status, json() }>)을 돌려준다.
  function mockFetch(url, scenario) {
    const u = new URL(url, "http://mock.local");
    const delay = scenario === "slow" ? 2000 : 250;
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        if (scenario === "network") {
          reject(new TypeError("Failed to fetch (목업 네트워크 실패)"));
          return;
        }
        if (scenario === "server_error" && u.pathname === "/api/v1/predictions") {
          resolve({ ok: false, status: 500, json: () => Promise.reject(new SyntaxError("not json")) });
          return;
        }
        const res = handle(u.pathname, u.searchParams, scenario, new Date());
        resolve({
          ok: res.status >= 200 && res.status < 300,
          status: res.status,
          json: () => Promise.resolve(JSON.parse(JSON.stringify(res.body))),
        });
      }, delay);
    });
  }

  window.MockApi = { fetch: mockFetch, SCENARIOS };
})();
