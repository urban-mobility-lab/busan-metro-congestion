// 웹 설정.
//   useMock: "auto" → apiBaseUrl 의 /api/v1/health 가 응답하면 실제 API, 아니면 web/mock/ 예시 데이터
//   useMock: true   → 항상 예시 데이터
//   useMock: false  → 항상 실제 API
//   apiBaseUrl: ""  → 같은 출처(API 서버가 web/ 을 함께 제공할 때). 다른 서버면 "http://localhost:8000" 처럼 적는다.
// 주소창 쿼리로도 덮어쓸 수 있다: ?api=http://localhost:8000 (실제 API), ?mock=1 (예시 데이터)
window.BUSAN_METRO_CONFIG = {
  useMock: "auto",
  apiBaseUrl: "",
};
