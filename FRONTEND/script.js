/* ============================================================
   RENDER LOGIC
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  renderTicker();
  renderDelayShareChart();
  renderSeverityTable();
  renderMonthlyTrendChart();
  initEventLog();
});

/* ---------- Ticker ---------- */
function renderTicker(){
  const items = [
    "April cluster identified as shared root cause",
    "PHOTOCELL: highest total volume — 2,329 min / 102 events",
    "ENCODER: FMEA RPN > 200 and above-average real-world severity — top evidence-based priority",
    "no FMEA sheet on file despite being top-5 delay contributors: PHOTOCELL, PROXIMITY",
    "47% of events resolved to a specific device — 47% remain non-field-device or unclassified",
  ];
  const el = document.getElementById("ticker");
  el.innerHTML = items.map(t => `<span class="dot">●</span> ${t} &nbsp;&nbsp;&nbsp; `).join("");
  el.innerHTML += el.innerHTML; // duplicate for seamless loop
}

/* ---------- Delay Share bar chart ---------- */
function renderDelayShareChart(){
  const sorted = [...DEVICES].sort((a, b) => b.total - a.total);
  const ctx = document.getElementById("delayShareChart");

  new Chart(ctx, {
    type: "bar",
    data: {
      labels: sorted.map(d => d.key),
      datasets: [{
        data: sorted.map(d => d.total),
        backgroundColor: sorted.map(d => d.color),
        borderRadius: 4,
        maxBarThickness: 34,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.parsed.x.toLocaleString()} min`
          }
        }
      },
      scales: {
        x: {
          grid: { color: "#1a2233" },
          ticks: { color: "#8b96a8", font: { family: "monospace", size: 11 } },
        },
        y: {
          grid: { display: false },
          ticks: { color: "#cfd6e2", font: { family: "monospace", size: 12, weight: "600" } },
        }
      }
    }
  });
}

/* ---------- Severity vs Frequency table ---------- */
function renderSeverityTable(){
  const sorted = [...DEVICES].sort((a, b) => b.avg - a.avg);
  const maxAvg = Math.max(...DEVICES.map(d => d.avg));
  const tbody = document.getElementById("sevTableBody");

  tbody.innerHTML = sorted.map(d => `
    <tr>
      <td class="device-name" style="color:${d.color}">${d.key}</td>
      <td>${d.avg.toFixed(1)}</td>
      <td>
        <div class="bar-track">
          <div class="bar-fill" style="width:${(d.avg / maxAvg * 100).toFixed(0)}%; background:${d.color}"></div>
        </div>
      </td>
      <td>${d.max} min</td>
      <td>${d.events}</td>
    </tr>
  `).join("");
}

/* ---------- Monthly trend line chart ---------- */
function renderMonthlyTrendChart(){
  const ctx = document.getElementById("monthlyTrendChart");
  const keys = Object.keys(MONTHLY_TREND);

  new Chart(ctx, {
    type: "line",
    data: {
      labels: MONTHS,
      datasets: keys.map(key => {
        const dev = DEVICES.find(d => d.key === key);
        return {
          label: key,
          data: MONTHLY_TREND[key],
          borderColor: dev.color,
          backgroundColor: dev.color,
          pointRadius: 4,
          pointHoverRadius: 6,
          tension: 0.35,
          borderWidth: 2,
        };
      })
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          grid: { color: "#1a2233" },
          ticks: { color: "#8b96a8", font: { family: "monospace", size: 11 } },
        },
        y: {
          grid: { color: "#1a2233" },
          ticks: { color: "#8b96a8", font: { family: "monospace", size: 11 } },
          beginAtZero: true,
        }
      }
    }
  });

  const legendEl = document.getElementById("trendLegend");
  legendEl.innerHTML = keys.map(key => {
    const dev = DEVICES.find(d => d.key === key);
    return `<span class="legend-item"><span class="legend-swatch" style="background:${dev.color}"></span>${key}</span>`;
  }).join("");
}

/* ---------- Event Log ---------- */
const PAGE_SIZE = 12;
let state = {
  search: "",
  device: "",
  month: "",
  sortKey: "date",
  sortDir: "desc",
  page: 1,
};

function initEventLog(){
  const deviceFilter = document.getElementById("deviceFilter");
  DEVICES.forEach(d => {
    const opt = document.createElement("option");
    opt.value = d.key;
    opt.textContent = d.key;
    deviceFilter.appendChild(opt);
  });
  const unresolvedOpt = document.createElement("option");
  unresolvedOpt.value = "__unresolved__";
  unresolvedOpt.textContent = "Unresolved / unclassified";
  deviceFilter.appendChild(unresolvedOpt);

  const monthFilter = document.getElementById("monthFilter");
  MONTHS.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    monthFilter.appendChild(opt);
  });

  document.getElementById("searchInput").addEventListener("input", (e) => {
    state.search = e.target.value.toLowerCase();
    state.page = 1;
    renderEventTable();
  });
  deviceFilter.addEventListener("change", (e) => {
    state.device = e.target.value;
    state.page = 1;
    renderEventTable();
  });
  monthFilter.addEventListener("change", (e) => {
    state.month = e.target.value;
    state.page = 1;
    renderEventTable();
  });

  document.querySelectorAll(".event-table thead th").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      if (state.sortKey === key){
        state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
      } else {
        state.sortKey = key;
        state.sortDir = "desc";
      }
      renderEventTable();
    });
  });

  renderEventTable();
}

function getFilteredEvents(){
  return EVENTS.filter(ev => {
    if (state.search && !ev.reason.toLowerCase().includes(state.search)) return false;
    if (state.month && ev.month !== state.month) return false;
    if (state.device === "__unresolved__" && ev.device !== null) return false;
    if (state.device && state.device !== "__unresolved__" && ev.device !== state.device) return false;
    return true;
  });
}

function sortEvents(list){
  const { sortKey, sortDir } = state;
  const dir = sortDir === "asc" ? 1 : -1;
  return [...list].sort((a, b) => {
    let av = a[sortKey], bv = b[sortKey];
    if (sortKey === "date") { av = new Date(a.date); bv = new Date(b.date); }
    if (sortKey === "device") { av = av || "zzz"; bv = bv || "zzz"; }
    if (av < bv) return -1 * dir;
    if (av > bv) return 1 * dir;
    return 0;
  });
}

function minutesClass(min){
  if (min >= 150) return "minutes-high";
  if (min >= 60) return "minutes-mid";
  return "minutes-low";
}

function renderEventTable(){
  const filtered = sortEvents(getFilteredEvents());
  document.getElementById("eventCount").textContent = `${filtered.length} events`;

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  state.page = Math.min(state.page, totalPages);
  const start = (state.page - 1) * PAGE_SIZE;
  const pageItems = filtered.slice(start, start + PAGE_SIZE);

  const tbody = document.getElementById("eventTableBody");
  tbody.innerHTML = pageItems.map(ev => `
    <tr>
      <td>${ev.date}</td>
      <td>${ev.month}</td>
      <td>${ev.device
        ? `<span class="device-tag" style="color:${DEVICES.find(d=>d.key===ev.device).color}">${ev.device}</span>`
        : `<span class="device-tag">Unresolved</span>`}</td>
      <td class="minutes-cell ${minutesClass(ev.minutes)}">${ev.minutes} min</td>
      <td class="reason-cell">${ev.reason}</td>
    </tr>
  `).join("");

  renderPagination(totalPages);
}

function renderPagination(totalPages){
  const el = document.getElementById("pagination");
  el.innerHTML = `
    <button id="prevPage" ${state.page <= 1 ? "disabled" : ""}>← Prev</button>
    <span>Page ${state.page} of ${totalPages}</span>
    <button id="nextPage" ${state.page >= totalPages ? "disabled" : ""}>Next →</button>
  `;
  document.getElementById("prevPage").addEventListener("click", () => {
    state.page--; renderEventTable();
  });
  document.getElementById("nextPage").addEventListener("click", () => {
    state.page++; renderEventTable();
  });
}