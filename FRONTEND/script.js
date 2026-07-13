const API_BASE = "";

// ==================== CHARTS ====================
let delayShareChart = null;
let monthlyTrendChart = null;

async function loadCharts() {
    try {
        const res = await fetch(`${API_BASE}/chart-data`);
        const data = await res.json();

        // Delay Share Chart
        const delayCtx = document.getElementById("delayShareChart");
        if (delayShareChart) delayShareChart.destroy();

        delayShareChart = new Chart(delayCtx, {
            type: "bar",
            data: {
                labels: data.delay_share.labels,
                datasets: [{
                    label: "Total Minutes",
                    data: data.delay_share.values,
                    backgroundColor: ["#f97316", "#eab308", "#3b82f6", "#22c55e", "#a855f7", "#64748b"]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, grid: { color: "#334155" } },
                    x: { grid: { color: "#334155" } }
                }
            }
        });

        // Monthly Trend Chart
        const trendCtx = document.getElementById("monthlyTrendChart");
        if (monthlyTrendChart) monthlyTrendChart.destroy();

        monthlyTrendChart = new Chart(trendCtx, {
            type: "line",
            data: {
                labels: data.monthly_trend.labels,
                datasets: [{
                    label: "Total Delay Minutes",
                    data: data.monthly_trend.values,
                    borderColor: "#f97316",
                    backgroundColor: "rgba(249, 115, 22, 0.1)",
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, grid: { color: "#334155" } },
                    x: { grid: { color: "#334155" } }
                }
            }
        });
    } catch (error) {
        console.error("Chart loading failed:", error);
    }
}

// ==================== EVENT LOG ====================
let allEvents = [];

async function loadEventLog() {
    try {
        const res = await fetch(`${API_BASE}/events`);
        allEvents = await res.json();
        renderEventTable(allEvents);
        populateDeviceFilter();
    } catch (error) {
        console.error("Failed to load events");
    }
}

function renderEventTable(events) {
    const tbody = document.getElementById("eventTableBody");
    tbody.innerHTML = "";

    if (events.length === 0) {
        tbody.innerHTML = `<tr><td colspan="4" class="px-4 py-6 text-center text-slate-400">No events found</td></tr>`;
        return;
    }

    events.forEach(event => {
        const row = document.createElement("tr");
        row.className = "border-b border-slate-700 hover:bg-slate-800";
        row.innerHTML = `
            <td class="px-4 py-3 text-sm">${event.date || '—'}</td>
            <td class="px-4 py-3">
                <span class="px-3 py-1 text-xs rounded-full bg-slate-700">${event.field_device || 'Unresolved'}</span>
            </td>
            <td class="px-4 py-3 font-semibold text-orange-400">${event.mins || 0} min</td>
            <td class="px-4 py-3 text-sm text-slate-300">${event.reason_text || ''}</td>
        `;
        tbody.appendChild(row);
    });
}

function filterEvents() {
    const search = document.getElementById("searchInput").value.toLowerCase();
    const device = document.getElementById("deviceFilter").value;

    const filtered = allEvents.filter(e => {
        const matchSearch = !search || (e.reason_text && e.reason_text.toLowerCase().includes(search));
        const matchDevice = !device || e.field_device === device;
        return matchSearch && matchDevice;
    });

    renderEventTable(filtered);
}

function populateDeviceFilter() {
    const select = document.getElementById("deviceFilter");
    const devices = [...new Set(allEvents.map(e => e.field_device).filter(Boolean))];
    
    devices.sort().forEach(device => {
        const option = document.createElement("option");
        option.value = device;
        option.textContent = device;
        select.appendChild(option);
    });
}

// ==================== PREDICT MODAL ====================
function showPredictModal() {
    const modal = document.getElementById("predictModal");
    modal.classList.remove("hidden");
    modal.classList.add("flex");
    document.getElementById("modalPredictionResult").innerHTML = "";
    document.getElementById("modalDelayText").focus();
}

function closePredictModal() {
    const modal = document.getElementById("predictModal");
    modal.classList.remove("flex");
    modal.classList.add("hidden");
}

async function runPrediction() {
    const text = document.getElementById("modalDelayText").value.trim();
    if (!text) return alert("Please enter a delay description");

    const resultDiv = document.getElementById("modalPredictionResult");
    resultDiv.innerHTML = `<p class="text-center text-slate-400 py-4">Analyzing...</p>`;

    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ delay_text: text })
        });

        const data = await res.json();

        let html = `
            <div class="bg-slate-800 p-5 rounded-2xl">
                <div class="mb-4">
                    <div class="text-sm text-slate-400">Top Prediction</div>
                    <div class="text-2xl font-bold">${data.predictions[0].device}</div>
                    <div class="text-lg">${data.predictions[0].probability}% confidence</div>
                </div>
        `;

        if (data.low_confidence) {
            html += `<div class="bg-red-900/30 border border-red-500 text-red-400 p-3 rounded-xl mb-4 text-sm">
                ⚠ Low Confidence — Manual verification recommended
            </div>`;
        }

        html += `
            <div class="grid grid-cols-2 gap-4 text-sm">
                <div>
                    <span class="text-slate-400">FMEA RPN:</span><br>
                    <span class="font-semibold">${data.fmea_risk.rpn}</span>
                </div>
                <div>
                    <span class="text-slate-400">Combined Risk:</span><br>
                    <span class="font-semibold">${data.combined_risk.level} (${data.combined_risk.score}%)</span>
                </div>
            </div>
        </div>`;

        resultDiv.innerHTML = html;

    } catch (error) {
        resultDiv.innerHTML = `<p class="text-red-400">Error: Could not get prediction</p>`;
    }
}

// ==================== INITIALIZE ====================
async function initializeDashboard() {
    await loadEventLog();
    await loadCharts();
}

window.onload = initializeDashboard;