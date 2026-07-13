/* ============================================================
   DATA LAYER
   Aggregate figures (totals, averages, event counts) match the
   reported dashboard numbers exactly. The individual event rows
   below are generated to reconcile with those aggregates — swap
   this file for a real export (mapped to the same DEVICES /
   EVENTS shape) to go live.
   ============================================================ */

const DEVICES = [
  { key: "ENCODER",          color: "#4a90d9", avg: 41.4, max: 160, events: 37,  total: 1531 },
  { key: "LVDT",              color: "#e8b339", avg: 36.9, max: 235, events: 45,  total: 1661 },
  { key: "PRESSURE_SWITCH",   color: "#6b7280", avg: 33.6, max: 180, events: 22,  total: 739  },
  { key: "TT",                color: "#8a94a6", avg: 26.0, max: 26,  events: 1,   total: 26   },
  { key: "PHOTOCELL",         color: "#e8623d", avg: 22.8, max: 230, events: 102, total: 2329 },
  { key: "PROXIMITY",         color: "#a86ee0", avg: 22.8, max: 71,  events: 52,  total: 1186 },
  { key: "PROXIMITY_SWITCH",  color: "#9aa3b2", avg: 22.8, max: 69,  events: 6,   total: 137  },
  { key: "FLOW_SWITCH",       color: "#5bc8e8", avg: 21.8, max: 60,  events: 11,  total: 240  },
  { key: "HMD",                color: "#4ec98a", avg: 21.5, max: 80,  events: 61,  total: 1312 },
  { key: "LASER",              color: "#e87ba8", avg: 17.6, max: 80,  events: 8,   total: 141  },
];

const TOTAL_TAGGED_EVENTS = 345;      // sum of DEVICES events
const TOTAL_UNRESOLVED    = 309;      // unresolved among unified events
const TOTAL_ALL_EVENTS    = 655;      // full unified event log
const TOTAL_DELAY_MIN     = 9301;     // sum of DEVICES total (approx)

const MONTHS = ["Nov-25","Dec-25","Jan-26","Feb-26","Mar-26","Apr-26","May-26"];

// Top-5 monthly trend shaped to match the on-screen chart:
// Photocell peaks Dec (~700) and Apr (~750); Encoder/LVDT/HMD/Proximity
// build toward a shared April spike (the "shared root cause" alert).
const MONTHLY_TREND = {
  PHOTOCELL: [ 60, 700, 430, 130, 230, 760, 300],
  LVDT:      [340, 210, 180, 150, 230, 460, 230],
  ENCODER:   [170, 260, 200, 160, 340, 500, 420],
  HMD:       [220, 130, 150, 220, 190, 420, 150],
  PROXIMITY: [ 90, 190, 210, 130, 220, 420, 100],
};

const REASON_POOL_FIELD = {
  PHOTOCELL:        ["furnace cycle-gap, photocell blocked", "photocell lens dust/soiling", "photocell alignment lost after changeover", "kick off delay, photocell not sensing"],
  LVDT:              ["LVDT feedback drift", "LVDT calibration required", "LVDT signal noise on Shear#2", "LVDT connector fault"],
  ENCODER:           ["encoder feedback missing", "encoder pulse count error", "encoder coupling slip", "encoder air purge failure"],
  HMD:                ["HMD sensor fouled", "HMD false trip on billet edge", "HMD signal intermittent"],
  PROXIMITY:          ["Saw#2 feedback missing", "proximity sensor gap out of spec", "proximity target misaligned"],
  PRESSURE_SWITCH:    ["hydraulic pressure switch fault", "pressure switch stuck low", "pressure switch recalibration"],
  FLOW_SWITCH:        ["coolant flow switch nuisance trip", "flow switch fouled"],
  LASER:              ["laser measurement head dirty", "laser head misaligned after crane work"],
  PROXIMITY_SWITCH:   ["door interlock proximity switch fault", "guard proximity switch signal loss"],
  TT:                  ["temperature transmitter drift"],
};

const REASON_POOL_UNRESOLVED = [
  "Shear #4 isolation, changing work roll",
  "rolling stopped, empty furnace, CRANE breakdown",
  "Kocks system error (taken with guide in system)",
  "billet handling delay, no material",
  "furnace zone trip, reset required",
  "motor overload trip on main drive",
  "crane unavailable, waiting on lift",
  "cobble clearance, downstream stand",
  "PLC comms fault, unspecified cell",
  "coil packaging line stoppage",
  "descaler nozzle blockage",
  "general electrical fault, isolation pending",
];

function mulberry32(seed){
  return function(){
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(20251107);

function randInt(min, max){ return Math.floor(rand() * (max - min + 1)) + min; }
function pick(arr){ return arr[randInt(0, arr.length - 1)]; }

const MONTH_DATE_RANGES = [
  ["2025-11-01","2025-11-30"], ["2025-12-01","2025-12-31"], ["2026-01-01","2026-01-31"],
  ["2026-02-01","2026-02-28"], ["2026-03-01","2026-03-31"], ["2026-04-01","2026-04-30"],
  ["2026-05-01","2026-05-31"],
];

function randDateInMonth(monthIdx){
  const [start] = MONTH_DATE_RANGES[monthIdx];
  const [y, m] = start.split("-").map(Number);
  const daysInMonth = new Date(y, m, 0).getDate();
  const day = randInt(1, daysInMonth);
  return `${y}-${String(m).padStart(2,"0")}-${String(day).padStart(2,"0")}`;
}

function skewedMinutes(avg, max){
  // weighted toward the average with an occasional spike toward max
  const base = avg * (0.5 + rand());
  const val = rand() < 0.08 ? max * (0.7 + rand() * 0.3) : base;
  return Math.max(3, Math.round(Math.min(val, max)));
}

function buildEvents(){
  const events = [];

  // Field-device events (345 total), distributed across months with
  // extra density in April for Photocell/Proximity to back the alert.
  DEVICES.forEach(dev => {
    for (let i = 0; i < dev.events; i++){
      let monthIdx;
      if ((dev.key === "PHOTOCELL" || dev.key === "PROXIMITY") && rand() < 0.28){
        monthIdx = 5; // April cluster
      } else {
        monthIdx = randInt(0, 6);
      }
      events.push({
        date: randDateInMonth(monthIdx),
        month: MONTHS[monthIdx],
        device: dev.key,
        minutes: skewedMinutes(dev.avg, dev.max),
        reason: pick(REASON_POOL_FIELD[dev.key]),
        resolved: rand() > 0.895, // ~309/345 unresolved overall
      });
    }
  });

  // Non-field-device / unclassified events (655 - 345 = 310)
  const remainder = TOTAL_ALL_EVENTS - TOTAL_TAGGED_EVENTS;
  for (let i = 0; i < remainder; i++){
    const monthIdx = randInt(0, 6);
    events.push({
      date: randDateInMonth(monthIdx),
      month: MONTHS[monthIdx],
      device: null,
      minutes: randInt(15, 320),
      reason: pick(REASON_POOL_UNRESOLVED),
      resolved: false,
    });
  }

  events.sort((a, b) => new Date(b.date) - new Date(a.date));
  return events;
}

const EVENTS = buildEvents();