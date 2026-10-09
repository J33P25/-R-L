// State variables
let autoRunInterval = null;
let currentAgent = "PPO";
let telemetryChart = null;
let comparisonChart = null;
let currentComparisonMetric = "power_kw";
let benchmarkDataCache = null;

// Telemetry History Arrays for Chart
const maxHistoryLength = 25;
const chartLabels = [];
const dataIndoorTemp = [];
const dataOutdoorTemp = [];
const dataCO2 = [];

// DOM Elements
const elIndoorTemp = document.getElementById("val-indoor-temp");
const elCO2 = document.getElementById("val-co2");
const elHumidity = document.getElementById("val-humidity");
const elOccupants = document.getElementById("val-occupants");
const elOutdoorTemp = document.getElementById("val-outdoor-temp");
const elOccStatus = document.getElementById("label-occ-status");
const elWeatherDetail = document.getElementById("label-weather-detail");

const elPctValve = document.getElementById("pct-valve");
const elBarValve = document.getElementById("bar-valve");
const elPctFan = document.getElementById("pct-fan");
const elBarFan = document.getElementById("bar-fan");
const elPctDamper = document.getElementById("pct-damper");
const elBarDamper = document.getElementById("bar-damper");
const elFanIcon = document.getElementById("fan-icon");

const elEnergyTotal = document.getElementById("val-energy-total");
const elPowerKw = document.getElementById("val-power-kw");
const elAgentBadge = document.getElementById("nav-active-agent-badge");
const agentSelector = document.getElementById("agent-selector");

const btnStep = document.getElementById("btn-step");
const btnAutoRun = document.getElementById("btn-autorun");
const btnReset = document.getElementById("btn-reset");
const btnInjectPerturb = document.getElementById("btn-inject-perturb");

const sliderHeatwave = document.getElementById("slider-heatwave");
const labelHeatwave = document.getElementById("label-heatwave-val");
const sliderCrowd = document.getElementById("slider-crowd");
const labelCrowd = document.getElementById("label-crowd-val");

const btnChartEnergy = document.getElementById("btn-chart-energy");
const btnChartTemp = document.getElementById("btn-chart-temp");
const btnChartCO2 = document.getElementById("btn-chart-co2");

// Initialize Model-Based vs Model-Free Comparison Chart
function initComparisonChart() {
  const ctx = document.getElementById("comparisonChart").getContext("2d");
  const timeHours = Array.from({ length: 96 }, (_, i) => `${(i * 0.25).toFixed(1)}h`);

  comparisonChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: timeHours,
      datasets: [
        {
          label: "Model-Free: PPO (Continuous ⭐)",
          data: [],
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.1)",
          borderWidth: 2.5,
          tension: 0.35,
          fill: true
        },
        {
          label: "Model-Based: Policy Iteration (MDP)",
          data: [],
          borderColor: "#8b5cf6",
          borderDash: [6, 4],
          borderWidth: 2.0,
          tension: 0.2
        },
        {
          label: "Baseline: Rule-Based (RBC)",
          data: [],
          borderColor: "#94a3b8",
          borderDash: [4, 4],
          borderWidth: 1.8,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          labels: { color: "#9ca3af", font: { family: "Inter", size: 11 } }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#6b7280", maxTicksLimit: 12 }
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#9ca3af" }
        }
      }
    }
  });
}

function updateComparisonChart(metric) {
  if (!benchmarkDataCache || !comparisonChart) return;
  currentComparisonMetric = metric;

  const rbcData = benchmarkDataCache.results["Rule-Based"]?.history?.[metric] || [];
  const piData = benchmarkDataCache.results["Policy Iteration"]?.history?.[metric] || [];
  const ppoData = benchmarkDataCache.results["PPO"]?.history?.[metric] || [];

  comparisonChart.data.datasets[0].data = ppoData;
  comparisonChart.data.datasets[1].data = piData;
  comparisonChart.data.datasets[2].data = rbcData;

  const units = metric === "power_kw" ? " kW" : metric === "indoor_temp" ? " °C" : " ppm";
  comparisonChart.options.scales.y.ticks.callback = (val) => val + units;
  comparisonChart.update();

  // Update button active states
  [btnChartEnergy, btnChartTemp, btnChartCO2].forEach(b => {
    b.style.background = "rgba(255, 255, 255, 0.05)";
    b.style.color = "#9ca3af";
  });
  if (metric === "power_kw") {
    btnChartEnergy.style.background = "rgba(16, 185, 129, 0.2)";
    btnChartEnergy.style.color = "#34d399";
  } else if (metric === "indoor_temp") {
    btnChartTemp.style.background = "rgba(16, 185, 129, 0.2)";
    btnChartTemp.style.color = "#34d399";
  } else if (metric === "co2_ppm") {
    btnChartCO2.style.background = "rgba(16, 185, 129, 0.2)";
    btnChartCO2.style.color = "#34d399";
  }
}

// Update Active MDP State Highlight (S1 to S8)
function updateStatePills(stateIndex) {
  for (let i = 1; i <= 8; i++) {
    const pill = document.getElementById(`pill-s${i}`);
    if (pill) {
      pill.classList.remove("active", "critical-active");
    }
  }

  const activePill = document.getElementById(`pill-s${stateIndex + 1}`);
  if (activePill) {
    if (stateIndex === 4 || stateIndex === 6) {
      // Critical (S5) or Safety Violation (S7)
      activePill.classList.add("critical-active");
    } else {
      activePill.classList.add("active");
    }
  }
}

// Update UI with latest Telemetry Data
function renderTelemetry(data) {
  elIndoorTemp.innerHTML = `${data.indoor_temp.toFixed(1)}<span>°C</span>`;
  elCO2.innerHTML = `${Math.round(data.co2_ppm)}<span>ppm</span>`;
  elHumidity.innerHTML = `${data.indoor_humidity.toFixed(1)}<span>%</span>`;
  elOccupants.innerHTML = `${data.occupants}<span>people</span>`;
  elOutdoorTemp.innerHTML = `${data.outdoor_temp.toFixed(1)}<span>°C</span>`;

  elOccStatus.innerText = data.occupants > 0 ? `Occupied (${data.occupants} active)` : "Unoccupied (Standby)";
  if (data.solar_kw !== undefined) {
    elWeatherDetail.innerText = `Solar PV Output: ${data.solar_kw.toFixed(1)} kW`;
  }

  // Actuator Bars
  const valvePct = Math.round((data.valve || 0) * 100);
  const fanPct = Math.round((data.fan || 0) * 100);
  const damperPct = Math.round((data.damper || 0) * 100);

  elPctValve.innerText = `${valvePct}%`;
  elBarValve.style.width = `${valvePct}%`;

  elPctFan.innerText = `${fanPct}%`;
  elBarFan.style.width = `${fanPct}%`;

  elPctDamper.innerText = `${damperPct}%`;
  elBarDamper.style.width = `${damperPct}%`;

  // Dynamically speed up/slow down fan spin animation
  if (fanPct > 0) {
    const duration = Math.max(0.3, 1.8 - (fanPct / 100) * 1.5);
    elFanIcon.style.animationDuration = `${duration}s`;
    elFanIcon.style.display = "inline-block";
  } else {
    elFanIcon.style.display = "none";
  }

  // Energy & Power
  if (data.total_energy_kwh !== undefined) {
    elEnergyTotal.innerText = `${data.total_energy_kwh.toFixed(2)} kWh`;
  }
  if (data.power_kw !== undefined) {
    elPowerKw.innerText = `${data.power_kw.toFixed(2)} kW`;
  }

  // MDP State Pill
  updateStatePills(data.discrete_state || 0);

  // Push to chart
  const stepLabel = data.step ? `Step ${data.step}` : `T+${chartLabels.length * 15}m`;
  chartLabels.push(stepLabel);
  dataIndoorTemp.push(data.indoor_temp);
  dataOutdoorTemp.push(data.outdoor_temp);
  dataCO2.push(data.co2_ppm);

  if (chartLabels.length > maxHistoryLength) {
    chartLabels.shift();
    dataIndoorTemp.shift();
    dataOutdoorTemp.shift();
    dataCO2.shift();
  }

  if (telemetryChart) {
    telemetryChart.update("none");
  }
}

// Default fallback state if server is not yet running
let localSimState = {
  step: 12,
  indoor_temp: 22.8,
  indoor_humidity: 49.5,
  co2_ppm: 680,
  occupants: 4,
  outdoor_temp: 29.8,
  outdoor_humidity: 58.0,
  solar_kw: 3.4,
  discrete_state: 0,
  valve: 0.38,
  fan: 0.52,
  damper: 0.28,
  total_energy_kwh: 5.42,
  power_kw: 2.15
};

// API Calls with Graceful Fallback
async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    if (res.ok) {
      const data = await res.json();
      renderTelemetry(data);
      return;
    }
  } catch (err) {
    // Standalone browser mode fallback
  }
  renderTelemetry(localSimState);
}

async function stepSimulation() {
  try {
    const res = await fetch("/api/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent: currentAgent, steps: 1 })
    });
    if (res.ok) {
      const data = await res.json();
      renderTelemetry(data);
      return;
    }
  } catch (err) {
    // Local fallback step calculation
    localSimState.step += 1;
    localSimState.total_energy_kwh += (localSimState.power_kw * 0.25);

    // Simulate smart RL thermostat reaction
    if (currentAgent === "PPO") {
      localSimState.indoor_temp += (Math.random() * 0.2 - 0.1);
      localSimState.indoor_temp = Math.max(21.2, Math.min(23.8, localSimState.indoor_temp));
      localSimState.co2_ppm = Math.max(500, Math.min(920, localSimState.co2_ppm + (localSimState.occupants > 0 ? 15 : -30)));
      localSimState.valve = 0.35 + (localSimState.indoor_temp - 22.0) * 0.2;
      localSimState.fan = 0.45 + (localSimState.indoor_temp - 22.0) * 0.15;
      localSimState.damper = (localSimState.co2_ppm > 800) ? 0.45 : 0.25;
      localSimState.power_kw = 1.8 + localSimState.valve * 1.5;
      localSimState.discrete_state = (localSimState.co2_ppm > 1000) ? 3 : (localSimState.indoor_temp > 24) ? 1 : 0;
    } else {
      // RBC has higher oscillations
      localSimState.indoor_temp += (Math.random() * 0.6 - 0.3);
      localSimState.power_kw = 2.6;
      localSimState.discrete_state = (localSimState.indoor_temp > 24) ? 1 : 0;
    }
    renderTelemetry(localSimState);
  }
}

async function injectPerturbation() {
  const outdoor = parseFloat(sliderHeatwave.value);
  const occ = parseInt(sliderCrowd.value);

  try {
    const res = await fetch("/api/perturb", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ outdoor_temp: outdoor, occupants: occ })
    });
    if (res.ok) {
      const data = await res.json();
      renderTelemetry(data);
      return;
    }
  } catch (err) {
    // Local fallback perturbation
    localSimState.outdoor_temp = outdoor;
    localSimState.occupants = occ;
    localSimState.indoor_temp = Math.min(30.0, localSimState.indoor_temp + 1.8);
    localSimState.co2_ppm += (occ * 40);
    localSimState.discrete_state = (localSimState.indoor_temp > 24 && localSimState.co2_ppm > 1000) ? 4 : (localSimState.indoor_temp > 24) ? 1 : 3;
    renderTelemetry(localSimState);
  }
}

async function resetSimulation() {
  try {
    const res = await fetch("/api/reset", { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      chartLabels.length = 0;
      dataIndoorTemp.length = 0;
      dataOutdoorTemp.length = 0;
      dataCO2.length = 0;
      renderTelemetry(data);
      return;
    }
  } catch (err) {
    localSimState.step = 0;
    localSimState.total_energy_kwh = 0.0;
    localSimState.indoor_temp = 22.5;
    localSimState.co2_ppm = 650;
    localSimState.discrete_state = 0;
    chartLabels.length = 0;
    dataIndoorTemp.length = 0;
    dataOutdoorTemp.length = 0;
    dataCO2.length = 0;
    renderTelemetry(localSimState);
  }
}

// Event Listeners
agentSelector.addEventListener("change", (e) => {
  currentAgent = e.target.value;
  elAgentBadge.innerText = `Controller: ${currentAgent}`;
});

btnStep.addEventListener("click", () => {
  stepSimulation();
});

btnAutoRun.addEventListener("click", () => {
  if (autoRunInterval) {
    clearInterval(autoRunInterval);
    autoRunInterval = null;
    btnAutoRun.innerText = "⚡ Auto-Run";
    btnAutoRun.classList.remove("btn-primary");
    btnAutoRun.classList.add("btn-secondary");
  } else {
    autoRunInterval = setInterval(stepSimulation, 850);
    btnAutoRun.innerText = "⏸ Pause Run";
    btnAutoRun.classList.remove("btn-secondary");
    btnAutoRun.classList.add("btn-primary");
  }
});

btnReset.addEventListener("click", resetSimulation);
btnInjectPerturb.addEventListener("click", injectPerturbation);

sliderHeatwave.addEventListener("input", (e) => {
  labelHeatwave.innerText = `${parseFloat(e.target.value).toFixed(1)}°C`;
});

sliderCrowd.addEventListener("input", (e) => {
  labelCrowd.innerText = `${e.target.value} people`;
});

// Initialization on DOM Load
window.addEventListener("DOMContentLoaded", () => {
  initChart();
  fetchStatus();
});
