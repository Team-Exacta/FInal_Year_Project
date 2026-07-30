/* ==========================================================================
   Trip Planner
   Renders the MOIP optimiser output in the same shape as the standalone
   Streamlit app (src/app.py): best route, day-by-day plans, score tables,
   per-day coloured map, route options 1-5, and evaluation & accuracy.
   ========================================================================== */

let globalItineraryMap = null;
const dayLayers = [];

// Day 1 = red, Day 2 = blue, Day 3 = green, Day 4 = purple, ... (matches app.py)
const DAY_COLORS = ['#e53935', '#1e88e5', '#2e7d32', '#8e24aa', '#f9a825', '#00897b', '#d81b60'];
const dayColor = d => DAY_COLORS[(d - 1) % DAY_COLORS.length];

/* ---------------------------------------------------------------- utils */
function esc(str) {
    return String(str ?? '').replace(/[&<>"']/g, c => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
}

function el(id) { return document.getElementById(id); }

function renderTable(rows) {
    if (!rows || !rows.length) return '<p class="muted">No data.</p>';
    const cols = Object.keys(rows[0]);
    return `<table class="score-table">
        <thead><tr>${cols.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead>
        <tbody>${rows.map(r => `<tr>${cols.map(c => {
            const v = r[c];
            const isNum = typeof v === 'number';
            return `<td class="${isNum ? 'num' : ''}">${esc(isNum ? v : (v ?? ''))}</td>`;
        }).join('')}</tr>`).join('')}</tbody>
    </table>`;
}

/* -------------------------------------------------- day-by-day rendering */
function renderDayByDay(days) {
    if (!days || !days.length) return '<p class="muted">No day plan.</p>';
    return days.map(d => {
        const status = d.within_limit ? 'within limit' : 'exceeds daily limit';
        return `<div class="day-block">
            <div class="day-head">
                <span class="day-swatch" style="background:${dayColor(d.day)}"></span>
                <b>Day ${d.day} – Estimated time: ${d.time_h} h, distance: ${d.distance_km} km
                (${status})</b>
            </div>
            ${d.connected_from
                ? `<p class="day-note">connected from previous day: ${esc(d.connected_from)}</p>` : ''}
            <p class="day-places">${esc(d.places_summary)}</p>
            ${d.route_line ? `<p class="day-note">route - ${esc(d.route_line)}</p>` : ''}
        </div>`;
    }).join('');
}

/* ------------------------------------------------------------- accuracy */
function renderAccuracy(acc) {
    if (!acc || (acc.top1 === null && acc.score_gap === null)) {
        return `<p class="muted">Accuracy needs a small fixed must-visit list (≤ 7 intermediate
                places) so every order can be checked.</p>`;
    }

    const gap = acc.score_gap;
    const metrics = [
        ['Top-1', acc.top1 ? 'Yes' : 'No'],
        ['Top-3 hit', acc.top3_hit ? 'Yes' : 'No'],
        ['Score gap', gap != null ? (gap * 100).toFixed(2) + '%' : '—'],
        ['ACO rank', acc.aco_rank ?? '—'],
    ];

    let html = `<p class="muted">Accuracy vs true best order (exhaustive check of all place
        sequences). Lower score gap is better; Top-1 = Yes means ACO found an optimal order.</p>
        <div class="metric-row">
            ${metrics.map(([k, v]) => `<div class="metric">
                <div class="metric-label">${esc(k)}</div>
                <div class="metric-value">${esc(v)}</div>
            </div>`).join('')}
        </div>`;

    if (acc.true_best_route && acc.true_best_route.length) {
        html += `<p class="day-note">True best route (exhaustive): ${esc(acc.true_best_route.join(' → '))}</p>`;
    }
    if (acc.true_best_score != null) {
        html += `<p class="day-note">True best score: ${esc(acc.true_best_score)} ·
                 ACO fair score: ${esc(acc.aco_fair_score ?? '—')}</p>`;
    }
    if (acc.comparison_table && acc.comparison_table.length) {
        html += `<div class="table-wrap" style="margin-top:14px">${renderTable(acc.comparison_table)}</div>`;
    }
    return html;
}

/* ------------------------------------------------------------------ map */
async function drawDayRoute(coords, color) {
    if (coords.length < 2) return;
    const coordStr = coords.map(c => `${c[1]},${c[0]}`).join(';');
    const url = `https://router.project-osrm.org/route/v1/driving/${coordStr}?overview=full&geometries=geojson`;
    try {
        const res = await fetch(url);
        const data = await res.json();
        if (data.routes && data.routes.length) {
            const layer = L.geoJSON(data.routes[0].geometry, {
                style: { color, weight: 5, opacity: 0.85 }
            }).addTo(globalItineraryMap);
            dayLayers.push(layer);
            return;
        }
    } catch (e) {
        console.error('OSRM routing failed:', e);
    }
    // fallback: straight dashed line
    const layer = L.polyline(coords, { color, weight: 4, dashArray: '5, 10' }).addTo(globalItineraryMap);
    dayLayers.push(layer);
}

function buildMap(days) {
    const container = el('itinerary-map');
    if (globalItineraryMap) {
        globalItineraryMap.remove();
        globalItineraryMap = null;
    }
    dayLayers.length = 0;

    globalItineraryMap = L.map(container);
    L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> · ' +
            'SRTM | &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (CC-BY-SA)',
        maxZoom: 17
    }).addTo(globalItineraryMap);

    globalItineraryMap.invalidateSize();

    const allCoords = [];
    // chain days so the drawn line continues from the previous day's last stop
    let prevLast = null;

    days.forEach(day => {
        const color = dayColor(day.day);
        const coords = (day.detailed_places || [])
            .filter(p => p.lat != null && p.lon != null)
            .map(p => [p.lat, p.lon]);
        if (!coords.length) return;

        allCoords.push(...coords);
        const legCoords = prevLast ? [prevLast, ...coords] : coords;
        drawDayRoute(legCoords, color);
        prevLast = coords[coords.length - 1];

        (day.detailed_places || []).forEach((p, idx) => {
            if (p.lat == null || p.lon == null) return;
            const icon = L.divIcon({
                className: '',
                html: `<div class="poi-marker" style="background:${color};color:#fff;font-weight:600">
                         <span>${idx + 1}</span></div>`,
                iconSize: [30, 30],
                iconAnchor: [15, 30],
                popupAnchor: [0, -28]
            });
            L.marker([p.lat, p.lon], { icon })
                .addTo(globalItineraryMap)
                .bindPopup(`<b>Day ${day.day} · Stop ${idx + 1}</b><br>${esc(p.label || p.name)}`);
        });
    });

    if (allCoords.length) {
        globalItineraryMap.fitBounds(L.latLngBounds(allCoords), { padding: [30, 30] });
    } else {
        globalItineraryMap.setView([7.6, 80.75], 7.4);
    }
}

/* --------------------------------------------------------- main render */
function renderPlanner(data) {
    const p = data.planner;
    el('results').style.display = 'block';

    // AI summary (RAG). Hidden when the summary could not be produced.
    const summary = (data.response || '').trim();
    if (summary) {
        el('itinerary-summary').innerHTML =
            typeof marked !== 'undefined' ? marked.parse(summary) : esc(summary);
        el('ai-summary-wrap').style.display = 'block';
    } else {
        el('ai-summary-wrap').style.display = 'none';
    }

    // subset warning
    const subset = p.subset_info;
    const warn = el('subset-warning');
    if (subset && subset.fitted_all === false) {
        let html = esc(subset.reason || '');
        if (subset.options && subset.options.length) {
            html += `<div class="table-wrap" style="margin-top:12px">${renderTable(
                subset.options.map(o => ({
                    Strategy: o.label,
                    Selected: (o.selected || []).join(', '),
                    Dropped: (o.dropped || []).join(', '),
                    'Est. h': o.estimated_time_h,
                    Satisfaction: o.total_satisfaction,
                }))
            )}</div>`;
        }
        warn.innerHTML = html;
        warn.style.display = 'block';
    } else {
        warn.style.display = 'none';
    }

    const best = p.options && p.options[0];
    if (!best) return;

    el('best-route-line').textContent = p.best_route_label || '';
    el('best-daybyday').innerHTML = renderDayByDay(best.days);
    el('best-scores-table').innerHTML = renderTable([best.scores]);

    buildMap(best.days);

    // Route options 2..5 (option 1 is already shown above as "Best route")
    const optionsHtml = (p.options || []).slice(1).map(opt => `
        <div class="route-option">
            <h3>Route ${opt.rank}</h3>
            <p class="route-line">${esc(opt.route_label)}</p>
            <h4>Route ${opt.rank} – Day-by-Day Plan</h4>
            ${renderDayByDay(opt.days)}
        </div>`).join('');
    el('route-options').innerHTML = optionsHtml ||
        '<p class="muted">Only one feasible route was found for these constraints.</p>';

    el('all-scores-table').innerHTML = renderTable(p.score_rows || []);
    el('accuracy-block').innerHTML = renderAccuracy(p.accuracy);

    const w = p.weights || {};
    el('weights-line').textContent = 'Weights used: {' +
        Object.entries(w).map(([k, v]) => `'${k}': ${v}`).join(', ') + '}';

    el('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* --------------------------------------------------------------- boot */
document.addEventListener('DOMContentLoaded', async () => {
    const form = el('trip-form');
    if (!form) return;

    // default trip start date = today
    const today = new Date();
    const iso = new Date(today.getTime() - today.getTimezoneOffset() * 60000)
        .toISOString().slice(0, 10);
    el('trip-start-date').value = iso;

    // slider value read-outs
    form.querySelectorAll('input[type=range][data-out]').forEach(r => {
        const out = el(r.dataset.out);
        r.addEventListener('input', () => { if (out) out.textContent = r.value; });
    });

    // +/- steppers
    document.querySelectorAll('.stepper button').forEach(btn => {
        btn.addEventListener('click', () => {
            const input = el(btn.dataset.target);
            if (!input) return;
            const step = parseFloat(btn.dataset.step);
            const min = parseFloat(input.min), max = parseFloat(input.max);
            let v = (parseFloat(input.value) || 0) + step;
            if (!isNaN(min)) v = Math.max(min, v);
            if (!isNaN(max)) v = Math.min(max, v);
            input.value = (step % 1 !== 0) ? v.toFixed(2) : v;
        });
    });

    // ---- populate the Start place dropdown from the real POI list ----
    const select = el('start-place');
    try {
        const res = await fetch('/api/pois');
        if (!res.ok) throw new Error(res.status);
        const data = await res.json();

        select.innerHTML = '';
        data.places.forEach(pl => {
            const opt = document.createElement('option');
            opt.value = pl.name;
            opt.textContent = pl.label || pl.name;
            select.appendChild(opt);
        });
        if (data.default) select.value = data.default;

        el('load-banner').textContent =
            `Loaded ${data.count} POIs and road network.`;
    } catch (err) {
        console.error('Failed to load POIs:', err);
        select.innerHTML = '<option value="">Could not load POIs</option>';
        const banner = el('load-banner');
        banner.textContent = 'Could not load the POI list — is the API server running?';
        banner.className = 'banner banner-error';
    }

    // deep link from Explore: planner.html?must_visit=Ella%20Rock
    const wanted = new URLSearchParams(window.location.search).get('must_visit');
    if (wanted) el('must-visit-input').value = wanted;

    // ---------------------------------------------------------- submit
    form.addEventListener('submit', async e => {
        e.preventDefault();
        const fd = new FormData(form);

        const payload = {
            start_place: select.value,
            days: parseInt(fd.get('days')),
            hours_per_day: parseFloat(fd.get('hours_per_day')),
            max_places: 12,                       // kept for API compatibility
            must_visit: fd.get('must_visit') || '',
            attraction_priority: parseInt(fd.get('attraction_priority')),
            budget_priority: parseInt(fd.get('budget_priority')),
            time_priority: parseInt(fd.get('time_priority')),
            popular_priority: parseInt(fd.get('popular_priority')),
            question: 'Plan a trip for me',
            use_all_pois: el('use-all-pois').checked,
            num_ants: parseInt(el('num-ants').value) || 80,
            iterations: parseInt(el('iterations').value) || 30,
            trip_start_date: el('trip-start-date').value || null,
            avoid_bad_weather: el('avoid-bad-weather').checked,
        };

        const btn = el('generate-btn');
        // write to the label span, not the button — the button also holds an icon
        const btnLabel = el('generate-label') || btn;
        btn.disabled = true;
        btnLabel.textContent = 'Generating…';
        el('loading-overlay').style.display = 'flex';
        el('error-banner').style.display = 'none';

        try {
            const res = await fetch('/api/plan_itinerary', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.detail || `Request failed (${res.status})`);
            }
            if (!data.planner) {
                throw new Error('The server did not return a plan.');
            }
            renderPlanner(data);
        } catch (err) {
            console.error('Error:', err);
            const banner = el('error-banner');
            banner.textContent = err.message || 'Could not generate itinerary.';
            banner.style.display = 'block';
            banner.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } finally {
            el('loading-overlay').style.display = 'none';
            btn.disabled = false;
            btnLabel.textContent = 'Generate optimised routes';
        }
    });
});
