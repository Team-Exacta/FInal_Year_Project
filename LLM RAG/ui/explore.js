/* ==========================================================================
   Explore Sri Lanka
   Interactive map + filterable place browser over the merged dataset built by
   scripts/build_explore_data.py (MOIP coordinates + UGC review insights +
   Wikipedia descriptions + graph features/activities).
   ========================================================================== */

const DATA_URL = 'data/explore_places.json';

/* ---------- category -> theme (32 raw categories collapse to 8 themes) ---- */
const THEMES = [
    { key: 'beach',    label: 'Beaches',     emoji: '🏖️', color: '#38bdf8', match: ['beach', 'lighthouse', 'bay'] },
    { key: 'heritage', label: 'Heritage',    emoji: '🛕', color: '#f5a524', match: ['religious', 'heritage', 'fort', 'historic', 'memorial', 'cultural', 'museum'] },
    { key: 'wildlife', label: 'Wildlife',    emoji: '🐘', color: '#a3e635', match: ['wildlife', 'national park', 'zoo', 'bird', 'forest', 'wetland'] },
    { key: 'water',    label: 'Waterfalls',  emoji: '💧', color: '#2dd4bf', match: ['waterfall', 'lake', 'river', 'hot springs'] },
    { key: 'views',    label: 'Views & hikes', emoji: '⛰️', color: '#ff6b5b', match: ['viewpoint', 'hiking', 'natural landmark'] },
    { key: 'parks',    label: 'Parks',       emoji: '🌳', color: '#4ade80', match: ['garden', 'park', 'promenade'] },
    { key: 'city',     label: 'City & culture', emoji: '🏙️', color: '#8b5cf6', match: ['shopping', 'market', 'tower', 'landmark', 'engineering', 'hotel', 'activity', 'farm', 'factory'] },
];
const FALLBACK_THEME = { key: 'other', label: 'Other', emoji: '📍', color: '#94a3b8' };

function themeFor(category) {
    const c = (category || '').toLowerCase();
    return THEMES.find(t => t.match.some(m => c.includes(m))) || FALLBACK_THEME;
}

/* ---------- review-insight filters --------------------------------------- */
const INSIGHT_FILTERS = [
    { id: 'time:EARLY_MORNING', label: '🌅 Best early morning', test: p => tod(p) === 'EARLY_MORNING' },
    { id: 'time:AFTERNOON',     label: '🌤️ Best afternoon',    test: p => tod(p) === 'AFTERNOON' },
    { id: 'time:EVENING',       label: '🌆 Best evening',       test: p => tod(p) === 'EVENING' },
    { id: 'crowd:quiet',        label: '🤫 Rarely crowded',     test: p => ['EMPTY', 'QUIET'].includes(crowdLabel(p)) },
    { id: 'crowd:packed',       label: '🔥 Very popular',       test: p => ['BUSY', 'PACKED'].includes(crowdLabel(p)) },
    { id: 'cost:free',          label: '🆓 Free entry',         test: p => costLevel(p) === 'FREE' },
    { id: 'cost:cheap',         label: '💸 Budget friendly',    test: p => ['FREE', 'LOW'].includes(costLevel(p)) },
    { id: 'season:DRY_SEASON',  label: '☀️ Dry season',         test: p => season(p) === 'DRY_SEASON' },
    { id: 'has:photo',          label: '📷 Has photo',          test: p => !!p.image },
];

const tod = p => p.ugc?.best_time?.time_of_day || null;
const season = p => p.ugc?.best_time?.season || null;
const crowdLabel = p => p.ugc?.crowd?.label || null;
const costLevel = p => p.ugc?.cost?.level || null;

/* ---------- formatting --------------------------------------------------- */
const titleCase = s => (s || '').toLowerCase().replace(/_/g, ' ')
    .replace(/^./, c => c.toUpperCase());

const CROWD_ICON = { EMPTY: '○', QUIET: '◔', MODERATE: '◑', BUSY: '◕', PACKED: '●' };
const COST_TEXT = { FREE: 'Free', LOW: 'Cheap', MODERATE: 'Moderate', HIGH: 'Pricey', VERY_HIGH: 'Expensive' };

function durationText(min) {
    if (!min) return null;
    const h = Math.floor(min / 60), m = min % 60;
    if (h && m) return `${h}h ${m}m`;
    if (h) return `${h}h`;
    return `${m}m`;
}

/* ==========================================================================
   State
   ========================================================================== */
let ALL = [];
let filtered = [];
let selectedId = null;

const state = {
    search: '',
    themes: new Set(),
    regions: new Set(),
    insights: new Set(),
};

let map = null;
const markers = new Map();      // place.id -> L.marker
let tileLayers = {};

/* ==========================================================================
   Boot
   ========================================================================== */
document.addEventListener('DOMContentLoaded', async () => {
    let payload;
    try {
        const res = await fetch(DATA_URL);
        if (!res.ok) throw new Error(res.status);
        payload = await res.json();
    } catch (err) {
        document.getElementById('place-list').innerHTML =
            `<div class="empty-state"><div>⚠️</div>
             <h3>Could not load place data</h3>
             <p>Run <code>python scripts/build_explore_data.py</code> to generate it.</p></div>`;
        console.error('Failed to load explore data:', err);
        return;
    }

    ALL = (payload.places || []).map(p => ({ ...p, theme: themeFor(p.category) }));
    document.getElementById('count-total').textContent = ALL.length;

    buildChips();
    initMap();
    wireEvents();
    applyFilters();

    // Deep link: explore.html?place=ella-rock
    const wanted = new URLSearchParams(location.search).get('place');
    if (wanted) {
        const hit = ALL.find(p => p.id === wanted);
        if (hit) openDrawer(hit);
    }
});

/* ==========================================================================
   Filter chips
   ========================================================================== */
function buildChips() {
    const themeBox = document.getElementById('theme-chips');
    const present = THEMES.filter(t => ALL.some(p => p.theme.key === t.key));
    present.forEach(t => {
        themeBox.appendChild(makeChip(`${t.emoji} ${t.label}`, () => toggle(state.themes, t.key), () => state.themes.has(t.key)));
    });

    const regionBox = document.getElementById('region-chips');
    const regions = [...new Set(ALL.map(p => p.region))].sort();
    regions.forEach(r => {
        regionBox.appendChild(makeChip(r, () => toggle(state.regions, r), () => state.regions.has(r)));
    });

    const insightBox = document.getElementById('insight-chips');
    INSIGHT_FILTERS.forEach(f => {
        if (!ALL.some(f.test)) return;
        insightBox.appendChild(makeChip(f.label, () => toggle(state.insights, f.id), () => state.insights.has(f.id)));
    });

    buildLegend(present);
}

function makeChip(label, onToggle, isActive) {
    const el = document.createElement('span');
    el.className = 'chip';
    el.textContent = label;
    el.addEventListener('click', () => {
        onToggle();
        el.classList.toggle('active', isActive());
        applyFilters();
    });
    el._sync = () => el.classList.toggle('active', isActive());
    return el;
}

function toggle(set, value) {
    set.has(value) ? set.delete(value) : set.add(value);
}

function buildLegend(themes) {
    const legend = document.getElementById('map-legend');
    themes.forEach(t => {
        const row = document.createElement('div');
        row.className = 'legend-item';
        row.innerHTML = `<span class="legend-dot" style="background:${t.color}"></span>${t.label}`;
        legend.appendChild(row);
    });
}

/* ==========================================================================
   Filtering
   ========================================================================== */
function applyFilters() {
    const q = state.search.trim().toLowerCase();

    filtered = ALL.filter(p => {
        if (state.themes.size && !state.themes.has(p.theme.key)) return false;
        if (state.regions.size && !state.regions.has(p.region)) return false;

        for (const id of state.insights) {
            const f = INSIGHT_FILTERS.find(x => x.id === id);
            if (f && !f.test(p)) return false;
        }

        if (q) {
            const haystack = [
                p.name, p.category, p.region, p.tagline,
                ...(p.activities || []).map(a => a.name),
                ...(p.features || []).map(f => f.name),
            ].join(' ').toLowerCase();
            if (!haystack.includes(q)) return false;
        }
        return true;
    });

    document.getElementById('count-shown').textContent = filtered.length;
    renderCards();
    renderMarkers();
}

/* ==========================================================================
   Cards
   ========================================================================== */
function renderCards() {
    const list = document.getElementById('place-list');
    list.innerHTML = '';

    if (!filtered.length) {
        list.innerHTML = `<div class="empty-state"><div>🧭</div>
            <h3>Nothing matches those filters</h3>
            <p>Try clearing a filter or searching for something else.</p></div>`;
        return;
    }

    const frag = document.createDocumentFragment();
    filtered.slice(0, 400).forEach((p, i) => {
        const card = document.createElement('article');
        card.className = 'place-card';
        card.dataset.id = p.id;
        card.style.animationDelay = `${Math.min(i, 20) * 18}ms`;

        const media = p.image
            ? `<img src="${p.image}" alt="${escapeHtml(p.name)}" loading="lazy"
                    onerror="this.parentElement.innerHTML='<div class=&quot;card-media-fallback&quot;>${p.theme.emoji}</div>'">`
            : `<div class="card-media-fallback">${p.theme.emoji}</div>`;

        card.innerHTML = `
            <div class="card-media">
                ${media}
                <span class="card-cat">${p.theme.emoji} ${escapeHtml(p.category)}</span>
                <span class="card-region">📍 ${escapeHtml(p.region)}</span>
            </div>
            <div class="card-body">
                <h3>${escapeHtml(p.name)}</h3>
                ${p.tagline ? `<p class="card-tagline">${escapeHtml(p.tagline)}</p>` : ''}
                <div class="ugc-row">${ugcTags(p)}</div>
            </div>`;

        card.addEventListener('click', () => openDrawer(p));
        card.addEventListener('mouseenter', () => highlightMarker(p.id, true));
        card.addEventListener('mouseleave', () => highlightMarker(p.id, false));
        frag.appendChild(card);
    });
    list.appendChild(frag);
}

function ugcTags(p) {
    const tags = [];
    const bt = p.ugc?.best_time;
    if (bt?.time_of_day) tags.push(`<span class="ugc-tag time">🕐 ${titleCase(bt.time_of_day)}</span>`);

    const cr = p.ugc?.crowd;
    if (cr?.label) tags.push(`<span class="ugc-tag crowd">${CROWD_ICON[cr.label] || '◑'} ${titleCase(cr.label)}</span>`);

    const co = p.ugc?.cost;
    if (co?.level) {
        const amount = co.median_lkr ? ` · ${co.median_lkr} LKR` : '';
        tags.push(`<span class="ugc-tag cost">💰 ${COST_TEXT[co.level] || titleCase(co.level)}${amount}</span>`);
    }

    if (!tags.length && p.ugc?.total_reviews) {
        tags.push(`<span class="ugc-tag">💬 ${p.ugc.total_reviews} reviews</span>`);
    }
    return tags.join('');
}

/* ==========================================================================
   Map
   ========================================================================== */
function initMap() {
    map = L.map('explore-map', { zoomControl: true, attributionControl: true })
        .setView([7.6, 80.75], 7.4);

    tileLayers = {
        dark: L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 19,
        }),
        satellite: L.tileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles &copy; Esri', maxZoom: 18,
        }),
        terrain: L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenTopoMap contributors', maxZoom: 17,
        }),
    };
    tileLayers.dark.addTo(map);

    document.querySelectorAll('.map-toggle button').forEach(btn => {
        btn.addEventListener('click', () => {
            Object.values(tileLayers).forEach(l => map.removeLayer(l));
            tileLayers[btn.dataset.layer].addTo(map);
            document.querySelectorAll('.map-toggle button')
                .forEach(b => b.classList.toggle('active', b === btn));
        });
    });
}

function renderMarkers() {
    const keep = new Set(filtered.map(p => p.id));

    // drop markers that filtered out
    markers.forEach((marker, id) => {
        if (!keep.has(id)) {
            map.removeLayer(marker);
            markers.delete(id);
        }
    });

    filtered.forEach(p => {
        if (p.lat == null || p.lon == null || markers.has(p.id)) return;

        const icon = L.divIcon({
            className: '',
            html: `<div class="poi-marker" data-id="${p.id}"
                        style="background:${p.theme.color}"><span>${p.theme.emoji}</span></div>`,
            iconSize: [30, 30],
            iconAnchor: [15, 30],
            popupAnchor: [0, -28],
        });

        const marker = L.marker([p.lat, p.lon], { icon, title: p.name }).addTo(map);
        marker.bindPopup(popupHtml(p), { minWidth: 210, maxWidth: 240 });
        marker.on('popupopen', () => {
            const btn = document.querySelector('.map-popup-btn[data-id="' + p.id + '"]');
            if (btn) btn.addEventListener('click', () => openDrawer(p));
        });
        marker.on('click', () => setSelected(p.id));
        markers.set(p.id, marker);
    });
}

function popupHtml(p) {
    const img = p.image
        ? `<img class="map-popup-img" src="${p.image}" alt="" onerror="this.remove()">` : '';
    const tags = ugcTags(p);
    return `${img}<b>${escapeHtml(p.name)}</b><br>
            <span style="color:#93a2bb;font-size:.8rem">${escapeHtml(p.category)} · ${escapeHtml(p.region)}</span>
            ${tags ? `<div class="ugc-row" style="margin-top:8px">${tags}</div>` : ''}
            <span class="map-popup-btn" data-id="${p.id}">View details →</span>`;
}

function highlightMarker(id, on) {
    const marker = markers.get(id);
    if (!marker) return;
    const el = marker.getElement()?.querySelector('.poi-marker');
    if (el) el.classList.toggle('is-active', on);
    if (on) marker.setZIndexOffset(1000);
    else marker.setZIndexOffset(0);
}

function setSelected(id) {
    if (selectedId) {
        highlightMarker(selectedId, false);
        document.querySelector(`.place-card[data-id="${selectedId}"]`)?.classList.remove('is-active');
    }
    selectedId = id;
    if (!id) return;
    highlightMarker(id, true);
    document.querySelector(`.place-card[data-id="${id}"]`)?.classList.add('is-active');
}

/* ==========================================================================
   Detail drawer
   ========================================================================== */
function openDrawer(p) {
    setSelected(p.id);

    const hero = document.getElementById('drawer-hero');
    hero.innerHTML = `
        ${p.image ? `<img src="${p.image_full || p.image}" alt="${escapeHtml(p.name)}"
             onerror="this.remove()">` : ''}
        <div class="drawer-hero-text">
            <h2>${escapeHtml(p.name)}</h2>
            <div class="drawer-meta">
                <span>${p.theme.emoji} ${escapeHtml(p.category)}</span>
                <span>📍 ${escapeHtml(p.region)}</span>
                ${p.ugc?.total_reviews ? `<span>💬 ${p.ugc.total_reviews} reviews</span>` : ''}
                ${durationText(p.duration_min) ? `<span>⏱️ ~${durationText(p.duration_min)} visit</span>` : ''}
            </div>
        </div>`;

    const content = document.getElementById('drawer-content');
    content.innerHTML = [
        descriptionSection(p),
        insightSection(p),
        evidenceSection(p),
        tagSection('What visitors do here', p.activities),
        tagSection('What they talk about', p.features),
        tagSection('Facilities mentioned', p.facilities),
        sourceSection(p),
    ].filter(Boolean).join('');

    const actions = document.getElementById('drawer-actions');
    actions.innerHTML = `
        <button class="btn btn-primary" id="act-plan">🗓️ Add to trip plan</button>
        <button class="btn btn-ghost" id="act-ask">💬 Ask the AI</button>`;
    document.getElementById('act-plan').onclick =
        () => location.href = 'planner.html?must_visit=' + encodeURIComponent(p.name);
    document.getElementById('act-ask').onclick =
        () => location.href = 'chat.html?q=' + encodeURIComponent(`Tell me about visiting ${p.name}`);

    document.getElementById('drawer').classList.add('open');
    document.getElementById('drawer').setAttribute('aria-hidden', 'false');
    document.getElementById('drawer-backdrop').classList.add('open');
    document.querySelector('.drawer-scroll').scrollTop = 0;

    if (p.lat != null && map) {
        map.flyTo([p.lat, p.lon], 11, { duration: 0.8 });
    }
}

function closeDrawer() {
    document.getElementById('drawer').classList.remove('open');
    document.getElementById('drawer').setAttribute('aria-hidden', 'true');
    document.getElementById('drawer-backdrop').classList.remove('open');
}

function descriptionSection(p) {
    const text = p.description || p.tagline;
    if (!text) return '';
    return `<div class="drawer-section">
        <h4>About</h4>
        <p class="drawer-desc">${escapeHtml(text)}</p>
    </div>`;
}

function insightSection(p) {
    const u = p.ugc;
    if (!u || (!u.best_time && !u.crowd && !u.cost)) return '';

    const cards = [];

    if (u.best_time) {
        const bt = u.best_time;
        const bits = [];
        if (bt.season) bits.push(titleCase(bt.season));
        if (bt.months?.length) bits.push(bt.months.join(', '));
        if (bt.avoid?.length) bits.push(`avoid ${bt.avoid.join(', ')}`);
        cards.push(insightCard('🕐', 'Best time',
            bt.time_of_day ? titleCase(bt.time_of_day) : (bt.season ? titleCase(bt.season) : 'Mentioned'),
            bits.join(' · '), bt.confidence, bt.based_on));
    }

    if (u.crowd) {
        const cr = u.crowd;
        const sub = [];
        if (cr.avg_level) sub.push(`avg ${cr.avg_level.toFixed(1)}/5`);
        if (cr.busiest_period?.length) sub.push(`busiest ${cr.busiest_period.join(', ')}`);
        cards.push(insightCard(CROWD_ICON[cr.label] || '◑', 'Crowd level',
            titleCase(cr.label), sub.join(' · '), cr.confidence, cr.based_on));
    }

    if (u.cost) {
        const co = u.cost;
        const sub = [];
        if (co.median_lkr) sub.push(`median ${co.median_lkr} LKR`);
        if (co.fee_type) sub.push(titleCase(co.fee_type));
        cards.push(insightCard('💰', 'Typical cost',
            COST_TEXT[co.level] || titleCase(co.level), sub.join(' · '), co.confidence, co.based_on));
    }

    return `<div class="drawer-section">
        <h4>What ${p.ugc.total_reviews || 'the'} reviews say</h4>
        <div class="insight-grid">${cards.join('')}</div>
    </div>`;
}

function insightCard(icon, label, value, sub, confidence, basedOn) {
    const pct = Math.round((confidence || 0) * 100);
    return `<div class="insight">
        <div class="insight-head">${icon} ${label}</div>
        <div class="insight-value">${escapeHtml(value || '—')}</div>
        ${sub ? `<div class="insight-sub">${escapeHtml(sub)}</div>` : ''}
        ${confidence != null ? `
            <div class="conf-bar"><div class="conf-fill" style="width:${pct}%"></div></div>
            <div class="insight-sub">${pct}% confidence${basedOn ? ` · ${basedOn} mentions` : ''}</div>` : ''}
    </div>`;
}

function evidenceSection(p) {
    const u = p.ugc;
    if (!u) return '';
    const quotes = [
        ...(u.best_time?.evidence || []),
        ...(u.crowd?.evidence || []),
        ...(u.cost?.evidence || []),
    ].slice(0, 4);
    if (!quotes.length) return '';

    return `<div class="drawer-section">
        <h4>Straight from the reviews</h4>
        ${quotes.map(q => `<div class="quote">"${escapeHtml(q.text)}"
            <span class="quote-meta">${q.rating ? '★'.repeat(q.rating) : ''} ${escapeHtml(q.date || '')}</span>
        </div>`).join('')}
    </div>`;
}

function tagSection(title, items) {
    if (!items || !items.length) return '';
    return `<div class="drawer-section">
        <h4>${title}</h4>
        <div class="tag-cloud">
            ${items.map(i => `<span class="tag ${i.sentiment === 'positive' ? 'positive' : i.sentiment === 'negative' ? 'negative' : ''}">
                <b>${escapeHtml(i.name)}</b><span class="tag-pct">${i.pct}%</span></span>`).join('')}
        </div>
    </div>`;
}

function sourceSection(p) {
    const links = [];
    if (p.wikipedia_url) {
        links.push(`<a href="${p.wikipedia_url}" target="_blank" rel="noopener"
            style="color:var(--sky);text-decoration:underline">Wikipedia article</a>`);
    }
    if (p.lat != null) {
        links.push(`<a href="https://www.google.com/maps/search/?api=1&query=${p.lat},${p.lon}"
            target="_blank" rel="noopener" style="color:var(--sky);text-decoration:underline">Open in Google Maps</a>`);
    }
    if (p.image_credit) {
        links.push(`<span style="color:var(--faint)">Photo: ${escapeHtml(p.image_credit)} (Wikimedia)</span>`);
    }
    if (!links.length) return '';
    return `<div class="drawer-section">
        <h4>Sources</h4>
        <div style="display:flex;flex-direction:column;gap:6px;font-size:.85rem">${links.join('')}</div>
    </div>`;
}

/* ==========================================================================
   Events
   ========================================================================== */
function wireEvents() {
    const search = document.getElementById('search-input');
    let timer;
    search.addEventListener('input', e => {
        clearTimeout(timer);
        timer = setTimeout(() => {
            state.search = e.target.value;
            applyFilters();
        }, 180);
    });

    document.getElementById('reset-btn').addEventListener('click', () => {
        state.search = '';
        state.themes.clear();
        state.regions.clear();
        state.insights.clear();
        search.value = '';
        document.querySelectorAll('.chip').forEach(c => c._sync && c._sync());
        applyFilters();
        map.flyTo([7.6, 80.75], 7.4, { duration: 0.7 });
    });

    document.getElementById('drawer-close').addEventListener('click', closeDrawer);
    document.getElementById('drawer-backdrop').addEventListener('click', closeDrawer);
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });

    // small screens: swap between the list and the map
    const body = document.getElementById('explore-body');
    const btn = document.getElementById('mobile-map-btn');
    btn.addEventListener('click', () => {
        body.classList.toggle('map-mode');
        btn.textContent = body.classList.contains('map-mode') ? '📋 List' : '🗺️ Map';
        if (body.classList.contains('map-mode')) setTimeout(() => map.invalidateSize(), 120);
    });
}

/* ---------- util ---------- */
function escapeHtml(str) {
    return String(str ?? '').replace(/[&<>"']/g, c => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
}
