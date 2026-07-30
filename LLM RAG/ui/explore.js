/* ==========================================================================
   Explore
   Filterable browser + map over the merged dataset built by
   scripts/build_explore_data.py (MOIP coordinates + UGC review insights +
   Wikipedia descriptions + graph features/activities).

   Cards, badges, icons and the match score all come from ui.js so this page
   and the homepage can never render a destination two different ways.
   ========================================================================== */
(function () {
    'use strict';

    const {
        DATA_URL, THEMES, themeFor, icon, escapeHtml, titleCase, durationText,
        placeCard, skeletonCards, stateHtml, matchScore, matchRing, matchBadge,
        ugcTags, COST_TEXT, CROWD_TEXT, ratingOf, reviewCount, syncWishlistBadges,
    } = window.LT;

    /* ---------------------------------------------- review-insight filters */
    const INSIGHT_FILTERS = [
        { id: 'time:EARLY_MORNING', label: 'Best early morning', icon: 'sun',
          test: p => tod(p) === 'EARLY_MORNING' },
        { id: 'time:AFTERNOON', label: 'Best afternoon', icon: 'cloud',
          test: p => tod(p) === 'AFTERNOON' },
        { id: 'time:EVENING', label: 'Best evening', icon: 'moon',
          test: p => tod(p) === 'EVENING' },
        { id: 'crowd:quiet', label: 'Rarely crowded', icon: 'users',
          test: p => ['EMPTY', 'QUIET'].includes(crowd(p)) },
        { id: 'crowd:packed', label: 'Very popular', icon: 'trending',
          test: p => ['BUSY', 'PACKED'].includes(crowd(p)) },
        { id: 'cost:free', label: 'Free entry', icon: 'wallet',
          test: p => cost(p) === 'FREE' },
        { id: 'cost:cheap', label: 'Budget friendly', icon: 'wallet',
          test: p => ['FREE', 'LOW'].includes(cost(p)) },
        { id: 'season:DRY_SEASON', label: 'Dry season', icon: 'sun',
          test: p => season(p) === 'DRY_SEASON' },
        { id: 'has:photo', label: 'Has photo', icon: 'camera', test: p => !!p.image },
    ];
    const INSIGHT_TESTS = Object.fromEntries(INSIGHT_FILTERS.map(f => [f.id, f.test]));

    const tod = p => p.ugc?.best_time?.time_of_day || null;
    const season = p => p.ugc?.best_time?.season || null;
    const crowd = p => p.ugc?.crowd?.label || null;
    const cost = p => p.ugc?.cost?.level || null;

    const COST_MAX = 6000, DURATION_MAX = 480;

    /* ============================================================== state */
    let ALL = [];
    let filtered = [];
    let selectedId = null;
    let ACTIVITIES = [];

    const state = {
        search: '',
        themes: new Set(),
        regions: new Set(),
        activities: new Set(),
        insights: new Set(),
        minRating: 0,
        maxCost: COST_MAX,
        maxDuration: DURATION_MAX,
        sort: 'match',
        view: 'grid',
        insightTests: INSIGHT_TESTS,
    };

    let map = null;
    let tileLayers = {};
    const markers = new Map();
    const el = id => document.getElementById(id);

    /* =============================================================== boot */
    document.addEventListener('DOMContentLoaded', async () => {
        el('place-list').innerHTML = skeletonCards(6);
        wireStaticEvents();

        let payload;
        try {
            const res = await fetch(DATA_URL);
            if (!res.ok) throw new Error('HTTP ' + res.status);
            payload = await res.json();
        } catch (err) {
            console.error('Failed to load explore data:', err);
            el('place-list').innerHTML = stateHtml({
                tone: 'error',
                icon: 'alert',
                title: 'Could not load destination data',
                body: 'Rebuild it with <code>python scripts/build_explore_data.py</code>, '
                    + 'then reload this page.',
                action: '<button class="btn btn-primary" type="button" onclick="location.reload()">Try again</button>',
            });
            return;
        }

        ALL = (payload.places || []).map(p => ({ ...p, theme: themeFor(p.category) }));
        ACTIVITIES = topActivities(ALL, 10);
        el('count-total').textContent = ALL.length;

        buildFilters();
        readUrlParams();
        initMap();
        wireFilterEvents();
        applyFilters();

        // deep link: explore.html?place=ella-rock
        const wanted = new URLSearchParams(location.search).get('place');
        if (wanted) {
            const hit = ALL.find(p => p.id === wanted);
            if (hit) openDrawer(hit);
        }
    });

    function topActivities(places, n) {
        const tally = new Map();
        places.forEach(p => (p.activities || []).forEach(a => {
            if (!a.name) return;
            tally.set(a.name, (tally.get(a.name) || 0) + 1);
        }));
        return [...tally.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, n)
            .map(([name]) => name);
    }

    /* ============================================================ filters */
    function buildFilters() {
        const themeBox = el('theme-chips');
        THEMES.filter(t => ALL.some(p => p.theme.key === t.key)).forEach(t => {
            themeBox.appendChild(chip(
                `${icon(t.icon)}${escapeHtml(t.label)}`,
                () => toggle(state.themes, t.key),
                () => state.themes.has(t.key),
            ));
        });

        const regionBox = el('region-chips');
        [...new Set(ALL.map(p => p.region))].sort().forEach(r => {
            regionBox.appendChild(chip(
                escapeHtml(r),
                () => toggle(state.regions, r),
                () => state.regions.has(r),
            ));
        });

        const actBox = el('activity-chips');
        ACTIVITIES.forEach(a => {
            actBox.appendChild(chip(
                escapeHtml(a),
                () => toggle(state.activities, a),
                () => state.activities.has(a),
            ));
        });

        const insightBox = el('insight-chips');
        INSIGHT_FILTERS.forEach(f => {
            if (!ALL.some(f.test)) return;
            insightBox.appendChild(chip(
                `${icon(f.icon)}${escapeHtml(f.label)}`,
                () => toggle(state.insights, f.id),
                () => state.insights.has(f.id),
            ));
        });

        buildLegend();
    }

    function chip(html, onToggle, isActive) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'chip';
        btn.innerHTML = html;
        btn.setAttribute('aria-pressed', 'false');
        const sync = () => {
            const on = isActive();
            btn.classList.toggle('active', on);
            btn.setAttribute('aria-pressed', String(on));
        };
        btn.addEventListener('click', () => { onToggle(); sync(); applyFilters(); });
        btn._sync = sync;
        return btn;
    }

    function toggle(set, value) {
        set.has(value) ? set.delete(value) : set.add(value);
    }

    function buildLegend() {
        const legend = el('map-legend');
        THEMES.filter(t => ALL.some(p => p.theme.key === t.key)).forEach(t => {
            const row = document.createElement('div');
            row.className = 'legend-item';
            row.innerHTML = `<span class="legend-dot" style="background:${t.color}"></span>${escapeHtml(t.label)}`;
            legend.appendChild(row);
        });
    }

    /* -------- entry points from the homepage search / footer links -------- */
    function readUrlParams() {
        const q = new URLSearchParams(location.search);
        const setIfPresent = (key, apply) => {
            const v = q.get(key);
            if (v) apply(v);
        };
        setIfPresent('q', v => { state.search = v; el('search-input').value = v; });
        setIfPresent('theme', v => v.split(',').forEach(x => state.themes.add(x)));
        setIfPresent('region', v => v.split(',').forEach(x => state.regions.add(x)));
        setIfPresent('insight', v => v.split(',').forEach(x => state.insights.add(x)));
        setIfPresent('sort', v => {
            const sel = el('sort-select');
            state.sort = v;
            sel.value = v;
            if (sel._dropdownSync) sel._dropdownSync();   // repaint the styled control
        });
        syncChips();
    }

    function syncChips() {
        document.querySelectorAll('.chip').forEach(c => c._sync && c._sync());
    }

    /* ========================================================== filtering */
    function applyFilters() {
        const q = state.search.trim().toLowerCase();

        filtered = ALL.filter(p => {
            if (state.themes.size && !state.themes.has(p.theme.key)) return false;
            if (state.regions.size && !state.regions.has(p.region)) return false;

            if (state.activities.size) {
                const names = (p.activities || []).map(a => a.name);
                for (const a of state.activities) if (!names.includes(a)) return false;
            }

            for (const id of state.insights) {
                const test = INSIGHT_TESTS[id];
                if (test && !test(p)) return false;
            }

            if (state.minRating > 0 && (ratingOf(p) || 0) < state.minRating) return false;

            if (state.maxCost < COST_MAX) {
                const c = typeof p.planner_cost === 'number' ? p.planner_cost : 0;
                if (c > state.maxCost) return false;
            }

            if (state.maxDuration < DURATION_MAX) {
                const d = p.duration_min || 0;
                if (d > state.maxDuration) return false;
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

        const prefs = activePrefs();
        filtered.forEach(p => { p._match = matchScore(p, prefs); });
        sortResults();

        el('count-shown').textContent = filtered.length;
        renderTokens();
        renderCards();
        renderMarkers();
    }

    /* Only the preferences the visitor has actually expressed feed the match
       score — a slider still sitting at its default is not a preference. */
    function activePrefs() {
        return {
            themes: state.themes,
            regions: state.regions,
            activities: state.activities,
            insights: state.insights,
            insightTests: INSIGHT_TESTS,
            minRating: state.minRating > 0 ? state.minRating : 0,
            maxCost: state.maxCost < COST_MAX ? state.maxCost : null,
            maxDuration: state.maxDuration < DURATION_MAX ? state.maxDuration : null,
        };
    }

    function sortResults() {
        const byName = (a, b) => a.name.localeCompare(b.name);
        const sorters = {
            match: (a, b) => (b._match - a._match) || byName(a, b),
            rating: (a, b) => ((ratingOf(b) || 0) - (ratingOf(a) || 0)) || byName(a, b),
            popular: (a, b) => (reviewCount(b) - reviewCount(a)) || byName(a, b),
            price: (a, b) => ((a.planner_cost || 0) - (b.planner_cost || 0)) || byName(a, b),
            name: byName,
        };
        filtered.sort(sorters[state.sort] || sorters.match);
    }

    /* ------------------------------- removable "active filter" tokens ---- */
    function renderTokens() {
        const box = el('active-tokens');
        const tokens = [];

        const add = (label, clear) => tokens.push({ label, clear });

        if (state.search) add(`“${state.search}”`, () => { state.search = ''; el('search-input').value = ''; });
        state.themes.forEach(k => {
            const t = THEMES.find(x => x.key === k);
            add(t ? t.label : k, () => state.themes.delete(k));
        });
        state.regions.forEach(r => add(r, () => state.regions.delete(r)));
        state.activities.forEach(a => add(a, () => state.activities.delete(a)));
        state.insights.forEach(id => {
            const f = INSIGHT_FILTERS.find(x => x.id === id);
            add(f ? f.label : id, () => state.insights.delete(id));
        });
        if (state.minRating > 0) add(`${state.minRating}★ and up`, () => setRating(0));
        if (state.maxCost < COST_MAX) add(`Under ${state.maxCost} LKR`, () => setCost(COST_MAX));
        if (state.maxDuration < DURATION_MAX) add(`Under ${durationText(state.maxDuration)}`, () => setDuration(DURATION_MAX));

        box.innerHTML = tokens.map((t, i) =>
            `<span class="token">${escapeHtml(t.label)}
                <button type="button" data-token="${i}" aria-label="Remove filter ${escapeHtml(t.label)}">
                    ${icon('x')}
                </button>
             </span>`).join('');

        box.querySelectorAll('[data-token]').forEach(btn => {
            btn.addEventListener('click', () => {
                tokens[Number(btn.dataset.token)].clear();
                syncChips();
                applyFilters();
            });
        });

        const count = el('filter-count');
        const n = tokens.length;
        count.textContent = String(n);
        count.hidden = n === 0;
    }

    /* ============================================================== cards */
    function renderCards() {
        const list = el('place-list');

        if (!filtered.length) {
            list.innerHTML = stateHtml({
                icon: 'search',
                title: 'No destinations match those filters',
                body: 'Try widening the budget, dropping a region, or clearing everything to start again.',
                action: '<button class="btn btn-primary" type="button" data-clear-all>Clear all filters</button>',
            });
            list.querySelector('[data-clear-all]')?.addEventListener('click', clearAll);
            return;
        }

        list.innerHTML = filtered.slice(0, 400)
            .map(p => placeCard(p, { match: p._match }))
            .join('');

        list.querySelectorAll('.place-card').forEach(card => {
            const id = card.dataset.id;
            card.addEventListener('mouseenter', () => highlightMarker(id, true));
            card.addEventListener('mouseleave', () => highlightMarker(id, false));
        });
        syncWishlistBadges();
    }

    /* ================================================================ map */
    /* The default CARTO basemap follows the site theme so a light page never
       sits next to a black map. Same layer object — only the tile URL swaps. */
    function basemapUrl() {
        return document.documentElement.getAttribute('data-theme') === 'dark'
            ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
            : 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png';
    }

    function initMap() {
        map = L.map('explore-map', { zoomControl: true, attributionControl: true })
            .setView([7.6, 80.75], 7.4);

        tileLayers = {
            dark: L.tileLayer(basemapUrl(), {
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

        document.addEventListener('themechange', () => tileLayers.dark.setUrl(basemapUrl()));

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

        markers.forEach((marker, id) => {
            if (!keep.has(id)) { map.removeLayer(marker); markers.delete(id); }
        });

        filtered.forEach(p => {
            if (p.lat == null || p.lon == null || markers.has(p.id)) return;

            const divIcon = L.divIcon({
                className: '',
                html: `<div class="poi-marker" data-id="${escapeHtml(p.id)}"
                            style="background:${p.theme.color}"><span>${icon(p.theme.icon)}</span></div>`,
                iconSize: [30, 30],
                iconAnchor: [15, 30],
                popupAnchor: [0, -28],
            });

            const marker = L.marker([p.lat, p.lon], { icon: divIcon, title: p.name }).addTo(map);
            marker.bindPopup(popupHtml(p), { minWidth: 220, maxWidth: 250 });
            marker.on('popupopen', () => {
                document.querySelector(`.map-popup-btn[data-id="${p.id}"]`)
                    ?.addEventListener('click', () => openDrawer(p));
            });
            marker.on('click', () => setSelected(p.id));
            markers.set(p.id, marker);
        });
    }

    function popupHtml(p) {
        const img = p.image
            ? `<img class="map-popup-img" src="${escapeHtml(p.image)}" alt="" onerror="this.remove()">` : '';
        const tags = ugcTags(p);
        return `${img}<b>${escapeHtml(p.name)}</b><br>
            <span style="color:var(--text-3)">${escapeHtml(p.category)} · ${escapeHtml(p.region)}</span>
            ${tags ? `<div class="ugc-row" style="margin-top:8px">${tags}</div>` : ''}
            <span class="map-popup-btn" data-id="${escapeHtml(p.id)}">Quick look →</span>`;
    }

    function highlightMarker(id, on) {
        const marker = markers.get(id);
        if (!marker) return;
        marker.getElement()?.querySelector('.poi-marker')?.classList.toggle('is-active', on);
        marker.setZIndexOffset(on ? 1000 : 0);
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

    /* ==================================================== quick-look drawer */
    function openDrawer(p) {
        setSelected(p.id);
        const pct = p._match != null ? p._match : matchScore(p, activePrefs());

        el('drawer-hero').innerHTML = `
            ${p.image ? `<img src="${escapeHtml(p.image_full || p.image)}" alt="${escapeHtml(p.name)}">` : ''}
            <div class="drawer-hero-text">
                <h2>${escapeHtml(p.name)}</h2>
                <div class="drawer-meta">
                    <span>${icon(p.theme.icon)}${escapeHtml(p.category)}</span>
                    <span>${icon('pin')}${escapeHtml(p.region)}</span>
                    ${reviewCount(p) ? `<span>${icon('message')}${reviewCount(p)} reviews</span>` : ''}
                    ${durationText(p.duration_min) ? `<span>${icon('clock')}~${durationText(p.duration_min)}</span>` : ''}
                </div>
            </div>`;

        el('drawer-content').innerHTML = [
            matchSection(p, pct),
            descriptionSection(p),
            insightSection(p),
            evidenceSection(p),
            tagSection('What visitors do here', p.activities),
            tagSection('What they talk about', p.features),
        ].filter(Boolean).join('');

        el('drawer-actions').innerHTML = `
            <a class="btn btn-primary" href="place.html?id=${encodeURIComponent(p.id)}">
                View full details ${icon('arrowRight')}
            </a>
            <button class="btn btn-secondary" type="button" data-fav="${escapeHtml(p.id)}"
                    aria-pressed="${window.LT.Wishlist.has(p.id)}">
                ${icon('heart')} Save
            </button>`;

        const drawer = el('drawer');
        drawer.classList.add('open');
        drawer.setAttribute('aria-hidden', 'false');
        el('drawer-backdrop').classList.add('open');
        document.querySelector('.drawer-scroll').scrollTop = 0;
        el('drawer-close').focus();
        syncWishlistBadges();

        if (p.lat != null && map) map.flyTo([p.lat, p.lon], 11, { duration: 0.8 });
    }

    function closeDrawer() {
        const drawer = el('drawer');
        if (!drawer.classList.contains('open')) return;
        drawer.classList.remove('open');
        drawer.setAttribute('aria-hidden', 'true');
        el('drawer-backdrop').classList.remove('open');
    }

    function matchSection(p, pct) {
        return `<div class="drawer-section">
            <div class="summary-match">
                ${matchRing(pct)}
                <p><b>${window.LT.matchWord(pct)}</b>
                Scored on your filters, the traveller rating and how much review evidence backs it.</p>
            </div>
        </div>`;
    }

    function descriptionSection(p) {
        const text = p.description || p.tagline;
        if (!text) return '';
        const short = text.length > 320 ? text.slice(0, 320).replace(/\s+\S*$/, '') + '…' : text;
        return `<div class="drawer-section">
            <h4>About</h4>
            <p class="drawer-desc">${escapeHtml(short)}</p>
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
            cards.push(insightCard('clock', 'Best time',
                bt.time_of_day ? titleCase(bt.time_of_day) : (bt.season ? titleCase(bt.season) : 'Mentioned'),
                bits.join(' · '), bt.confidence, bt.based_on));
        }
        if (u.crowd) {
            const cr = u.crowd;
            const sub = [];
            if (cr.avg_level) sub.push(`avg ${cr.avg_level.toFixed(1)}/5`);
            if (cr.busiest_period?.length) sub.push(`busiest ${cr.busiest_period.join(', ')}`);
            cards.push(insightCard('users', 'Crowd level',
                CROWD_TEXT[cr.label] || titleCase(cr.label), sub.join(' · '), cr.confidence, cr.based_on));
        }
        if (u.cost) {
            const co = u.cost;
            const sub = [];
            if (co.median_lkr) sub.push(`median ${co.median_lkr} LKR`);
            if (co.fee_type) sub.push(titleCase(co.fee_type));
            cards.push(insightCard('wallet', 'Typical cost',
                COST_TEXT[co.level] || titleCase(co.level), sub.join(' · '), co.confidence, co.based_on));
        }

        return `<div class="drawer-section">
            <h4>What ${reviewCount(p) || 'the'} reviews say</h4>
            <div class="insight-grid">${cards.join('')}</div>
        </div>`;
    }

    function insightCard(iconName, label, value, sub, confidence, basedOn) {
        const pct = Math.round((confidence || 0) * 100);
        return `<div class="insight">
            <div class="insight-head">${icon(iconName)}${escapeHtml(label)}</div>
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
        ].slice(0, 3);
        if (!quotes.length) return '';

        return `<div class="drawer-section">
            <h4>Straight from the reviews</h4>
            ${quotes.map(q => `<div class="quote">“${escapeHtml(q.text)}”
                <span class="quote-meta">${q.rating ? '★'.repeat(q.rating) : ''} ${escapeHtml(q.date || '')}</span>
            </div>`).join('')}
        </div>`;
    }

    function tagSection(title, items) {
        if (!items || !items.length) return '';
        return `<div class="drawer-section">
            <h4>${escapeHtml(title)}</h4>
            <div class="tag-cloud">
                ${items.slice(0, 10).map(i => `<span class="tag ${i.sentiment === 'positive' ? 'positive'
                    : i.sentiment === 'negative' ? 'negative' : ''}">
                    <b>${escapeHtml(i.name)}</b><span class="tag-pct">${i.pct}%</span></span>`).join('')}
            </div>
        </div>`;
    }

    /* ============================================================= events */
    function wireStaticEvents() {
        // filter drawer (tablet + mobile)
        const aside = el('filter-aside');
        const open = el('filter-open');
        const close = el('filter-close');
        const backdrop = el('drawer-backdrop');
        const setOpen = on => {
            aside.classList.toggle('open', on);
            open.setAttribute('aria-expanded', String(on));
            // the shared backdrop dims the page behind whichever panel is open
            backdrop.classList.toggle('open', on);
            if (on) close.focus(); else open.focus();
        };
        open.addEventListener('click', () => setOpen(true));
        close.addEventListener('click', () => setOpen(false));

        // drawer
        el('drawer-close').addEventListener('click', closeDrawer);
        backdrop.addEventListener('click', () => {
            closeDrawer();
            if (aside.classList.contains('open')) setOpen(false);
        });
        document.addEventListener('keydown', e => {
            if (e.key !== 'Escape') return;
            closeDrawer();
            if (aside.classList.contains('open')) setOpen(false);
        });

        // list ⇄ map on small screens
        const body = el('explore-body');
        const mapBtn = el('mobile-map-btn');
        mapBtn.addEventListener('click', () => {
            const on = body.classList.toggle('map-mode');
            el('map-btn-label').textContent = on ? 'List' : 'Map';
            if (on && map) setTimeout(() => map.invalidateSize(), 140);
        });
    }

    function wireFilterEvents() {
        let timer;
        el('search-input').addEventListener('input', e => {
            clearTimeout(timer);
            timer = setTimeout(() => { state.search = e.target.value; applyFilters(); }, 180);
        });

        el('min-rating').addEventListener('input', e => setRating(Number(e.target.value)));
        el('max-cost').addEventListener('input', e => setCost(Number(e.target.value)));
        el('max-duration').addEventListener('input', e => setDuration(Number(e.target.value)));

        el('sort-select').addEventListener('change', e => {
            state.sort = e.target.value;
            sortResults();
            renderCards();
        });

        document.querySelectorAll('.view-toggle button').forEach(btn => {
            btn.addEventListener('click', () => {
                state.view = btn.dataset.view;
                el('place-list').classList.toggle('is-list', state.view === 'list');
                document.querySelectorAll('.view-toggle button').forEach(b => {
                    const on = b === btn;
                    b.classList.toggle('active', on);
                    b.setAttribute('aria-pressed', String(on));
                });
            });
        });

        el('reset-btn').addEventListener('click', clearAll);
    }

    function setRating(v) {
        state.minRating = v;
        el('min-rating').value = v;
        el('min-rating-out').textContent = v > 0 ? `${v}★` : 'Any';
        applyFilters();
    }
    function setCost(v) {
        state.maxCost = v;
        el('max-cost').value = v;
        el('max-cost-out').textContent = v >= COST_MAX ? 'Any' : (v === 0 ? 'Free only' : `${v} LKR`);
        applyFilters();
    }
    function setDuration(v) {
        state.maxDuration = v;
        el('max-duration').value = v;
        el('max-duration-out').textContent = v >= DURATION_MAX ? 'Any' : durationText(v);
        applyFilters();
    }

    function clearAll() {
        state.search = '';
        state.themes.clear();
        state.regions.clear();
        state.activities.clear();
        state.insights.clear();
        state.minRating = 0;
        state.maxCost = COST_MAX;
        state.maxDuration = DURATION_MAX;

        el('search-input').value = '';
        el('min-rating').value = 0;
        el('min-rating-out').textContent = 'Any';
        el('max-cost').value = COST_MAX;
        el('max-cost-out').textContent = 'Any';
        el('max-duration').value = DURATION_MAX;
        el('max-duration-out').textContent = 'Any';

        syncChips();
        applyFilters();
        if (map) map.flyTo([7.6, 80.75], 7.4, { duration: 0.7 });
    }
})();
