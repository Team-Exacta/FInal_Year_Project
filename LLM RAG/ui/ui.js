/* ==========================================================================
   Lanka Travel AI — shared UI kit
   ---------------------------------------------------------------------------
   One place for everything more than one page needs: the icon set, the place
   taxonomy, the recommendation-match maths, the wishlist store, and the
   render helpers for cards / badges / empty states.

   Loaded on every page BEFORE the page script. Everything hangs off `window.LT`
   so the page scripts stay small and none of them re-implement a card.
   ========================================================================== */
(function (global) {
    'use strict';

    /* ======================================================================
       Icons — one set, one style (24px grid, 1.75 stroke, round caps).
       Nothing else in the UI is allowed to introduce a second icon style.
       ====================================================================== */
    const ICONS = {
        compass: '<circle cx="12" cy="12" r="10"/><polygon points="16.2 7.8 14.1 14.1 7.8 16.2 9.9 9.9"/>',
        pin: '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
        search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
        filter: '<polygon points="22 3 2 3 10 12.5 10 19 14 21 14 12.5"/>',
        sliders: '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
        heart: '<path d="M19 14c1.5-1.5 3-3.2 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.8 0-3 .5-4.5 2-1.5-1.5-2.7-2-4.5-2A5.5 5.5 0 0 0 2 8.5C2 10.8 3.5 12.5 5 14l7 7Z"/>',
        star: '<polygon points="12 2 15.1 8.3 22 9.3 17 14.1 18.2 21 12 17.8 5.8 21 7 14.1 2 9.3 8.9 8.3"/>',
        calendar: '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/>',
        wallet: '<path d="M21 12V7H5a2 2 0 0 1 0-4h14v4"/><path d="M3 5v14a2 2 0 0 0 2 2h16v-5"/><path d="M18 12a2 2 0 0 0 0 4h4v-4Z"/>',
        clock: '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
        users: '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8"/>',
        share: '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><path d="m8.6 13.5 6.8 4M15.4 6.5l-6.8 4"/>',
        map: '<polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2"/><path d="M8 2v16M16 6v16"/>',
        sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4"/>',
        moon: '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
        cloud: '<path d="M17.5 19a4.5 4.5 0 0 0 0-9 6 6 0 0 0-11.6 1.6A4 4 0 0 0 6.5 19Z"/>',
        sparkles: '<path d="m12 3 1.9 4.6L18.5 9.5l-4.6 1.9L12 16l-1.9-4.6L5.5 9.5l4.6-1.9Z"/><path d="m19 15 .9 2.1 2.1.9-2.1.9-.9 2.1-.9-2.1-2.1-.9 2.1-.9Z"/>',
        chevronRight: '<polyline points="9 18 15 12 9 6"/>',
        chevronDown: '<polyline points="6 9 12 15 18 9"/>',
        arrowRight: '<path d="M5 12h14"/><polyline points="12 5 19 12 12 19"/>',
        x: '<path d="M18 6 6 18M6 6l12 12"/>',
        menu: '<path d="M3 6h18M3 12h18M3 18h18"/>',
        check: '<polyline points="20 6 9 17 4 12"/>',
        grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
        list: '<path d="M8 6h13M8 12h13M8 18h13"/><circle cx="3.5" cy="6" r="1.2"/><circle cx="3.5" cy="12" r="1.2"/><circle cx="3.5" cy="18" r="1.2"/>',
        alert: '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4M12 17h.01"/>',
        info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
        imageOff: '<path d="M2 2l20 20"/><path d="M10.4 10.4a2 2 0 1 0 2.8 2.8"/><path d="M21 15V5a2 2 0 0 0-2-2H9"/><path d="M3 7v12a2 2 0 0 0 2 2h12"/>',
        camera: '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2Z"/><circle cx="12" cy="13" r="4"/>',
        mountain: '<path d="m8 3 4 8 5-5 5 15H2Z"/>',
        waves: '<path d="M2 6c.6.5 1.2 1 2.5 1C7 7 7 5 9.5 5c2.6 0 2.4 2 5 2 1.3 0 1.9-.5 2.5-1"/><path d="M2 12c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 1.3 0 1.9-.5 2.5-1"/><path d="M2 18c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 1.3 0 1.9-.5 2.5-1"/>',
        tree: '<path d="M12 2 6 12h4l-4 7h12l-4-7h4Z"/><path d="M12 19v3"/>',
        paw: '<circle cx="11" cy="4.5" r="2"/><circle cx="18" cy="8" r="2"/><circle cx="4" cy="8" r="2"/><circle cx="6.5" cy="14.5" r="2"/><path d="M13.8 13c1.7 1 2.7 2.6 2.7 4.3a3.7 3.7 0 0 1-3.7 3.7h-1.6a3.7 3.7 0 0 1-3.7-3.7c0-1.7 1-3.3 2.7-4.3a3.5 3.5 0 0 1 3.6 0Z"/>',
        landmark: '<path d="M3 22h18M6 18v-7M10 18v-7M14 18v-7M18 18v-7"/><polygon points="12 2 20 7 4 7"/>',
        droplet: '<path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5S12.5 5.5 12 3c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7Z"/>',
        building: '<rect x="4" y="2" width="16" height="20" rx="2"/><path d="M9 22v-4h6v4"/><path d="M8 6h.01M12 6h.01M16 6h.01M8 10h.01M12 10h.01M16 10h.01M8 14h.01M12 14h.01M16 14h.01"/>',
        trending: '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
        route: '<circle cx="6" cy="19" r="3"/><path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15"/><circle cx="18" cy="5" r="3"/>',
        shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/>',
        target: '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
        message: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2Z"/>',
        send: '<path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>',
        mail: '<rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 6-10 7L2 6"/>',
        external: '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><path d="M10 14 21 3"/>',
        globe: '<circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15 15 0 0 1 4 10 15 15 0 0 1-4 10 15 15 0 0 1-4-10 15 15 0 0 1 4-10Z"/>',
        eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
        book: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z"/>',
        quote: '<path d="M6 17h3l2-4V7H5v6h3Zm9 0h3l2-4V7h-6v6h3Z"/>',
        facebook: '<path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3Z"/>',
        instagram: '<rect x="2" y="2" width="20" height="20" rx="5"/><circle cx="12" cy="12" r="4"/><circle cx="17.5" cy="6.5" r="1"/>',
        twitter: '<path d="M4 4l7.5 9.8L4.4 20h2l6-5.4L17 20h3l-7.9-10.3L19.5 4h-2l-5.4 4.9L8.2 4Z"/>',
        youtube: '<rect x="2" y="5" width="20" height="14" rx="4"/><polygon points="10.5 9 15.5 12 10.5 15"/>',
        plus: '<path d="M12 5v14M5 12h14"/>',
        minus: '<path d="M5 12h14"/>',
    };

    /** Returns an <svg> string for `name`; unknown names render nothing. */
    function icon(name, extraClass) {
        const body = ICONS[name];
        if (!body) return '';
        return `<svg class="icon${extraClass ? ' ' + extraClass : ''}" viewBox="0 0 24 24" `
            + `aria-hidden="true" focusable="false">${body}</svg>`;
    }

    /* ======================================================================
       Place taxonomy — 32 raw categories collapse into 8 browsable themes.
       Shared by Explore (chips, pins, legend), Home (category tiles) and the
       details page, so the colour/icon for "Beaches" is defined exactly once.
       ====================================================================== */
    const THEMES = [
        { key: 'beach', label: 'Beaches', icon: 'waves', color: '#0EA5E9',
          match: ['beach', 'lighthouse', 'bay'] },
        { key: 'heritage', label: 'Heritage', icon: 'landmark', color: '#F97316',
          match: ['religious', 'heritage', 'fort', 'historic', 'memorial', 'cultural', 'museum'] },
        { key: 'wildlife', label: 'Wildlife', icon: 'paw', color: '#16A34A',
          match: ['wildlife', 'national park', 'zoo', 'bird', 'forest', 'wetland'] },
        { key: 'water', label: 'Waterfalls', icon: 'droplet', color: '#14B8A6',
          match: ['waterfall', 'lake', 'river', 'hot springs'] },
        { key: 'views', label: 'Views & hikes', icon: 'mountain', color: '#6366F1',
          match: ['viewpoint', 'hiking', 'natural landmark'] },
        { key: 'parks', label: 'Parks & gardens', icon: 'tree', color: '#65A30D',
          match: ['garden', 'park', 'promenade'] },
        { key: 'city', label: 'City & culture', icon: 'building', color: '#0369A1',
          match: ['shopping', 'market', 'tower', 'landmark', 'engineering', 'hotel', 'activity', 'farm', 'factory'] },
    ];
    const FALLBACK_THEME = { key: 'other', label: 'Other', icon: 'pin', color: '#64748B', match: [] };

    function themeFor(category) {
        const c = String(category || '').toLowerCase();
        return THEMES.find(t => t.match.some(m => c.includes(m))) || FALLBACK_THEME;
    }

    /* ======================================================================
       Formatting
       ====================================================================== */
    const COST_TEXT = { FREE: 'Free', LOW: 'Budget', MODERATE: 'Moderate', HIGH: 'Premium', VERY_HIGH: 'Luxury' };
    const COST_PRICE = { FREE: 'Free', LOW: '$', MODERATE: '$$', HIGH: '$$$', VERY_HIGH: '$$$$' };
    const CROWD_TEXT = { EMPTY: 'Very quiet', QUIET: 'Quiet', MODERATE: 'Moderate', BUSY: 'Busy', PACKED: 'Packed' };

    const titleCase = s => String(s || '').toLowerCase().replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());

    function durationText(min) {
        if (!min) return null;
        const h = Math.floor(min / 60), m = min % 60;
        if (h && m) return `${h}h ${m}m`;
        if (h) return `${h}h`;
        return `${m}m`;
    }

    const tod = p => p?.ugc?.best_time?.time_of_day || null;
    const season = p => p?.ugc?.best_time?.season || null;
    const crowdLabel = p => p?.ugc?.crowd?.label || null;
    const costLevel = p => p?.ugc?.cost?.level || null;
    const reviewCount = p => p?.ugc?.total_reviews || 0;
    const ratingOf = p => (typeof p?.satisfaction === 'number' && p.satisfaction > 0) ? p.satisfaction : null;

    function priceText(p) {
        const lvl = costLevel(p);
        if (lvl) return COST_PRICE[lvl] || titleCase(lvl);
        if (typeof p?.planner_cost === 'number') return p.planner_cost > 0 ? '$$' : 'Free';
        return null;
    }

    function seasonText(p) {
        const bt = p?.ugc?.best_time;
        if (!bt) return null;
        if (bt.months && bt.months.length) return bt.months.slice(0, 3).join(', ');
        if (bt.season) return titleCase(bt.season);
        if (bt.time_of_day) return titleCase(bt.time_of_day);
        return null;
    }

    function escapeHtml(str) {
        return String(str ?? '').replace(/[&<>"']/g, c => (
            { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
        ));
    }

    /* ======================================================================
       Recommendation match
       ---------------------------------------------------------------------
       Deterministic and explainable — no random numbers. Two halves:

         quality (60%) traveller rating, how many reviews back it up, and how
                       much review-mined insight we actually hold for the place
         fit     (40%) share of the preferences the visitor has actually set
                       that this place satisfies. With nothing set it settles
                       at a neutral 0.72 so the ranking still reflects quality.

       High ≥ 80 (teal), medium ≥ 65 (blue), lower (amber). Never red — a 58%
       match is still a real suggestion, not an error.
       ====================================================================== */
    function matchScore(place, prefs) {
        const rating = ratingOf(place);
        const reviews = reviewCount(place);
        const u = place.ugc || {};

        const ratingPart = rating ? (rating / 5) : 0.62;
        const volumePart = Math.min(1, Math.log10(1 + reviews) / 2.7);   // ~500 reviews ⇒ 1
        const depthPart = ((u.best_time ? 1 : 0) + (u.crowd ? 1 : 0) + (u.cost ? 1 : 0)) / 3;
        const quality = (ratingPart * 0.55) + (volumePart * 0.25) + (depthPart * 0.20);

        let wanted = 0, met = 0;
        if (prefs) {
            if (prefs.themes && prefs.themes.size) {
                wanted++;
                if (prefs.themes.has(themeFor(place.category).key)) met++;
            }
            if (prefs.regions && prefs.regions.size) {
                wanted++;
                if (prefs.regions.has(place.region)) met++;
            }
            if (prefs.insights && prefs.insights.size) {
                prefs.insights.forEach(id => {
                    wanted++;
                    const f = (prefs.insightTests || {})[id];
                    if (f && f(place)) met++;
                });
            }
            if (prefs.activities && prefs.activities.size) {
                const names = (place.activities || []).map(a => a.name);
                prefs.activities.forEach(a => {
                    wanted++;
                    if (names.includes(a)) met++;
                });
            }
            // Range filters only count once the visitor moves them off their
            // default — otherwise every place would trivially "match" them.
            if (prefs.minRating) {
                wanted++;
                if ((rating || 0) >= prefs.minRating) met++;
            }
            if (prefs.maxCost != null) {
                wanted++;
                const c = typeof place.planner_cost === 'number' ? place.planner_cost : 0;
                if (c <= prefs.maxCost) met++;
            }
            if (prefs.maxDuration != null) {
                wanted++;
                if ((place.duration_min || 0) <= prefs.maxDuration) met++;
            }
        }
        const fit = wanted ? (met / wanted) : 0.72;

        const pct = Math.round(((quality * 0.6) + (fit * 0.4)) * 100);
        return Math.max(38, Math.min(99, pct));
    }

    const matchLevel = pct => (pct >= 80 ? 'high' : pct >= 65 ? 'mid' : 'low');
    const matchWord = pct => (pct >= 80 ? 'Great match' : pct >= 65 ? 'Good match' : 'Partial match');

    function matchRing(pct, big) {
        return `<div class="match match-ring${big ? ' match-ring-lg' : ''} is-${matchLevel(pct)}"
                     style="--pct:${pct}" role="img"
                     aria-label="${pct} percent match — ${matchWord(pct)}"><span>${pct}%</span></div>`;
    }

    function matchBadge(pct) {
        return `<span class="match match-badge is-${matchLevel(pct)}"
                      title="${matchWord(pct)} — based on your filters, traveller rating and review volume">
                    ${icon('target')}${pct}% match
                </span>`;
    }

    function matchBar(pct) {
        return `<div class="match is-${matchLevel(pct)}">
                    <div class="match-bar"><i style="width:${pct}%"></i></div>
                </div>`;
    }

    /* ======================================================================
       Wishlist — saved places, persisted in localStorage.
       Emits `wishlistchange` so every open card and the nav counter stay in
       sync without any of them knowing about each other.
       ====================================================================== */
    const WISHLIST_KEY = 'lta-wishlist';

    const Wishlist = {
        _read() {
            try { return JSON.parse(localStorage.getItem(WISHLIST_KEY)) || []; }
            catch (e) { return []; }
        },
        _write(ids) {
            try { localStorage.setItem(WISHLIST_KEY, JSON.stringify(ids)); }
            catch (e) { /* private mode — wishlist stays in-memory for the session */ }
            document.dispatchEvent(new CustomEvent('wishlistchange', { detail: { ids } }));
        },
        all() { return this._read(); },
        count() { return this._read().length; },
        has(id) { return this._read().indexOf(id) !== -1; },
        toggle(id) {
            const ids = this._read();
            const i = ids.indexOf(id);
            if (i === -1) ids.push(id); else ids.splice(i, 1);
            this._write(ids);
            return i === -1;
        },
        remove(id) {
            const ids = this._read().filter(x => x !== id);
            this._write(ids);
        },
        clear() { this._write([]); },
    };

    /* ======================================================================
       Render helpers
       ====================================================================== */
    function ugcTags(p) {
        const tags = [];
        const bt = p.ugc && p.ugc.best_time;
        if (bt && bt.time_of_day) {
            tags.push(`<span class="ugc-tag time">${icon('clock')}${escapeHtml(titleCase(bt.time_of_day))}</span>`);
        }
        const cr = p.ugc && p.ugc.crowd;
        if (cr && cr.label) {
            tags.push(`<span class="ugc-tag crowd">${icon('users')}${escapeHtml(CROWD_TEXT[cr.label] || titleCase(cr.label))}</span>`);
        }
        const co = p.ugc && p.ugc.cost;
        if (co && co.level) {
            const amount = co.median_lkr ? ` · ${co.median_lkr} LKR` : '';
            tags.push(`<span class="ugc-tag cost">${icon('wallet')}${escapeHtml((COST_TEXT[co.level] || titleCase(co.level)) + amount)}</span>`);
        }
        if (!tags.length && reviewCount(p)) {
            tags.push(`<span class="ugc-tag">${icon('message')}${reviewCount(p)} reviews</span>`);
        }
        return tags.join('');
    }

    function mediaHtml(p, theme) {
        if (!p.image) {
            return `<div class="card-media-fallback">${icon(theme.icon)}</div>`;
        }
        return `<img src="${escapeHtml(p.image)}" alt="${escapeHtml(p.name)}" loading="lazy" decoding="async">`;
    }

    /**
     * The one destination card used by Home, Explore and Saved.
     * opts: { match: number|null, showFav: bool, href: string, compact: bool }
     */
    function placeCard(p, opts) {
        opts = opts || {};
        const theme = p.theme || themeFor(p.category);
        const href = opts.href || `place.html?id=${encodeURIComponent(p.id)}`;
        const rating = ratingOf(p);
        const price = priceText(p);
        const saved = Wishlist.has(p.id);
        const pct = opts.match;

        const stats = [];
        if (rating) {
            stats.push(`<span class="rating">${icon('star')}${rating.toFixed(1)}
                        <small>(${reviewCount(p)})</small></span>`);
        } else if (reviewCount(p)) {
            stats.push(`<span>${icon('message')}${reviewCount(p)} reviews</span>`);
        }
        const dur = durationText(p.duration_min);
        if (dur) stats.push(`<span>${icon('clock')}~${dur}</span>`);
        const best = seasonText(p);
        if (best && !opts.compact) stats.push(`<span>${icon('calendar')}${escapeHtml(best)}</span>`);

        return `
        <article class="place-card" data-id="${escapeHtml(p.id)}">
            <div class="card-media">
                ${mediaHtml(p, theme)}
                <span class="card-cat">${icon(theme.icon)}${escapeHtml(p.category)}</span>
                ${opts.showFav === false ? '' : `
                <button class="card-fav" type="button" data-fav="${escapeHtml(p.id)}"
                        aria-pressed="${saved}"
                        aria-label="${saved ? 'Remove' : 'Save'} ${escapeHtml(p.name)} ${saved ? 'from' : 'to'} your saved places">
                    ${icon('heart')}
                </button>`}
                <span class="card-region">${icon('pin')}${escapeHtml(p.region)}</span>
                ${pct != null ? `<span class="card-match-onmedia">${matchBadge(pct)}</span>` : ''}
            </div>
            <div class="card-body">
                <h3><a href="${href}">${escapeHtml(p.name)}</a></h3>
                <span class="card-place">${icon('globe')}${escapeHtml(p.region)} · Sri Lanka</span>
                ${p.tagline ? `<p class="card-tagline">${escapeHtml(p.tagline)}</p>` : ''}
                ${stats.length ? `<div class="card-stats">${stats.join('')}</div>` : ''}
                <div class="ugc-row">${ugcTags(p)}</div>
                <div class="card-foot">
                    <span class="card-price">${price ? `<b>${escapeHtml(price)}</b> entry` : 'Entry cost unknown'}</span>
                    <a class="btn btn-text btn-sm" href="${href}">View details ${icon('arrowRight')}</a>
                </div>
            </div>
        </article>`;
    }

    function skeletonCards(n) {
        let out = '';
        for (let i = 0; i < n; i++) {
            out += `<div class="skeleton-card" aria-hidden="true">
                        <div class="skeleton sk-media"></div>
                        <div class="sk-body">
                            <div class="skeleton sk-line w-70"></div>
                            <div class="skeleton sk-line w-45"></div>
                            <div class="skeleton sk-line w-90"></div>
                        </div>
                    </div>`;
        }
        return out;
    }

    /** Empty / error / no-results panel. `action` is raw HTML for the CTA. */
    function stateHtml(o) {
        return `<div class="state ${o.tone ? 'state-' + o.tone : ''}" role="status">
            <div class="state-icon">${icon(o.icon || 'compass')}</div>
            <h3>${escapeHtml(o.title)}</h3>
            ${o.body ? `<p>${o.body}</p>` : ''}
            ${o.action || ''}
        </div>`;
    }

    /* ======================================================================
       Styled dropdown
       ---------------------------------------------------------------------
       A native <select>'s option list is drawn by the OS and cannot be themed,
       so it lands on the page as a plain white box. This wraps the select in a
       button + listbox we control, and keeps the real <select> in the DOM
       (hidden, untabbable) so form serialisation, `.value` and `change`
       listeners all keep working exactly as before.

       Options are rebuilt on every open, so selects populated late — the hero's
       trip-type list arrives with the dataset — stay in sync.
       ====================================================================== */
    function enhanceSelect(select) {
        if (!select || select.dataset.enhanced) return;
        select.dataset.enhanced = '1';

        const parent = select.parentElement;
        // drop a decorative chevron sibling; the new button brings its own
        parent.querySelectorAll(':scope > svg.icon').forEach(svg => svg.remove());

        const wrap = document.createElement('div');
        wrap.className = 'dropdown';

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'dropdown-btn';
        btn.setAttribute('aria-haspopup', 'listbox');
        btn.setAttribute('aria-expanded', 'false');

        const labelEl = select.id && document.querySelector(`label[for="${select.id}"]`);
        if (labelEl) btn.setAttribute('aria-label', labelEl.textContent.trim());

        const valueEl = document.createElement('span');
        valueEl.className = 'dropdown-value';
        btn.appendChild(valueEl);
        btn.insertAdjacentHTML('beforeend', icon('chevronDown'));

        const menu = document.createElement('ul');
        menu.className = 'dropdown-menu';
        menu.setAttribute('role', 'listbox');
        menu.hidden = true;

        parent.insertBefore(wrap, select);
        wrap.append(btn, menu, select);
        select.classList.add('sr-only');
        select.tabIndex = -1;
        select.setAttribute('aria-hidden', 'true');

        const syncValue = () => {
            const opt = select.options[select.selectedIndex];
            valueEl.textContent = opt ? opt.textContent : '';
        };

        function build() {
            menu.innerHTML = '';
            Array.prototype.forEach.call(select.options, (opt, i) => {
                const li = document.createElement('li');
                li.className = 'dropdown-option';
                li.setAttribute('role', 'option');
                li.setAttribute('aria-selected', String(i === select.selectedIndex));
                li.tabIndex = -1;
                li.dataset.index = String(i);
                li.innerHTML = `<span>${escapeHtml(opt.textContent)}</span>${icon('check')}`;
                menu.appendChild(li);
            });
        }

        const isOpen = () => !menu.hidden;

        function open() {
            build();
            menu.hidden = false;
            btn.setAttribute('aria-expanded', 'true');
            wrap.classList.add('is-open');
            const active = menu.querySelector('[aria-selected="true"]') || menu.firstElementChild;
            if (active) active.focus();
        }

        function close(focusBtn) {
            if (!isOpen()) return;
            menu.hidden = true;
            btn.setAttribute('aria-expanded', 'false');
            wrap.classList.remove('is-open');
            if (focusBtn) btn.focus();
        }

        function choose(index) {
            select.selectedIndex = index;
            syncValue();
            close(true);
            select.dispatchEvent(new Event('change', { bubbles: true }));
        }

        btn.addEventListener('click', () => (isOpen() ? close(true) : open()));
        btn.addEventListener('keydown', e => {
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); open(); }
        });

        menu.addEventListener('click', e => {
            const li = e.target.closest('.dropdown-option');
            if (li) choose(Number(li.dataset.index));
        });

        menu.addEventListener('keydown', e => {
            const items = Array.from(menu.children);
            const i = items.indexOf(document.activeElement);
            if (e.key === 'ArrowDown') { e.preventDefault(); (items[i + 1] || items[0]).focus(); }
            else if (e.key === 'ArrowUp') { e.preventDefault(); (items[i - 1] || items[items.length - 1]).focus(); }
            else if (e.key === 'Home') { e.preventDefault(); items[0].focus(); }
            else if (e.key === 'End') { e.preventDefault(); items[items.length - 1].focus(); }
            else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(i); }
            else if (e.key === 'Escape') { e.preventDefault(); close(true); }
            else if (e.key === 'Tab') close(false);
        });

        document.addEventListener('click', e => {
            if (!wrap.contains(e.target)) close(false);
        });

        // lets a page update the control after setting select.value in code
        select._dropdownSync = syncValue;
        syncValue();
    }

    function enhanceSelects(scope) {
        (scope || document).querySelectorAll('select[data-enhance]').forEach(enhanceSelect);
    }

    /* ======================================================================
       Page chrome — runs on every page
       ====================================================================== */
    function initNav() {
        const nav = document.querySelector('.site-nav');
        if (!nav) return;

        // Solid + shadowed once the user leaves the top of the hero.
        const overHero = nav.classList.contains('is-over-hero');
        const threshold = overHero ? 80 : 4;
        let ticking = false;
        const sync = () => {
            const y = window.scrollY;
            nav.classList.toggle('is-scrolled', y > threshold);
            if (overHero) nav.classList.toggle('is-over-hero', y <= threshold);
            ticking = false;
        };
        window.addEventListener('scroll', () => {
            if (!ticking) { ticking = true; requestAnimationFrame(sync); }
        }, { passive: true });
        sync();

        // Mobile menu
        const toggle = document.querySelector('.nav-toggle');
        const panel = document.querySelector('.nav-panel');
        if (toggle && panel) {
            toggle.addEventListener('click', () => {
                const open = panel.classList.toggle('open');
                toggle.setAttribute('aria-expanded', String(open));
                toggle.innerHTML = icon(open ? 'x' : 'menu');
            });
            panel.addEventListener('click', e => {
                if (e.target.closest('a')) {
                    panel.classList.remove('open');
                    toggle.setAttribute('aria-expanded', 'false');
                    toggle.innerHTML = icon('menu');
                }
            });
        }
    }

    function syncWishlistBadges() {
        const n = Wishlist.count();
        document.querySelectorAll('[data-wishlist-count]').forEach(el => {
            el.textContent = n > 99 ? '99+' : String(n);
            el.hidden = n === 0;
        });
        document.querySelectorAll('[data-fav]').forEach(btn => {
            const on = Wishlist.has(btn.dataset.fav);
            btn.setAttribute('aria-pressed', String(on));
        });
    }

    /** One delegated handler serves every favourite button on the page. */
    function initWishlist() {
        document.addEventListener('click', e => {
            const btn = e.target.closest('[data-fav]');
            if (!btn) return;
            e.preventDefault();
            e.stopPropagation();
            const added = Wishlist.toggle(btn.dataset.fav);
            btn.setAttribute('aria-pressed', String(added));
            if (added) {
                btn.classList.add('just-saved');
                setTimeout(() => btn.classList.remove('just-saved'), 420);
            }
        });
        document.addEventListener('wishlistchange', syncWishlistBadges);
        syncWishlistBadges();
    }

    /** Fade sections in once, cheaply; a no-op under prefers-reduced-motion. */
    function initReveal() {
        const items = document.querySelectorAll('.reveal');
        if (!items.length) return;
        if (!('IntersectionObserver' in window) ||
            window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            items.forEach(el => el.classList.add('is-visible'));
            return;
        }
        const io = new IntersectionObserver((entries, obs) => {
            entries.forEach(entry => {
                if (!entry.isIntersecting) return;
                entry.target.classList.add('is-visible');
                obs.unobserve(entry.target);
            });
        }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
        items.forEach(el => io.observe(el));
    }

    /* Broken remote photo → labelled placeholder instead of a torn icon.
       `error` doesn't bubble, so this listens in the capture phase. */
    function initImageFallbacks() {
        document.addEventListener('error', e => {
            const img = e.target;
            if (!img || img.tagName !== 'IMG' || img.dataset.fallbackDone) return;
            // hero art degrades to its own backdrop; a plate would look worse
            if (img.hasAttribute('data-no-fallback')) return;
            img.dataset.fallbackDone = '1';
            const box = document.createElement('div');
            box.className = 'img-fallback';
            box.innerHTML = icon('imageOff') + '<span>Photo unavailable</span>';
            if (img.parentElement) img.parentElement.replaceChild(box, img);
        }, true);
    }

    function initFooter() {
        document.querySelectorAll('[data-year]').forEach(el => {
            el.textContent = String(new Date().getFullYear());
        });
        const form = document.querySelector('[data-newsletter]');
        if (form) {
            form.addEventListener('submit', e => {
                e.preventDefault();
                const note = form.parentElement.querySelector('.newsletter-note');
                if (note) {
                    note.textContent = 'Thanks — travel ideas are on their way to your inbox.';
                    note.classList.add('is-ok');
                }
                form.reset();
            });
        }
    }

    function init() {
        initNav();
        initWishlist();
        initReveal();
        initImageFallbacks();
        initFooter();
        enhanceSelects();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    /* ====================================================================== */
    global.LT = {
        ICONS, icon,
        THEMES, FALLBACK_THEME, themeFor,
        COST_TEXT, COST_PRICE, CROWD_TEXT,
        titleCase, durationText, escapeHtml, priceText, seasonText,
        tod, season, crowdLabel, costLevel, reviewCount, ratingOf,
        matchScore, matchLevel, matchWord, matchRing, matchBadge, matchBar,
        Wishlist, syncWishlistBadges,
        ugcTags, placeCard, skeletonCards, stateHtml,
        enhanceSelect, enhanceSelects,
        DATA_URL: 'data/explore_places.json',
    };
})(window);
