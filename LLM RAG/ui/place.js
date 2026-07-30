/* ==========================================================================
   Destination details — place.html?id=<place id>
   Renders one destination in full: gallery, overview, review-mined facts,
   activities, travel tips, evidence quotes, map, similar places, and a sticky
   trip-planning summary on desktop.
   ========================================================================== */
(function () {
    'use strict';

    const {
        DATA_URL, themeFor, icon, escapeHtml, titleCase, durationText, placeCard,
        stateHtml, matchScore, matchRing, matchWord, Wishlist, syncWishlistBadges,
        COST_TEXT, CROWD_TEXT, ratingOf, reviewCount, priceText,
    } = window.LT;

    const root = document.getElementById('place-root');
    let place = null;
    let miniMap = null;

    document.addEventListener('DOMContentLoaded', boot);

    async function boot() {
        const id = new URLSearchParams(location.search).get('id');
        if (!id) return fail('No destination selected', 'Pick a place from the Explore page to see its details.');

        let payload;
        try {
            const res = await fetch(DATA_URL);
            if (!res.ok) throw new Error('HTTP ' + res.status);
            payload = await res.json();
        } catch (err) {
            console.error('Failed to load destination data:', err);
            return fail('Could not load destination data',
                'The dataset did not respond. Check your connection and try again.', true);
        }

        const all = (payload.places || []).map(p => ({ ...p, theme: themeFor(p.category) }));
        place = all.find(p => p.id === id);
        if (!place) {
            return fail('We could not find that destination',
                'It may have been renamed or removed from the dataset.');
        }

        document.title = `${place.name} — Lanka Travel AI`;
        render(all);
        initMiniMap();
        syncWishlistBadges();
        wireActions();
    }

    function fail(title, body, retry) {
        root.innerHTML = `<div class="container">${stateHtml({
            tone: 'error',
            icon: retry ? 'alert' : 'search',
            title,
            body,
            action: retry
                ? '<button class="btn btn-primary" type="button" onclick="location.reload()">Try again</button>'
                : '<a class="btn btn-primary" href="explore.html">Browse destinations</a>',
        })}</div>`;
    }

    /* ================================================================ view */
    function render(all) {
        const p = place;
        const pct = matchScore(p, null);
        const rating = ratingOf(p);

        root.innerHTML = `
            ${galleryHtml(p)}
            <div class="container">
                <div class="detail-layout">
                    <div>
                        ${headerHtml(p, rating)}
                        ${overviewHtml(p)}
                        ${factsHtml(p)}
                        ${listSection('Recommended activities', 'compass', p.activities)}
                        ${listSection('What visitors talk about', 'message', p.features)}
                        ${listSection('Facilities mentioned', 'shield', p.facilities)}
                        ${tipsHtml(p)}
                        ${evidenceHtml(p)}
                        ${mapHtml(p)}
                        ${similarHtml(p, all)}
                        ${sourcesHtml(p)}
                    </div>
                    <aside class="detail-aside">${summaryHtml(p, pct, rating)}</aside>
                </div>
            </div>`;
    }

    function galleryHtml(p) {
        // The corpus carries one Wikimedia photo per place. The grid expands on
        // its own the day more images are added — it never pads with duplicates.
        const images = [p.image_full, p.image].filter(Boolean)
            .filter((v, i, a) => a.indexOf(v) === i);

        if (!images.length) {
            return `<div class="place-hero"><div class="gallery">
                        <figure><div class="img-fallback">${icon('imageOff')}<span>No photo yet</span></div></figure>
                    </div></div>`;
        }

        // The stylesheet's default is the 3-up mosaic. One or two photos need
        // their own track counts, otherwise the grid leaves an empty cell.
        const gridStyle = images.length === 1
            ? 'grid-template-columns:1fr;grid-template-rows:1fr'
            : images.length === 2
                ? 'grid-template-rows:1fr'
                : '';
        const figStyle = images.length < 3 ? ' style="grid-row:auto"' : '';

        return `<div class="place-hero">
            <div class="gallery"${gridStyle ? ` style="${gridStyle}"` : ''}>
                ${images.map((src, i) => `
                    <figure${figStyle}>
                        <img src="${escapeHtml(src)}" alt="${escapeHtml(p.name)}"
                             loading="${i === 0 ? 'eager' : 'lazy'}" decoding="async">
                        ${i === 0 && p.image_credit
                            ? `<span class="gallery-more">${icon('camera')} ${escapeHtml(p.image_credit)}</span>`
                            : ''}
                    </figure>`).join('')}
            </div>
        </div>`;
    }

    function headerHtml(p, rating) {
        return `
        <nav class="breadcrumb" aria-label="Breadcrumb">
            <a href="index.html">Home</a>${icon('chevronRight')}
            <a href="explore.html">Explore</a>${icon('chevronRight')}
            <span aria-current="page">${escapeHtml(p.name)}</span>
        </nav>

        <div class="detail-title-row">
            <div>
                <h1>${escapeHtml(p.name)}</h1>
                <div class="detail-meta">
                    <span class="badge badge-blue">${icon(p.theme.icon)}${escapeHtml(p.category)}</span>
                    <span>${icon('pin')}${escapeHtml(p.region)}, Sri Lanka</span>
                    ${rating ? `<span class="rating">${icon('star')}${rating.toFixed(1)}
                        <small>(${reviewCount(p)} reviews)</small></span>`
                        : (reviewCount(p) ? `<span>${icon('message')}${reviewCount(p)} reviews</span>` : '')}
                    ${durationText(p.duration_min) ? `<span>${icon('clock')}~${durationText(p.duration_min)} visit</span>` : ''}
                </div>
            </div>
            <div class="detail-actions">
                <button class="btn-icon" type="button" data-fav="${escapeHtml(p.id)}"
                        aria-pressed="${Wishlist.has(p.id)}"
                        aria-label="Save ${escapeHtml(p.name)} to your places">${icon('heart')}</button>
                <button class="btn-icon" type="button" id="share-btn" aria-label="Share this destination">
                    ${icon('share')}
                </button>
            </div>
        </div>`;
    }

    function overviewHtml(p) {
        const text = p.description || p.tagline;
        if (!text) return '';
        return `<section class="detail-block">
            <h2>${icon('book')} Overview</h2>
            <p class="detail-lead">${escapeHtml(text)}</p>
        </section>`;
    }

    function factsHtml(p) {
        const u = p.ugc || {};
        const facts = [];

        if (u.best_time) {
            const bt = u.best_time;
            const sub = [];
            if (bt.months?.length) sub.push(bt.months.join(', '));
            if (bt.season) sub.push(titleCase(bt.season));
            if (bt.avoid?.length) sub.push(`avoid ${bt.avoid.join(', ').toLowerCase()}`);
            facts.push(fact('sun', 'Best time to visit',
                bt.time_of_day ? titleCase(bt.time_of_day) : (bt.season ? titleCase(bt.season) : 'See reviews'),
                sub.join(' · '), bt.confidence, bt.based_on));
        }
        if (u.crowd) {
            const cr = u.crowd;
            const sub = [];
            if (cr.avg_level) sub.push(`${cr.avg_level.toFixed(1)}/5 average`);
            if (cr.busiest_period?.length) sub.push(`busiest ${cr.busiest_period.join(', ').toLowerCase()}`);
            facts.push(fact('users', 'How busy it gets',
                CROWD_TEXT[cr.label] || titleCase(cr.label), sub.join(' · '), cr.confidence, cr.based_on));
        }
        if (u.cost) {
            const co = u.cost;
            const sub = [];
            if (co.median_lkr) sub.push(`median ${co.median_lkr} LKR`);
            if (co.fee_type) sub.push(titleCase(co.fee_type));
            facts.push(fact('wallet', 'Estimated cost',
                COST_TEXT[co.level] || titleCase(co.level), sub.join(' · '), co.confidence, co.based_on));
        }
        if (p.duration_min) {
            facts.push(fact('clock', 'Suggested duration', `~${durationText(p.duration_min)}`,
                'Typical time on site, used by the trip planner'));
        }

        if (!facts.length) return '';
        return `<section class="detail-block">
            <h2>${icon('sparkles')} What the reviews revealed</h2>
            <p class="muted" style="margin-bottom:var(--sp-4)">
                Mined from ${reviewCount(p) || 'the'} traveller reviews. Confidence reflects how
                consistently reviewers agreed.
            </p>
            <div class="fact-grid">${facts.join('')}</div>
        </section>`;
    }

    function fact(iconName, label, value, sub, confidence, basedOn) {
        const pct = confidence != null ? Math.round(confidence * 100) : null;
        return `<div class="fact">
            <div class="fact-head">${icon(iconName)}${escapeHtml(label)}</div>
            <div class="fact-value">${escapeHtml(value || '—')}</div>
            ${sub ? `<div class="fact-sub">${escapeHtml(sub)}</div>` : ''}
            ${pct != null ? `
                <div class="conf-bar"><div class="conf-fill" style="width:${pct}%"></div></div>
                <div class="fact-sub">${pct}% confidence${basedOn ? ` · ${basedOn} mentions` : ''}</div>` : ''}
        </div>`;
    }

    function listSection(title, iconName, items) {
        if (!items || !items.length) return '';
        return `<section class="detail-block">
            <h2>${icon(iconName)} ${escapeHtml(title)}</h2>
            <div class="tag-cloud">
                ${items.slice(0, 14).map(i => `
                    <span class="tag ${i.sentiment === 'positive' ? 'positive'
                        : i.sentiment === 'negative' ? 'negative' : ''}">
                        <b>${escapeHtml(i.name)}</b><span class="tag-pct">${i.pct}%</span>
                    </span>`).join('')}
            </div>
        </section>`;
    }

    /* Tips are derived from the same mined facts — never invented advice. */
    function tipsHtml(p) {
        const u = p.ugc || {};
        const tips = [];

        const bt = u.best_time;
        if (bt?.time_of_day) {
            tips.push(`Aim to arrive in the ${titleCase(bt.time_of_day).toLowerCase()} — that is when
                       reviewers consistently had the best experience.`);
        }
        if (bt?.avoid?.length) {
            tips.push(`Reviewers suggest avoiding ${bt.avoid.join(', ').toLowerCase()}.`);
        }
        if (bt?.months?.length) {
            tips.push(`${bt.months.join(', ')} came up most often as the right months to go.`);
        }
        if (u.crowd && ['BUSY', 'PACKED'].includes(u.crowd.label)) {
            tips.push('Expect company. Going early or late in the day is the usual advice for busy sites.');
        }
        // Cost level and median amount are mined separately and occasionally
        // disagree, so only ever emit one of these two tips.
        if (u.cost?.level === 'FREE' && !u.cost?.median_lkr) {
            tips.push('No entry fee was reported, so this one is easy to slot into any day plan.');
        } else if (u.cost?.median_lkr) {
            tips.push(`Budget around ${u.cost.median_lkr} LKR per person for entry — carry cash, card
                       acceptance is inconsistent outside the cities.`);
        }
        if (p.duration_min) {
            tips.push(`Allow about ${durationText(p.duration_min)} on site when you build your itinerary.`);
        }
        if (!tips.length) return '';

        return `<section class="detail-block">
            <h2>${icon('info')} Travel tips</h2>
            <ul class="tip-list">
                ${tips.slice(0, 5).map(t => `<li>${icon('check')}<span>${escapeHtml(t.replace(/\s+/g, ' '))}</span></li>`).join('')}
            </ul>
        </section>`;
    }

    function evidenceHtml(p) {
        const u = p.ugc || {};
        const quotes = [
            ...(u.best_time?.evidence || []),
            ...(u.crowd?.evidence || []),
            ...(u.cost?.evidence || []),
        ].slice(0, 6);
        if (!quotes.length) return '';

        return `<section class="detail-block">
            <h2>${icon('quote')} Straight from the reviews</h2>
            ${quotes.map(q => `<blockquote class="quote">“${escapeHtml(q.text)}”
                <span class="quote-meta">${q.rating ? '★'.repeat(q.rating) : ''} ${escapeHtml(q.date || '')}</span>
            </blockquote>`).join('')}
        </section>`;
    }

    function mapHtml(p) {
        if (p.lat == null || p.lon == null) return '';
        return `<section class="detail-block">
            <h2>${icon('map')} Where it is</h2>
            <div class="detail-map"><div id="mini-map" role="application"
                aria-label="Map showing ${escapeHtml(p.name)}"></div></div>
            <p class="muted" style="margin-top:var(--sp-3)">
                <a href="https://www.google.com/maps/search/?api=1&query=${p.lat},${p.lon}"
                   target="_blank" rel="noopener">Open in Google Maps ${icon('external')}</a>
            </p>
        </section>`;
    }

    function similarHtml(p, all) {
        const similar = all
            .filter(x => x.id !== p.id && x.image)
            .map(x => {
                let affinity = 0;
                if (x.theme.key === p.theme.key) affinity += 3;
                if (x.region === p.region) affinity += 2;
                if (x.ugc?.best_time?.time_of_day &&
                    x.ugc.best_time.time_of_day === p.ugc?.best_time?.time_of_day) affinity += 1;
                return { x, affinity, pct: matchScore(x, null) };
            })
            .filter(o => o.affinity > 0)
            .sort((a, b) => (b.affinity - a.affinity) || (b.pct - a.pct))
            .slice(0, 3);

        if (!similar.length) return '';
        return `<section class="detail-block">
            <h2>${icon('compass')} Similar destinations</h2>
            <div class="card-grid">
                ${similar.map(o => placeCard(o.x, { match: o.pct, compact: true })).join('')}
            </div>
        </section>`;
    }

    function sourcesHtml(p) {
        const links = [];
        if (p.wikipedia_url) {
            links.push(`<li>${icon('external')}<span><a href="${escapeHtml(p.wikipedia_url)}" target="_blank"
                rel="noopener">Wikipedia article</a> — description and background.</span></li>`);
        }
        links.push(`<li>${icon('message')}<span>${reviewCount(p) || 'Traveller'} reviews mined for timing,
            crowd and cost signals.</span></li>`);
        if (p.image_credit) {
            links.push(`<li>${icon('camera')}<span>Photo: ${escapeHtml(p.image_credit)} via Wikimedia
                Commons.</span></li>`);
        }
        return `<section class="detail-block">
            <h2>${icon('shield')} Sources</h2>
            <ul class="tip-list">${links.join('')}</ul>
        </section>`;
    }

    function summaryHtml(p, pct, rating) {
        const rows = [];
        const push = (iconName, label, value) => {
            if (!value) return;
            rows.push(`<div class="summary-row">
                <span>${icon(iconName)}${escapeHtml(label)}</span><b>${escapeHtml(value)}</b>
            </div>`);
        };
        push('star', 'Traveller rating', rating ? `${rating.toFixed(1)} / 5` : null);
        push('sun', 'Best time', p.ugc?.best_time?.time_of_day ? titleCase(p.ugc.best_time.time_of_day) : null);
        push('users', 'Crowd level', p.ugc?.crowd?.label ? (CROWD_TEXT[p.ugc.crowd.label] || titleCase(p.ugc.crowd.label)) : null);
        push('wallet', 'Typical cost', priceText(p));
        push('clock', 'Time on site', durationText(p.duration_min));

        return `<div class="summary-card">
            <div class="summary-match">
                ${matchRing(pct, true)}
                <p><b>${matchWord(pct)}</b>Based on rating, review volume and how complete our insight is.</p>
            </div>
            <div class="summary-rows">${rows.join('')}</div>
            <div class="summary-actions">
                <a class="btn btn-primary btn-block"
                   href="planner.html?must_visit=${encodeURIComponent(p.name)}">
                    ${icon('calendar')} Add to trip plan
                </a>
                <a class="btn btn-secondary btn-block"
                   href="chat.html?q=${encodeURIComponent('Tell me about visiting ' + p.name)}">
                    ${icon('message')} Ask the AI about it
                </a>
                <button class="btn btn-quiet btn-block" type="button" data-fav="${escapeHtml(p.id)}"
                        aria-pressed="${Wishlist.has(p.id)}">
                    ${icon('heart')} Save to my places
                </button>
            </div>
        </div>`;
    }

    /* ================================================================ map */
    function initMiniMap() {
        const holder = document.getElementById('mini-map');
        if (!holder || place.lat == null || typeof L === 'undefined') return;

        const url = () => document.documentElement.getAttribute('data-theme') === 'dark'
            ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
            : 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png';

        miniMap = L.map(holder, { scrollWheelZoom: false }).setView([place.lat, place.lon], 11);
        const tiles = L.tileLayer(url(), { attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 19 })
            .addTo(miniMap);
        document.addEventListener('themechange', () => tiles.setUrl(url()));

        L.marker([place.lat, place.lon], {
            icon: L.divIcon({
                className: '',
                html: `<div class="poi-marker" style="background:${place.theme.color}">
                         <span>${icon(place.theme.icon)}</span></div>`,
                iconSize: [30, 30],
                iconAnchor: [15, 30],
            }),
            title: place.name,
        }).addTo(miniMap);
    }

    /* ============================================================ actions */
    function wireActions() {
        const share = document.getElementById('share-btn');
        if (!share) return;

        share.addEventListener('click', async () => {
            const data = {
                title: place.name,
                text: `${place.name} — ${place.tagline || 'a destination in ' + place.region}`,
                url: location.href,
            };
            try {
                if (navigator.share) { await navigator.share(data); return; }
                await navigator.clipboard.writeText(location.href);
                flash(share, 'Link copied');
            } catch (err) {
                if (err && err.name === 'AbortError') return;   // user dismissed the sheet
                flash(share, 'Could not share');
            }
        });
    }

    function flash(btn, message) {
        const note = document.createElement('span');
        note.className = 'badge badge-teal';
        note.textContent = message;
        note.style.cssText = 'position:absolute;transform:translate(-100%,44px);white-space:nowrap';
        btn.style.position = 'relative';
        btn.appendChild(note);
        setTimeout(() => note.remove(), 1800);
    }
})();
