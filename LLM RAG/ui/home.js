/* ==========================================================================
   Homepage
   Populates every data-driven section from the same dataset the Explore page
   uses, so the marketing surface can never drift from the real corpus.
   ========================================================================== */
(function () {
    'use strict';

    const {
        DATA_URL, THEMES, themeFor, placeCard, skeletonCards, stateHtml,
        matchScore, escapeHtml, icon, ratingOf, reviewCount, syncWishlistBadges,
    } = window.LT;

    const grids = {
        popular: document.getElementById('popular-grid'),
        recommended: document.getElementById('recommended-grid'),
        trending: document.getElementById('trending-grid'),
    };

    /* skeletons while the JSON is in flight */
    Object.values(grids).forEach(g => { if (g) g.innerHTML = skeletonCards(4); });

    document.addEventListener('DOMContentLoaded', boot);

    async function boot() {
        wireHeroSearch(null);

        let payload;
        try {
            const res = await fetch(DATA_URL);
            if (!res.ok) throw new Error('HTTP ' + res.status);
            payload = await res.json();
        } catch (err) {
            console.error('Failed to load destination data:', err);
            showLoadError();
            return;
        }

        const places = (payload.places || []).map(p => ({ ...p, theme: themeFor(p.category) }));
        renderStats(payload, places);
        renderCategories(places);
        renderPopular(places);
        renderRecommended(places);
        renderTrending(places);
        renderQuotes(places);
        wireHeroSearch(places);
        syncWishlistBadges();
    }

    /* ---------------------------------------------------------------- stats */
    function renderStats(payload, places) {
        const counts = payload.counts || {};
        const reviews = places.reduce((sum, p) => sum + reviewCount(p), 0);

        const set = (id, value) => {
            const el = document.getElementById(id);
            if (el && value != null) el.textContent = value;
        };
        set('stat-places', counts.places != null ? counts.places : places.length);
        set('stat-reviews', reviews ? reviews.toLocaleString() : '—');
        set('stat-timing', counts.with_best_time);
        set('stat-cost', counts.with_cost);

        const eyebrow = document.getElementById('eyebrow-reviews');
        if (eyebrow && reviews) eyebrow.textContent = reviews.toLocaleString();
    }

    /* ----------------------------------------------------------- categories */
    function renderCategories(places) {
        const box = document.getElementById('category-grid');
        if (!box) return;

        box.innerHTML = THEMES.map(t => {
            const n = places.filter(p => p.theme.key === t.key).length;
            if (!n) return '';
            return `<a class="cat-tile" href="explore.html?theme=${t.key}">
                        <span class="cat-tile-icon" style="color:${t.color};background:${t.color}1a">
                            ${icon(t.icon)}
                        </span>
                        <b>${escapeHtml(t.label)}</b>
                        <span>${n} ${n === 1 ? 'place' : 'places'}</span>
                    </a>`;
        }).join('');
    }

    /* -------------------------------------------------------------- sections */
    function renderPopular(places) {
        const top = [...places]
            .filter(p => p.image)
            .sort((a, b) => reviewCount(b) - reviewCount(a))
            .slice(0, 4);
        fill(grids.popular, top, false);
    }

    function renderRecommended(places) {
        // No filters are set on the homepage, so the score reflects pure quality:
        // rating × evidence volume × how complete the review insight is.
        const scored = places
            .filter(p => p.image && p.ugc && p.ugc.best_time)
            .map(p => ({ p, pct: matchScore(p, null) }))
            .sort((a, b) => b.pct - a.pct)
            .slice(0, 4);

        if (!grids.recommended) return;
        grids.recommended.innerHTML = scored.length
            ? scored.map(x => placeCard(x.p, { match: x.pct })).join('')
            : stateHtml({ icon: 'compass', title: 'No recommendations yet' });
    }

    function renderTrending(places) {
        const top = [...places]
            .filter(p => p.image && ratingOf(p) && reviewCount(p) >= 40)
            .sort((a, b) => (ratingOf(b) - ratingOf(a)) || (reviewCount(b) - reviewCount(a)))
            .slice(0, 4);
        fill(grids.trending, top, false);
    }

    function fill(grid, list, withMatch) {
        if (!grid) return;
        grid.innerHTML = list.length
            ? list.map(p => placeCard(p, { match: withMatch ? matchScore(p, null) : null })).join('')
            : stateHtml({ icon: 'compass', title: 'Nothing to show yet' });
    }

    /* ---------------------------------------------------------------- quotes */
    function renderQuotes(places) {
        const box = document.getElementById('quote-grid');
        if (!box) return;

        // Real review sentences from the corpus, longest-first so the cards read
        // as opinions rather than fragments.
        const pool = [];
        places.forEach(p => {
            const u = p.ugc || {};
            [].concat(
                (u.best_time && u.best_time.evidence) || [],
                (u.crowd && u.crowd.evidence) || [],
            ).forEach(q => {
                if (q && q.text && q.text.length > 70 && q.text.length < 190 && q.rating >= 4) {
                    pool.push({ q, place: p });
                }
            });
        });

        const picked = pool
            .sort((a, b) => b.q.text.length - a.q.text.length)
            .filter((x, i, arr) => arr.findIndex(y => y.place.id === x.place.id) === i)
            .slice(0, 3);

        if (!picked.length) { box.closest('section').hidden = true; return; }

        box.innerHTML = picked.map(({ q, place }) => `
            <figure class="quote-card">
                <span class="rating" aria-label="${q.rating} out of 5">
                    ${icon('star').repeat(Math.min(5, q.rating || 5))}
                </span>
                <blockquote>“${escapeHtml(q.text)}”</blockquote>
                <figcaption class="quote-who">
                    <span class="quote-avatar" aria-hidden="true">${escapeHtml(initials(place.name))}</span>
                    <span>
                        <b>Verified review</b>
                        <span>${escapeHtml(place.name)} · ${escapeHtml(q.date || 'Recent')}</span>
                    </span>
                </figcaption>
            </figure>`).join('');
    }

    const initials = name => String(name || '?').split(/\s+/).slice(0, 2).map(w => w[0]).join('').toUpperCase();

    /* ---------------------------------------------------- hero search form */
    function wireHeroSearch(places) {
        const form = document.getElementById('hero-search');
        if (!form || form.dataset.wired) return;

        const typeSelect = document.getElementById('hs-type');
        if (typeSelect && places) {
            THEMES.forEach(t => {
                if (!places.some(p => p.theme.key === t.key)) return;
                const opt = document.createElement('option');
                opt.value = t.key;
                opt.textContent = t.label;
                typeSelect.appendChild(opt);
            });
        }
        if (!places) return;              // wait for data before accepting a submit

        form.dataset.wired = '1';
        form.addEventListener('submit', e => {
            e.preventDefault();
            const params = new URLSearchParams();
            const q = form.querySelector('[name=q]').value.trim();
            const theme = form.querySelector('[name=theme]').value;
            const cost = form.querySelector('[name=cost]').value;

            if (q) params.set('q', q);
            if (theme) params.set('theme', theme);
            if (cost === 'FREE') params.set('insight', 'cost:free');
            else if (cost === 'LOW') params.set('insight', 'cost:cheap');
            else if (cost) params.set('cost', cost);

            location.href = 'explore.html' + (params.toString() ? '?' + params : '');
        });
    }

    /* ----------------------------------------------------------- error state */
    function showLoadError() {
        const html = stateHtml({
            tone: 'error',
            icon: 'alert',
            title: 'Could not load destination data',
            body: 'The dataset did not respond. Check your connection, or rebuild it with '
                + '<code>python scripts/build_explore_data.py</code>.',
            action: '<button class="btn btn-primary" type="button" onclick="location.reload()">Try again</button>',
        });
        Object.values(grids).forEach(g => { if (g) g.innerHTML = html; });
    }
})();
