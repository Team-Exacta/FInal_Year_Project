/* ==========================================================================
   Saved places (wishlist)
   Reads the ids kept by LT.Wishlist and resolves them against the dataset, so
   a saved place that later leaves the corpus simply drops out of the list
   instead of rendering a broken card.
   ========================================================================== */
(function () {
    'use strict';

    const {
        DATA_URL, themeFor, placeCard, skeletonCards, stateHtml, matchScore,
        Wishlist, syncWishlistBadges,
    } = window.LT;

    const grid = document.getElementById('saved-grid');
    const countEl = document.getElementById('saved-count');
    const planBtn = document.getElementById('plan-all');
    const clearBtn = document.getElementById('clear-saved');
    let ALL = [];

    document.addEventListener('DOMContentLoaded', boot);

    async function boot() {
        if (!Wishlist.count()) { renderEmpty(); wire(); return; }

        grid.innerHTML = skeletonCards(3);
        try {
            const res = await fetch(DATA_URL);
            if (!res.ok) throw new Error('HTTP ' + res.status);
            const payload = await res.json();
            ALL = (payload.places || []).map(p => ({ ...p, theme: themeFor(p.category) }));
        } catch (err) {
            console.error('Failed to load destination data:', err);
            grid.innerHTML = stateHtml({
                tone: 'error',
                icon: 'alert',
                title: 'Could not load your saved places',
                body: 'The dataset did not respond, so we cannot show the details for your saved ids.',
                action: '<button class="btn btn-primary" type="button" onclick="location.reload()">Try again</button>',
            });
            return;
        }

        render();
        wire();
        document.addEventListener('wishlistchange', render);
    }

    function saved() {
        const ids = Wishlist.all();
        return ids.map(id => ALL.find(p => p.id === id)).filter(Boolean);
    }

    function render() {
        const list = saved();
        countEl.textContent = String(list.length);

        if (!list.length) { renderEmpty(); return; }

        grid.innerHTML = list.map(p => placeCard(p, { match: matchScore(p, null) })).join('');
        planBtn.href = 'planner.html?must_visit=' +
            encodeURIComponent(list.map(p => p.name).join(', '));
        planBtn.hidden = false;
        clearBtn.hidden = false;
        syncWishlistBadges();
    }

    function renderEmpty() {
        countEl.textContent = '0';
        if (planBtn) planBtn.hidden = true;
        if (clearBtn) clearBtn.hidden = true;
        grid.innerHTML = stateHtml({
            icon: 'heart',
            title: 'Nothing saved yet',
            body: 'Tap the heart on any destination and it will wait for you here — '
                + 'then send the whole list to the trip planner in one go.',
            action: '<a class="btn btn-primary" href="explore.html">Find places to save</a>',
        });
    }

    function wire() {
        if (!clearBtn) return;
        clearBtn.addEventListener('click', () => {
            if (!Wishlist.count()) return;
            Wishlist.clear();
            render();
        });
    }
})();
