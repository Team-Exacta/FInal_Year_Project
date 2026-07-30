/* ==========================================================================
   Light / dark theme switch.
   Loaded synchronously from <head> on every page so `data-theme` lands on
   <html> before the first paint — a deferred script would flash the wrong
   theme. Button wiring waits for DOMContentLoaded because the nav does not
   exist yet at that point.
   ========================================================================== */
(function () {
    'use strict';

    var KEY = 'lta-theme';
    var root = document.documentElement;

    var SUN = '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/>'
        + '<path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2'
        + 'M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4"/></svg>';
    var MOON = '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">'
        + '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>';

    function stored() {
        try { return localStorage.getItem(KEY); } catch (e) { return null; }
    }
    function remember(theme) {
        try { localStorage.setItem(KEY, theme); } catch (e) { /* private mode */ }
    }
    function prefersDark() {
        return !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
    }

    var saved = stored();
    var theme = (saved === 'light' || saved === 'dark') ? saved : (prefersDark() ? 'dark' : 'light');
    root.setAttribute('data-theme', theme);

    function paintButtons() {
        var label = theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme';
        var buttons = document.querySelectorAll('.theme-toggle');
        for (var i = 0; i < buttons.length; i++) {
            buttons[i].innerHTML = theme === 'dark' ? SUN : MOON;
            buttons[i].setAttribute('aria-label', label);
            buttons[i].setAttribute('title', label);
        }
    }

    function toggle() {
        theme = theme === 'dark' ? 'light' : 'dark';
        root.setAttribute('data-theme', theme);
        remember(theme);
        paintButtons();
        // explore.js / place.js listen for this to re-tint the Leaflet basemap
        document.dispatchEvent(new CustomEvent('themechange', { detail: { theme: theme } }));
    }

    document.addEventListener('DOMContentLoaded', function () {
        paintButtons();
        var buttons = document.querySelectorAll('.theme-toggle');
        for (var i = 0; i < buttons.length; i++) {
            buttons[i].addEventListener('click', toggle);
        }
    });
})();
