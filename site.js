// Shared behaviour for every page: hamburger, in-page smooth scroll, obfuscated contact email, copy buttons.
document.addEventListener('DOMContentLoaded', function () {
    // current section in the header, from the URL (the header markup is identical on every page)
    var path = location.pathname;
    document.querySelectorAll('.nav-links a[data-section]').forEach(function (a) {
        var on = path.indexOf('/' + a.dataset.section + '/') !== -1;
        a.classList.toggle('active', on);
        if (on) a.setAttribute('aria-current', 'page');
    });
    if (/\/projects\//.test(path)) { var p = document.querySelector('.nav-dropdown a[href$="projects/"]'); if (p) p.classList.add('active'); }
    if (/\/(sponsor|community)\//.test(path)) { var s = document.querySelector('.nav-dropdown a[href$="sponsor/"]'); if (s) s.classList.add('active'); }

    var heroNews = document.getElementById('hero-news');
    if (heroNews) {
        fetch('news.json', { cache: 'no-cache' }).then(function (r) { return r.json(); }).then(function (news) {
            var items = (news.items || []).slice().sort(function (a, b) { return new Date(b.date) - new Date(a.date); });
            if (!items.length) return;
            heroNews.querySelector('.hero-news-title').textContent = items[0].title;
            heroNews.querySelector('.hero-news-label').innerHTML = '<i class="fas fa-bolt" aria-hidden="true"></i> ' + items[0].date;
        }).catch(function () { /* keep the static line */ });
    }

    var emailSlot = document.getElementById('email-contact');
    if (emailSlot) {
        var addr = 'hello' + '@' + 'kornia.org';
        emailSlot.innerHTML = '<a href="mailto:' + addr + '">' + addr + '</a>';
    }

    document.querySelectorAll('a[href^="#"]').forEach(function (link) {
        link.addEventListener('click', function (e) {
            var targetId = this.getAttribute('href');
            if (targetId === '#') return;
            var target = document.querySelector(targetId);
            if (!target) return;
            e.preventDefault();
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            history.pushState(null, null, targetId);
        });
    });

    var navToggle = document.querySelector('.nav-toggle');
    var navLinks = document.getElementById('primary-nav-links');
    function closeNav() {
        if (!navToggle || !navLinks) return;
        navLinks.classList.remove('is-open');
        navToggle.classList.remove('is-active');
        navToggle.setAttribute('aria-expanded', 'false');
    }
    if (navToggle && navLinks) {
        navToggle.addEventListener('click', function (e) {
            e.stopPropagation();
            var open = navLinks.classList.toggle('is-open');
            navToggle.classList.toggle('is-active', open);
            navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
        });
        // a dropdown trigger opens its menu in place; every other link closes the panel
        navLinks.querySelectorAll('a').forEach(function (a) { if (a.getAttribute('aria-haspopup')) return; a.addEventListener('click', closeNav); });
        document.addEventListener('click', function (e) { if (!e.target.closest('nav')) closeNav(); });
        document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeNav(); });
    }

    // Nav dropdowns: hover on desktop via CSS; click/tap toggles, outside click and Escape close.
    var dropdowns = Array.prototype.slice.call(document.querySelectorAll('.nav-dropdown'));
    function closeDropdowns(except) {
        dropdowns.forEach(function (dd) {
            if (dd === except) return;
            dd.classList.remove('is-open');
            var trig = dd.querySelector('a[aria-haspopup="menu"]');
            if (trig) trig.setAttribute('aria-expanded', 'false');
        });
    }
    dropdowns.forEach(function (dd) {
        var trigger = dd.querySelector('a[aria-haspopup="menu"]');
        if (!trigger) return;
        trigger.addEventListener('click', function (e) {
            // desktop: hover shows the menu, a click goes to the section's page (Projects, Docs, Sponsor);
            // in the mobile menu there is no hover, so the first tap opens and the second follows the link
            var mobile = window.matchMedia('(max-width: 1024px)').matches;
            var noPage = trigger.getAttribute('href') === '#';   // Docs has no page of its own: a click only toggles
            if (!noPage && (!mobile || dd.classList.contains('is-open'))) return;
            if (noPage && dd.classList.contains('is-open')) { e.preventDefault(); closeDropdowns(); return; }
            e.preventDefault();
            e.stopPropagation();
            dd.classList.add('is-open');
            trigger.setAttribute('aria-expanded', 'true');
            closeDropdowns(dd);
        });
    });
    document.addEventListener('click', function (e) { if (!e.target.closest('.nav-dropdown')) closeDropdowns(); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeDropdowns(); });

    // Code tabs: [data-tabs] holds [data-tab] buttons and [data-panel] panels with matching names.
    document.querySelectorAll('[data-tabs]').forEach(function (box) {
        var buttons = box.querySelectorAll('[data-tab]');
        buttons.forEach(function (btn) {
            btn.addEventListener('click', function () {
                buttons.forEach(function (b) { b.setAttribute('aria-selected', b === btn ? 'true' : 'false'); });
                box.querySelectorAll('[data-panel]').forEach(function (panel) { panel.hidden = panel.dataset.panel !== btn.dataset.tab; });
            });
        });
    });

    document.querySelectorAll('.copy-btn').forEach(function (btn) {
        btn.addEventListener('click', function () {
            var src = document.getElementById(btn.dataset.copyTarget);
            if (!src || !navigator.clipboard) return;
            navigator.clipboard.writeText(src.textContent.trim()).then(function () {
                var prev = btn.textContent;
                btn.textContent = 'Copied';
                setTimeout(function () { btn.textContent = prev; }, 1500);
            }).catch(function () { /* clipboard unavailable; the command stays selectable */ });
        });
    });
});
