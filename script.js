// Live project statistics.
//
// Numbers are baked into stats.json by .github/workflows/stats.yml, which runs
// daily against the authenticated GitHub API. The page therefore makes zero
// third-party requests on a normal load and is not subject to any rate limit.
//
// The GitHub API fallback below only runs if stats.json cannot be read at all
// (e.g. the file was not deployed). It is unauthenticated and capped at 60
// requests/hour/IP, so it is a safety net, not the primary path.
//
// Contract with index.html — an element opts in by carrying:
//   data-stat="stars"     + data-repo="kornia/kornia"
//   data-stat="downloads" + data-pkg="kornia"
//   data-stat="contributors" / "dependents" + data-repo="kornia/kornia"
//   data-stat="playground_ops" / "playground_live"   (counts from playground/registry.json)
// The value is written into a descendant .count if one exists, otherwise into
// the element's own text. Elements without data-stat are never touched.
//
// Invariant: if every source fails, whatever is already in the HTML stays put.
// A value is only ever written after passing isCount(), so "undefined"/"NaN"
// can never reach the page.

const STATS_URL = new URL('stats.json', document.currentScript ? document.currentScript.src : location.href).href;
const CACHE_KEY = 'kornia:stats:v1';
const CACHE_TTL_MS = 6 * 60 * 60 * 1000; // 6 hours

// Only a finite, non-negative integer is allowed to reach the DOM.
function isCount(value) {
    return typeof value === 'number' && Number.isFinite(value) && value >= 0;
}

// 11297 -> "11.3k", 12000 -> "12k", 682 -> "682"
function formatStars(count) {
    if (count < 1000) {
        return String(count);
    }
    const thousands = (count / 1000).toFixed(1);
    return thousands.replace(/\.0$/, '') + 'k';
}

// 2991596 -> "2.9M+". Truncated rather than rounded so the figure is never
// overstated; 2991596 must not be advertised as "3M+".
function formatDownloads(count) {
    if (count < 1000) {
        return String(count);
    }
    if (count < 1000000) {
        return Math.floor(count / 1000) + 'K+';
    }
    const millions = (Math.floor(count / 100000) / 10).toFixed(1);
    return millions.replace(/\.0$/, '') + 'M+';
}

// The lookup key for an element, or null if it is not a well-formed target.
function statKey(element) {
    const stat = element.dataset.stat;
    if (stat === 'stars' || stat === 'contributors' || stat === 'dependents') {
        return element.dataset.repo || null;
    }
    if (stat === 'downloads') {
        return element.dataset.pkg || null;
    }
    if (stat === 'playground_ops' || stat === 'playground_live') {
        return 'playground';
    }
    return null;
}

function render(element, value) {
    if (!isCount(value)) {
        return; // leave the hardcoded HTML in place
    }
    const kind = element.dataset.stat;
    let text;
    if (kind === 'downloads') text = formatDownloads(value);
    else if (kind === 'stars' || kind === 'dependents') text = formatStars(value) + (kind === 'dependents' ? '+' : '');
    else text = value.toLocaleString('en-US'); // contributors, playground counts
    const slot = element.querySelector('.count') || element;
    slot.textContent = text;
}

// Pull a value out of the stats.json shape, tolerating a partial file.
function lookup(stats, element) {
    const key = statKey(element);
    if (!key || !stats) {
        return undefined;
    }
    const kind = element.dataset.stat;
    if (kind === 'stars' || kind === 'contributors' || kind === 'dependents') {
        const entry = stats.repos && stats.repos[key];
        return entry ? entry[kind] : undefined;
    }
    if (kind === 'playground_ops' || kind === 'playground_live') {
        const entry = stats.playground || {};
        return entry[kind === 'playground_ops' ? 'operators' : 'live'];
    }
    const entry = stats.packages && stats.packages[key];
    return entry ? entry.downloads_last_month : undefined;
}

async function loadLocalStats() {
    const response = await fetch(STATS_URL, { cache: 'no-cache' });
    if (!response.ok) {
        throw new Error(`${STATS_URL} responded ${response.status}`);
    }
    return response.json();
}

function readCache() {
    try {
        const raw = sessionStorage.getItem(CACHE_KEY);
        if (!raw) {
            return null;
        }
        const cached = JSON.parse(raw);
        if (!cached || (Date.now() - cached.at) > CACHE_TTL_MS) {
            return null;
        }
        return cached.stars;
    } catch (error) {
        return null; // private mode / disabled storage / corrupt entry
    }
}

function writeCache(stars) {
    try {
        sessionStorage.setItem(CACHE_KEY, JSON.stringify({ at: Date.now(), stars }));
    } catch (error) {
        /* storage unavailable or full — the cache is optional */
    }
}

// Unauthenticated GitHub API, used only when stats.json is unreachable.
// A rate-limited reply is a *successful* 403 response, not a thrown error, so
// response.ok must be checked explicitly — that check is the whole point of
// this function.
async function fetchStars(repo) {
    try {
        const response = await fetch(`https://api.github.com/repos/${repo}`);
        if (!response.ok) {
            return [repo, undefined]; // 403 rate limit, 404, 5xx
        }
        const data = await response.json();
        return [repo, isCount(data.stargazers_count) ? data.stargazers_count : undefined];
    } catch (error) {
        return [repo, undefined]; // network/DNS/CORS
    }
}

async function fetchAllStars(repos) {
    const cached = readCache();
    if (cached) {
        return cached;
    }
    // Concurrent, not the serial await-in-loop this file used to run.
    const entries = await Promise.all(repos.map(fetchStars));
    const stars = {};
    for (const [repo, count] of entries) {
        if (isCount(count)) {
            stars[repo] = count;
        }
    }
    if (Object.keys(stars).length > 0) {
        writeCache(stars);
    }
    return stars;
}

async function updateStats() {
    const elements = Array.from(document.querySelectorAll('[data-stat]'));
    if (elements.length === 0) {
        return;
    }

    // Primary path: the committed, build-time file.
    const resolved = new Set();
    try {
        const stats = await loadLocalStats();
        for (const element of elements) {
            const value = lookup(stats, element);
            if (isCount(value)) {
                render(element, value);
                resolved.add(element);
            }
        }
        if (resolved.size === elements.length) {
            return; // everything resolved; no third-party request needed
        }
    } catch (error) {
        console.warn('stats.json unavailable, falling back to GitHub API:', error);
    }

    // Fallback: only the star elements stats.json did not resolve, which are
    // still showing their hardcoded value. Downloads have no GitHub equivalent,
    // so they simply keep the HTML value.
    const pending = elements.filter(
        el => !resolved.has(el) && el.dataset.stat === 'stars' && statKey(el)
    );
    if (pending.length === 0) {
        return;
    }
    const repos = Array.from(new Set(pending.map(statKey)));
    const stars = await fetchAllStars(repos);
    for (const element of pending) {
        render(element, stars[statKey(element)]);
    }
}

document.addEventListener('DOMContentLoaded', updateStats);
