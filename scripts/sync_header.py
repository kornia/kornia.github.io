#!/usr/bin/env python
"""One site header and footer for every page.

The header lives here as a template; this script rewrites the <header>...</header> block of every HTML
page (root pages, playground shells, generated operator and model pages, hub, robot, news) with the link
prefix each page needs. Run it after editing the template or adding a page:

    python scripts/sync_header.py

The current section is not baked into the pages: site.js adds `active` to the link that matches the URL.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OC = "https://opencollective.com/kornia"

PROJECTS = [
    ("kornia", "https://github.com/kornia/kornia", '<i class="fab fa-python" aria-hidden="true"></i>', "Differentiable vision for PyTorch"),
    ("kornia-rs", "https://github.com/kornia/kornia-rs", '<i class="fab fa-rust" aria-hidden="true"></i>', "Image and 3D kernels in Rust"),
    ("Bubbaloop", "https://github.com/kornia/bubbaloop", '<span aria-hidden="true">&#129412;</span>', "Vision agent runtime for Jetson and Pi"),
    ("kornia-slam", "https://github.com/kornia/kornia-slam", '<i class="fas fa-map-location-dot" aria-hidden="true"></i>', "Real-time pose and mapping"),
    ("vision-rt", "https://github.com/kornia/vision-rt", '<i class="fas fa-microchip" aria-hidden="true"></i>', "TensorRT inference on Jetson Orin"),
    ("sensor-rt", "https://github.com/kornia/sensor-rt", '<i class="fas fa-camera" aria-hidden="true"></i>', "Camera, stereo and IMU drivers"),
]


def docs_items(prefix: str) -> list[tuple[str, str, str, str]]:
    return [
        ("Kornia (PyTorch)", "https://kornia.readthedocs.io", '<i class="fab fa-python" aria-hidden="true"></i>', "API reference and guides"),
        ("kornia-rs", "https://docs.rs/kornia-imgproc", '<i class="fab fa-rust" aria-hidden="true"></i>', "Crate docs on docs.rs"),
        ("Bubbaloop", "https://www.kornia.org/bubbaloop/", '<span aria-hidden="true">&#129412;</span>', "Install and run an agent"),
        ("Tutorials", "https://www.kornia.org/tutorials/", '<i class="fas fa-graduation-cap" aria-hidden="true"></i>', "Runnable notebooks"),
    ]


def support_items(prefix: str) -> list[tuple[str, str, str, str]]:
    return [
        ("Sponsor Kornia", prefix + "sponsor/", '<i class="fas fa-heart" aria-hidden="true"></i>', "What funding buys, and the tiers"),
        ("Community", prefix + "community/", '<i class="fas fa-users" aria-hidden="true"></i>', "Get help, contribute, programmes"),
        ("Discord", "https://discord.gg/HfnywwpBnD", '<i class="fab fa-discord" aria-hidden="true"></i>', "Quick questions, same-day answers"),
        ("GitHub Discussions", "https://github.com/kornia/kornia/discussions", '<i class="fas fa-comments" aria-hidden="true"></i>', "Design questions and proposals"),
        ("Report a bug", "https://github.com/kornia/kornia/issues/new/choose", '<i class="fas fa-bug" aria-hidden="true"></i>', "Issue templates"),
    ]


def dropdown(label: str, href: str, items: list[tuple[str, str, str, str]], footer: tuple[str, str] | None) -> str:
    rows = "\n".join(
        f'''                        <a href="{url}"{' target="_blank" rel="noopener"' if url.startswith("http") else ""}>
                            <span class="nav-dd-icon">{icon}</span>
                            <span><span class="nav-dd-name">{name}</span><span class="nav-dd-desc">{desc}</span></span>
                        </a>'''
        for name, url, icon, desc in items
    )
    foot = f'\n                        <a class="nav-dd-all" href="{footer[1]}">{footer[0]} &rarr;</a>' if footer else ""
    return f'''                <div class="nav-dropdown">
                    <a href="{href}" role="button" aria-haspopup="menu" aria-expanded="false">{label} <i class="fas fa-chevron-down" aria-hidden="true"></i></a>
                    <div class="nav-dropdown-content">
{rows}{foot}
                    </div>
                </div>'''


def header(prefix: str) -> str:
    home = prefix if prefix else "./"
    return f'''    <header>
        <nav>
            <div class="logo">
                <a href="{home}">kornia.org</a>
            </div>

            <button type="button"
                    class="nav-toggle"
                    aria-label="Toggle navigation menu"
                    aria-expanded="false"
                    aria-controls="primary-nav-links">
                <span class="nav-toggle-bar"></span>
                <span class="nav-toggle-bar"></span>
                <span class="nav-toggle-bar"></span>
            </button>

            <div class="nav-links" id="primary-nav-links">
                <div class="nav-dropdown nav-mega">
                    <a href="{prefix}playground/" class="nav-experiment" data-section="playground" role="button" aria-haspopup="menu" aria-expanded="false"><i class="fas fa-flask" aria-hidden="true"></i> Experiment <i class="fas fa-chevron-down" aria-hidden="true"></i></a>
                    <div class="nav-dropdown-content nav-mega-content">
                        <a class="nav-card" href="{prefix}playground/">
                            <span class="nav-card-icon"><i class="fas fa-sliders" aria-hidden="true"></i></span>
                            <span class="nav-card-title">Vision operators</span>
                            <span class="nav-card-desc">Every kornia image operator running live in your browser, with the code in Python and Rust and an ONNX download.</span>
                        </a>
                        <a class="nav-card" href="{prefix}playground/models/object-detection-rtdetr/">
                            <span class="nav-card-icon"><i class="fas fa-brain" aria-hidden="true"></i></span>
                            <span class="nav-card-title">Models</span>
                            <span class="nav-card-desc">Detection, faces, depth, features, classification and edges, in your browser or on kornia's server.</span>
                        </a>
                        <a class="nav-card" href="{prefix}playground/pipelines/">
                            <span class="nav-card-icon"><i class="fas fa-diagram-project" aria-hidden="true"></i></span>
                            <span class="nav-card-title">Pipelines</span>
                            <span class="nav-card-desc">Chain operators and a model, run the chain on images or clips, export it as one ONNX graph.</span>
                        </a>
                        <a class="nav-card" href="{prefix}robot/" data-section="robot">
                            <span class="nav-card-icon"><i class="fas fa-robot" aria-hidden="true"></i></span>
                            <span class="nav-card-title">Robot simulator</span>
                            <span class="nav-card-desc">Simulated robots in MuJoCo, a car, an arm and a humanoid, with kornia operators on their cameras and the true depth beside them.</span>
                        </a>
                    </div>
                </div>
{dropdown("Projects", prefix + "projects/", PROJECTS, ("All projects, with install commands", prefix + "projects/"))}
{dropdown("Docs", "#", docs_items(prefix), None)}
{dropdown("Support", prefix + "sponsor/", support_items(prefix), None)}
                <button type="button" class="nav-icon nav-theme" aria-label="Switch theme"><i class="fas fa-moon" aria-hidden="true"></i></button>
                <a href="https://github.com/kornia" target="_blank" rel="noopener" class="nav-icon" aria-label="Kornia on GitHub"><i class="fab fa-github" aria-hidden="true"></i></a>
                <div class="nav-auth" id="nav-auth"><a class="btn nav-signin-btn" href="{prefix}dashboard/"><i class="fas fa-user" aria-hidden="true"></i> Sign in</a></div>
                <script>/* paint the remembered signed-in state at parse time; auth.js confirms it once Firebase answers */(function(){{try{{var s=JSON.parse(localStorage.getItem("kornia-auth-state")||"null");if(!s)return;var c=document.getElementById("nav-auth");c.innerHTML='<a class="nav-avatar" href="{prefix}dashboard/" title="'+(s.displayName||s.email||"").replace(/"/g,"")+'"><img alt="" width="28" height="28" referrerpolicy="no-referrer" src="'+(s.photoURL||"{prefix}assets/kornia-logo-mini.svg").replace(/"/g,"")+'"></a>';}}catch(e){{}}}})();</script>
            </div>
        </nav>
    </header>'''


def footer(prefix: str) -> str:
    return f'''    <footer>
        <div class="footer-content">
            <div class="footer-section">
                <h4>Kornia</h4>
                <p>Computer vision for robotics &amp; spatial AI</p>
            </div>
            <div class="footer-section">
                <h4>Projects</h4>
                <a href="https://github.com/kornia/kornia">Kornia</a>
                <a href="https://github.com/kornia/kornia-rs">kornia-rs</a>
                <a href="https://github.com/kornia/kornia-slam">kornia-slam</a>
                <a href="https://github.com/kornia/vision-rt">vision-rt</a>
                <a href="https://github.com/kornia/sensor-rt">sensor-rt</a>
                <a href="https://github.com/kornia/bubbaloop">Bubbaloop</a>
            </div>
            <div class="footer-section">
                <h4>Resources</h4>
                <a href="https://kornia.readthedocs.io">Documentation</a>
                <a href="https://www.kornia.org/tutorials/">Tutorials</a>
                <a href="{prefix}playground/">Experiment</a>
                <a href="{prefix}robot/">Robot simulator</a>
                <a href="{prefix}news/">News &amp; updates</a>
                <a href="https://kornia.readthedocs.io/en/latest/get-started/governance.html">Governance</a>
            </div>
            <div class="footer-section">
                <h4>Contribute</h4>
                <a href="{prefix}community/#contribute">Start contributing</a>
                <a href="https://github.com/kornia/kornia/blob/main/CONTRIBUTING.md">Contributor guide</a>
                <a href="https://github.com/kornia/kornia/blob/main/CODE_OF_CONDUCT.md">Code of conduct</a>
                <a href="{prefix}sponsor/">Sponsor</a>
            </div>
            <div class="footer-section">
                <h4>Community</h4>
                <a href="https://discord.gg/HfnywwpBnD">Discord</a>
                <a href="https://x.com/kornia_foss">X</a>
                <a href="https://www.linkedin.com/company/kornia/">LinkedIn</a>
            </div>
            <div class="footer-section">
                <h4>Contact</h4>
                <span id="email-contact"></span>
            </div>
        </div>
        <div class="footer-bottom">
            <p>&copy; 2026 kornia.org. Sample images from the Kodak Lossless True Color Image Suite; clips from Wikimedia Commons (public domain and CC0, see <a href="{prefix}playground/CREDITS.md">credits</a>). In-browser demos run with <a href="https://onnxruntime.ai/docs/tutorials/web/" target="_blank" rel="noopener">onnxruntime-web</a>; the robot with <a href="https://mujoco.org" target="_blank" rel="noopener">MuJoCo</a> and <a href="https://threejs.org" target="_blank" rel="noopener">three.js</a>.</p>
        </div>
    </footer>'''


def main() -> None:
    pages = [p for p in ROOT.glob("*.html") if p.name not in ("404.html",)]
    pages += [Path(p) for p in glob.glob(str(ROOT / "*/index.html"))]
    pages += [Path(p) for p in glob.glob(str(ROOT / "playground/*/index.html"))]
    pages += [Path(p) for p in glob.glob(str(ROOT / "playground/*/*/index.html"))]
    # the page template lives in the private kornia-backend checkout (KORNIA_BACKEND, or a sibling directory)

    template = Path(os.environ.get("KORNIA_BACKEND", ROOT.parent / "kornia-backend")) / "build" / "op_template.html"

    if template.exists():

        pages.append(template)
    pat = re.compile(r"    <header>\n.*?    </header>", re.S)
    fpat = re.compile(r"    <footer>\n.*?    </footer>", re.S)
    n = 0
    for page in sorted(set(pages)):
        html = page.read_text()
        m = re.search(r'href="((?:\.\./)*)styles\.css', html)
        if not m or not pat.search(html):
            continue
        new = pat.sub(lambda _: header(m.group(1)), html, count=1)
        if fpat.search(new):
            new = fpat.sub(lambda _: footer(m.group(1)), new, count=1)
        if new != html:
            page.write_text(new)
            n += 1
    print(f"header and footer synced on {n} pages")


if __name__ == "__main__":
    main()
