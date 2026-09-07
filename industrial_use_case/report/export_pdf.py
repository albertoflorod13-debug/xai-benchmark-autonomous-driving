"""
export_pdf.py
==============
Converts the already-built report.html into a static PDF: the constant-speed
animation tab has no equivalent on paper (no Play/Pause/slider), so the PDF
keeps only the static route view and the per-conflict XAI sections. Run
after build_report.py, which must have already produced report.html.
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_HTML = REPO_ROOT / "results" / "industrial_use_case" / "report" / "report.html"
REPORT_PDF = REPO_ROOT / "results" / "industrial_use_case" / "report" / "report.pdf"


def main() -> None:
    if not REPORT_HTML.exists():
        raise RuntimeError(f"{REPORT_HTML} not found -- run build_report.py first.")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(REPORT_HTML.as_uri())
        page.wait_for_function("window.__routePlotReady === true")
        page.emulate_media(media="print")
        page.pdf(
            path=str(REPORT_PDF), format="A4", print_background=True,
            margin={"top": "1cm", "bottom": "1cm", "left": "1cm", "right": "1cm"},
        )
        browser.close()

    print(f"PDF written to {REPORT_PDF}")


if __name__ == "__main__":
    main()