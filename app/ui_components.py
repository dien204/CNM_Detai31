from html import escape
from typing import Iterable

APP_TITLE = "Identity trust, risk and behavior analytics"


def topbar_html(username: str, role_label: str, runtime_label: str, database_label: str = "SQLite") -> str:
    subtitle = f"{escape(username)} • {escape(role_label)} • {escape(runtime_label)} • {escape(database_label)}"
    return f"""
    <div class="site-header-brand">
        <div class="brand-mark">🛡️</div>
        <div>
            <div class="brand-title-small">User Trust Platform</div>
            <div class="brand-sub">{subtitle}</div>
        </div>
    </div>
    """


def nav_link_html(page: str, active_page: str, variant: str = "secondary") -> str:
    active = " active" if page == active_page else ""
    safe_page = escape(page)
    return f'<span class="nav-link nav-{variant}{active}">{safe_page}</span>'


def nav_group_html(pages: Iterable[str], active_page: str, variant: str = "secondary") -> str:
    links = "".join(nav_link_html(page, active_page, variant) for page in pages)
    return f'<div class="nav-row nav-row-{variant}">{links}</div>'
