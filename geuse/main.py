import os
import sys
from pathlib import Path

os.environ["PYWEBVIEW_GUI"] = "qt"
import webview
from app.api import Api
from app.database import init_db, get_user


def get_base_dir() -> Path:
    """Return the directory that contains ui/ and assets/.

    In a PyInstaller frozen build sys._MEIPASS holds all bundled files.
    In normal development mode the parent of this file (geuse/) is used.
    """
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent.resolve()


if __name__ == "__main__":
    init_db()

    existing_user = get_user()
    start_page = "daily_checkin" if existing_user else "welcome"

    api = Api()

    BASE_DIR = get_base_dir()
    window = webview.create_window(
        title="Geuse",
        url=(BASE_DIR / "ui" / "pages" / f"{start_page}.html").as_uri(),
        js_api=api,
        width=1100,
        height=720,
        background_color="#0d0d0d",
        resizable=True,
    )

    webview.start(func=window.maximize, debug=False)
