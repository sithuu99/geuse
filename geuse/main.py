import webview
from app.api import Api
from app.database import init_db

if __name__ == "__main__":
    init_db()

    api = Api()

    window = webview.create_window(
        title="Geuse",
        url="ui/pages/welcome.html",
        js_api=api,
        width=1100,
        height=720,
        background_color="#0d0d0d",
        resizable=True,
    )

    webview.start(debug=False)
