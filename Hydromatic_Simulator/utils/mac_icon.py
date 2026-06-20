import sys

if sys.platform == "darwin":
    from AppKit import NSApplication, NSImage
    from pathlib import Path

    _mac_app_icon = None

    def set_dock_icon(image_path: str):
        global _mac_app_icon
        app = NSApplication.sharedApplication()
        icon_path = str(Path(image_path).resolve())
        nsimage = NSImage.alloc().initWithContentsOfFile_(icon_path)
        app.setApplicationIconImage_(nsimage)
        _mac_app_icon = nsimage 

    def refresh_dock_icon(event=None):
        global _mac_app_icon
        if _mac_app_icon:
            app = NSApplication.sharedApplication()
            app.setApplicationIconImage_(_mac_app_icon)
else:
    def set_dock_icon(image_path: str):
        pass

    def refresh_dock_icon(event=None):
        pass
