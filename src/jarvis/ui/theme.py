"""Dark and light themes (QPalette + Qt style sheet).

Widgets never hard-code colours: they set an ``objectName`` and, where state
matters, a ``tone``/``mode`` dynamic property that the style sheet selects on.
Rich-text documents (chat bubbles) read the active :class:`ThemeColors` via
:func:`current_colors` when they render.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPen, QPolygonF
from PySide6.QtWidgets import QApplication, QWidget

from jarvis.config.settings import Theme


@dataclass(frozen=True)
class ThemeColors:
    name: str
    window: str
    sidebar: str
    surface: str
    surface_alt: str
    border: str
    text: str
    text_muted: str
    accent: str
    accent_hover: str
    accent_text: str
    user_bubble: str
    user_text: str
    assistant_bubble: str
    code_bg: str
    link: str
    selection: str
    danger: str
    danger_bg: str
    warning_text: str
    warning_bg: str
    success: str
    success_bg: str
    info_bg: str


DARK = ThemeColors(
    name="dark",
    window="#0d1117",
    sidebar="#10151c",
    surface="#161b22",
    surface_alt="#1f2630",
    border="#2a313c",
    text="#e6edf3",
    text_muted="#8b949e",
    accent="#22b3d6",
    accent_hover="#3cc6e6",
    accent_text="#061016",
    user_bubble="#173445",
    user_text="#e6f4fa",
    assistant_bubble="#161b22",
    code_bg="#0a0e13",
    link="#5cc8ea",
    selection="#1f5f7a",
    danger="#f85149",
    danger_bg="#3a1519",
    warning_text="#f2cc60",
    warning_bg="#33290f",
    success="#3fb950",
    success_bg="#12301c",
    info_bg="#132a36",
)

LIGHT = ThemeColors(
    name="light",
    window="#f6f8fa",
    sidebar="#eef1f5",
    surface="#ffffff",
    surface_alt="#eef1f4",
    border="#d0d7de",
    text="#1f2328",
    text_muted="#59636e",
    accent="#0a7ea4",
    accent_hover="#0b8fba",
    accent_text="#ffffff",
    user_bubble="#ddf1f9",
    user_text="#0f2a36",
    assistant_bubble="#ffffff",
    code_bg="#f1f3f5",
    link="#0969da",
    selection="#b6e0f0",
    danger="#cf222e",
    danger_bg="#ffebe9",
    warning_text="#6b4f00",
    warning_bg="#fff5c2",
    success="#1a7f37",
    success_bg="#dafbe1",
    info_bg="#e3f3fa",
)

_current: ThemeColors = DARK


def colors_for(theme: Theme | str) -> ThemeColors:
    value = theme.value if isinstance(theme, Theme) else str(theme)
    return LIGHT if value == Theme.LIGHT.value else DARK


def current_colors() -> ThemeColors:
    return _current


def build_palette(c: ThemeColors) -> QPalette:
    pal = QPalette()
    role = QPalette.ColorRole
    group = QPalette.ColorGroup
    pal.setColor(role.Window, QColor(c.window))
    pal.setColor(role.WindowText, QColor(c.text))
    pal.setColor(role.Base, QColor(c.surface))
    pal.setColor(role.AlternateBase, QColor(c.surface_alt))
    pal.setColor(role.ToolTipBase, QColor(c.surface_alt))
    pal.setColor(role.ToolTipText, QColor(c.text))
    pal.setColor(role.PlaceholderText, QColor(c.text_muted))
    pal.setColor(role.Text, QColor(c.text))
    pal.setColor(role.Button, QColor(c.surface_alt))
    pal.setColor(role.ButtonText, QColor(c.text))
    pal.setColor(role.BrightText, QColor(c.danger))
    pal.setColor(role.Link, QColor(c.link))
    pal.setColor(role.LinkVisited, QColor(c.link))
    pal.setColor(role.Highlight, QColor(c.selection))
    pal.setColor(role.HighlightedText, QColor(c.text))
    for r in (role.WindowText, role.Text, role.ButtonText):
        pal.setColor(group.Disabled, r, QColor(c.text_muted))
    return pal


_icon_dir: Path | None = None


def _icon_directory() -> Path:
    """Per-process folder for the small generated images the style sheet references."""
    global _icon_dir
    if _icon_dir is None or not _icon_dir.exists():
        _icon_dir = Path(tempfile.mkdtemp(prefix="jarvis-theme-"))
        atexit.register(shutil.rmtree, str(_icon_dir), True)
    return _icon_dir


def _icon(kind: str, color: str, size: int = 24) -> str:
    """Render an arrow/check glyph to PNG (no image plugins or resources needed)."""
    path = _icon_directory() / f"{kind}-{color.lstrip('#')}.png"
    if not path.exists():
        image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if kind == "check":
            pen = QPen(QColor(color), size / 8)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPolyline(
                [QPointF(size * 0.2, size * 0.52), QPointF(size * 0.42, size * 0.74), QPointF(size * 0.8, size * 0.28)]
            )
        else:
            top, bottom = (size * 0.3, size * 0.7) if kind == "down" else (size * 0.7, size * 0.3)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color))
            painter.drawPolygon(
                QPolygonF([QPointF(size * 0.15, top), QPointF(size * 0.85, top), QPointF(size * 0.5, bottom)])
            )
        painter.end()
        image.save(str(path), "PNG")
    return path.as_posix()


def build_stylesheet(c: ThemeColors) -> str:
    down = _icon("down", c.text_muted)
    up = _icon("up", c.text_muted)
    check = _icon("check", c.accent_text)
    return f"""
QWidget {{
    color: {c.text};
    font-size: 10pt;
}}
QMainWindow, QDialog {{
    background: {c.window};
}}
QToolTip {{
    background: {c.surface_alt};
    color: {c.text};
    border: 1px solid {c.border};
    padding: 4px;
}}
QMenuBar {{
    background: {c.window};
    border-bottom: 1px solid {c.border};
}}
QMenuBar::item {{ padding: 4px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {c.surface_alt}; border-radius: 4px; }}
QMenu {{
    background: {c.surface};
    border: 1px solid {c.border};
    padding: 4px;
}}
QMenu::item {{ padding: 6px 24px 6px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: {c.selection}; }}
QMenu::item:disabled {{ color: {c.text_muted}; }}
QMenu::separator {{ height: 1px; background: {c.border}; margin: 4px 8px; }}
QStatusBar {{
    background: {c.sidebar};
    border-top: 1px solid {c.border};
    color: {c.text_muted};
}}
QStatusBar QLabel {{ color: {c.text_muted}; padding: 0 6px; }}
QStatusBar QLabel[tone="danger"] {{ color: {c.danger}; }}
QStatusBar QLabel[tone="success"] {{ color: {c.success}; }}
QSplitter::handle {{ background: {c.border}; }}

/* ---- inputs ---------------------------------------------------------- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {c.surface};
    border: 1px solid {c.border};
    border-radius: 6px;
    padding: 5px 7px;
    selection-background-color: {c.selection};
    selection-color: {c.text};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {c.accent};
}}
QLineEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{
    color: {c.text_muted};
}}
QAbstractSpinBox {{ padding-right: 24px; }}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 24px;
    border: none;
}}
QComboBox::down-arrow {{ image: url("{down}"); width: 10px; height: 10px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    subcontrol-origin: border;
    width: 20px;
    border: none;
    border-left: 1px solid {c.border};
    background: transparent;
}}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 6px; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; border-bottom-right-radius: 6px; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: {c.surface_alt}; }}
QAbstractSpinBox::up-arrow {{ image: url("{up}"); width: 8px; height: 8px; }}
QAbstractSpinBox::down-arrow {{ image: url("{down}"); width: 8px; height: 8px; }}
QComboBox QAbstractItemView {{
    background: {c.surface};
    border: 1px solid {c.border};
    selection-background-color: {c.selection};
}}
QCheckBox::indicator, QTreeView::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {c.border};
    border-radius: 4px;
    background: {c.surface};
}}
QCheckBox::indicator:checked, QTreeView::indicator:checked {{
    background: {c.accent};
    border-color: {c.accent};
    image: url("{check}");
}}

/* ---- buttons --------------------------------------------------------- */
QPushButton, QToolButton {{
    background: {c.surface_alt};
    border: 1px solid {c.border};
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover, QToolButton:hover {{ border-color: {c.accent}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {c.selection}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {c.text_muted}; border-color: {c.border}; }}
QPushButton:default {{ border: 1px solid {c.accent}; }}
QPushButton[variant="primary"] {{
    background: {c.accent};
    color: {c.accent_text};
    border: 1px solid {c.accent};
    font-weight: 600;
}}
QPushButton[variant="primary"]:hover {{ background: {c.accent_hover}; }}
QPushButton[variant="primary"]:disabled {{
    background: {c.surface_alt}; color: {c.text_muted}; border-color: {c.border};
}}
QPushButton[variant="danger"] {{
    background: {c.danger_bg};
    color: {c.danger};
    border: 1px solid {c.danger};
    font-weight: 600;
}}
QPushButton#SendButton {{ min-width: 76px; min-height: 30px; }}
QToolButton#DisclosureButton {{
    background: transparent; border: none; padding: 2px 4px; color: {c.text_muted};
}}
QToolButton#DisclosureButton:hover {{ color: {c.text}; }}

/* ---- sidebar --------------------------------------------------------- */
QFrame#Sidebar {{
    background: {c.sidebar};
    border-right: 1px solid {c.border};
}}
QLabel#AppTitle {{
    font-size: 15pt;
    font-weight: 700;
    color: {c.accent};
    letter-spacing: 1px;
}}
QListWidget#ConversationList {{
    background: transparent;
    border: none;
    outline: none;
}}
QListWidget#ConversationList::item {{
    padding: 8px 8px;
    border-radius: 6px;
    margin: 1px 0;
}}
QListWidget#ConversationList::item:hover {{ background: {c.surface_alt}; }}
QListWidget#ConversationList::item:selected {{
    background: {c.selection};
    color: {c.text};
}}

/* ---- chat ------------------------------------------------------------ */
QScrollArea#ChatView, QWidget#ChatContainer {{
    background: {c.window};
    border: none;
}}
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {c.border}; border-radius: 5px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {c.text_muted}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
QFrame#UserBubble {{
    background: {c.user_bubble};
    border: 1px solid {c.user_bubble};
    border-radius: 12px;
}}
QFrame#AssistantBubble {{
    background: {c.assistant_bubble};
    border: 1px solid {c.border};
    border-radius: 12px;
}}
QFrame#ErrorBubble {{
    background: {c.danger_bg};
    border: 1px solid {c.danger};
    border-radius: 10px;
}}
QFrame#NoticeBubble {{
    background: {c.info_bg};
    border: 1px solid {c.border};
    border-radius: 10px;
}}
QLabel#BubbleAuthor {{
    color: {c.text_muted};
    font-size: 8pt;
    font-weight: 600;
}}
QTextBrowser#MessageText {{
    background: transparent;
    border: none;
    padding: 0;
}}
QFrame#UserBubble QTextBrowser#MessageText {{ color: {c.user_text}; }}
QFrame#ErrorBubble QTextBrowser#MessageText {{ color: {c.danger}; }}
QFrame#ToolCard {{
    background: {c.surface};
    border: 1px solid {c.border};
    border-left: 3px solid {c.accent};
    border-radius: 8px;
}}
QFrame#ToolCard[tone="danger"] {{ border-left: 3px solid {c.danger}; }}
QFrame#ToolCard[tone="warning"] {{ border-left: 3px solid {c.warning_text}; }}
QFrame#ToolCard[tone="success"] {{ border-left: 3px solid {c.success}; }}
QLabel#ToolName {{ font-weight: 700; }}
QLabel#Badge {{
    border-radius: 8px;
    padding: 1px 8px;
    font-size: 8pt;
    font-weight: 600;
    background: {c.surface_alt};
    color: {c.text_muted};
}}
QLabel#Badge[tone="success"] {{ background: {c.success_bg}; color: {c.success}; }}
QLabel#Badge[tone="danger"] {{ background: {c.danger_bg}; color: {c.danger}; }}
QLabel#Badge[tone="warning"] {{ background: {c.warning_bg}; color: {c.warning_text}; }}
QLabel#Badge[tone="info"] {{ background: {c.info_bg}; color: {c.accent}; }}
QPlainTextEdit#ToolDetails, QPlainTextEdit#Monospace {{
    background: {c.code_bg};
    font-family: "Cascadia Mono", "Consolas", "Courier New", monospace;
    font-size: 9pt;
}}
QLabel#EmptyState {{ color: {c.text_muted}; font-size: 12pt; }}

/* ---- composer -------------------------------------------------------- */
QFrame#Composer {{
    background: {c.surface};
    border: 1px solid {c.border};
    border-radius: 10px;
}}
QPlainTextEdit#ChatInput {{
    background: transparent;
    border: none;
    padding: 4px;
}}
QPushButton#SendButton[mode="stop"] {{
    background: {c.danger_bg};
    color: {c.danger};
    border: 1px solid {c.danger};
}}

/* ---- banners & inline messages -------------------------------------- */
QFrame#Banner {{
    border-radius: 8px;
    border: 1px solid {c.border};
    background: {c.info_bg};
}}
QFrame#Banner[tone="warning"] {{ background: {c.warning_bg}; border-color: {c.warning_text}; }}
QFrame#Banner[tone="danger"] {{ background: {c.danger_bg}; border-color: {c.danger}; }}
QFrame#Banner QLabel {{ background: transparent; }}
QFrame#Banner QToolButton {{
    background: transparent;
    border: none;
    padding: 2px 6px;
    color: {c.text_muted};
}}
QFrame#Banner QToolButton:hover {{ color: {c.text}; }}
QFrame#Banner[tone="warning"] QLabel {{ color: {c.warning_text}; }}
QFrame#Banner[tone="danger"] QLabel {{ color: {c.danger}; }}
QLabel#InlineError {{ color: {c.danger}; }}
QLabel#InlineSuccess {{ color: {c.success}; }}
QLabel#Hint {{ color: {c.text_muted}; font-size: 9pt; }}
QLabel#SectionTitle {{ font-weight: 700; font-size: 11pt; }}
QLabel#RiskBadge {{
    border-radius: 8px; padding: 2px 10px; font-weight: 700;
    background: {c.warning_bg}; color: {c.warning_text};
}}
QLabel#RiskBadge[tone="danger"] {{ background: {c.danger_bg}; color: {c.danger}; }}
QLabel#RiskBadge[tone="success"] {{ background: {c.success_bg}; color: {c.success}; }}

/* ---- tabs / lists in dialogs ---------------------------------------- */
QTabWidget::pane {{
    border: 1px solid {c.border};
    border-radius: 6px;
    top: -1px;
    background: {c.window};
}}
QTabBar::tab {{
    background: transparent;
    padding: 7px 14px;
    border: 1px solid transparent;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: {c.text_muted};
}}
QTabBar::tab:selected {{
    background: {c.window};
    border-color: {c.border};
    border-bottom-color: {c.window};
    color: {c.text};
}}
QTreeWidget, QListWidget {{
    background: {c.surface};
    border: 1px solid {c.border};
    border-radius: 6px;
    alternate-background-color: {c.surface_alt};
}}
QHeaderView::section {{
    background: {c.surface_alt};
    border: none;
    border-bottom: 1px solid {c.border};
    padding: 5px;
    font-weight: 600;
}}
QProgressBar {{
    background: {c.surface_alt};
    border: none;
    border-radius: 3px;
    max-height: 6px;
}}
QProgressBar::chunk {{ background: {c.accent}; border-radius: 3px; }}
"""


def apply_theme(app: QApplication, theme: Theme | str) -> ThemeColors:
    """Apply ``theme`` to the whole application and return its colours."""
    global _current
    colors = colors_for(theme)
    _current = colors
    app.setPalette(build_palette(colors))
    app.setStyleSheet(build_stylesheet(colors))
    return colors


def repolish(widget: QWidget) -> None:
    """Re-evaluate the style sheet after a dynamic property changed."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_tone(widget: QWidget, tone: str, prop: str = "tone") -> None:
    if widget.property(prop) != tone:
        widget.setProperty(prop, tone)
        repolish(widget)
