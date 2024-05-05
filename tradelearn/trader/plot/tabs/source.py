import inspect

from bokeh.models import Div
from bokeh.layouts import column

from ..tab import BacktraderPlottingTab


class SourceTab(BacktraderPlottingTab):

    def _is_useable(self):
        return not self._app.is_iplot()

    def _getSource(self):
        try:
            text = inspect.getsource(
                self._figurepage.strategy.__class__)
        except Exception:
            text = ''
        return text

    def _format_html_code(self, html_code):
        lines = html_code.split('\n')
        formatted_code = '<pre>\n'

        for line in lines:
            formatted_code += line + '\n'

        formatted_code += '</pre><br><hr>'
        return formatted_code

    def _get_tab_panel(self):
        title = Div(
            text='Source Code',
            css_classes=['tab-panel-title'])
        child = column(
            [title,
             Div(text=self._format_html_code(self._getSource()),
                 css_classes=['source-pre'],
                 sizing_mode='stretch_width')],
            sizing_mode='stretch_width')
        return child, 'Source Code'
