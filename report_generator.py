import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FuncFormatter
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
    BaseDocTemplate, Frame, PageTemplate
)
from reportlab.platypus.flowables import HRFlowable, KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from datetime import datetime
import numpy as np
import pandas as pd
import io
import warnings

warnings.filterwarnings('ignore')


class StockAnalysisReport:
    def __init__(self, symbol):
        self.symbol = symbol.upper()
        self.stock = yf.Ticker(self.symbol)
        self.company_info = {}
        self.financial_data = {}
        self.competitor_data = {}
        self.factor_data = {}
        self.filename = f"{self.symbol}_Investment_Analysis_Report.pdf"
        self._current_doc = None

        # Aesthetics
        plt.style.use('default')
        sns.set_palette("husl")

        # Theme
        self.THEME = {
            'bg_light': colors.HexColor('#F6F7F9'),
            'bg_row': colors.HexColor('#FBFCFD'),
            'text': colors.HexColor('#2C3E50'),
            'muted': colors.HexColor('#7F8C8D'),
            'accent': colors.HexColor('#16A085'),
            'accent_dark': colors.HexColor('#117A65'),
            'header': colors.HexColor('#2C3E50'),
            'line': colors.HexColor('#DCE3EA'),
            'warn': colors.HexColor('#E67E22'),
            'danger': colors.HexColor('#C0392B'),
            'ok': colors.HexColor('#27AE60'),
        }

    # ================== Styles and layout helpers ==================
    def _build_styles(self):
        styles = getSampleStyleSheet()

        base_font = 'Helvetica'
        base_bold = 'Helvetica-Bold'

        styles.add(ParagraphStyle(
            'TitleXL',
            parent=styles['Title'],
            fontName=base_bold,
            fontSize=26,
            leading=30,
            textColor=self.THEME['header'],
            alignment=TA_CENTER,
            spaceAfter=16
        ))
        styles.add(ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontName=base_font,
            fontSize=12,
            leading=16,
            textColor=self.THEME['muted'],
            alignment=TA_CENTER,
            spaceAfter=24
        ))
        styles.add(ParagraphStyle(
            'H1',
            parent=styles['Heading1'],
            fontName=base_bold,
            fontSize=18,
            leading=22,
            textColor=self.THEME['header'],
            spaceBefore=12,
            spaceAfter=10
        ))
        styles.add(ParagraphStyle(
            'H2',
            parent=styles['Heading2'],
            fontName=base_bold,
            fontSize=14,
            leading=18,
            textColor=self.THEME['accent'],
            spaceBefore=10,
            spaceAfter=8
        ))
        styles.add(ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName=base_font,
            fontSize=11,
            leading=16,
            textColor=self.THEME['text'],
            spaceAfter=10,
            alignment=TA_JUSTIFY
        ))
        styles.add(ParagraphStyle(
            'Small',
            parent=styles['Normal'],
            fontName=base_font,
            fontSize=9,
            leading=12,
            textColor=self.THEME['muted'],
            spaceAfter=6
        ))
        styles.add(ParagraphStyle(
            'KPI',
            parent=styles['Normal'],
            fontName=base_bold,
            fontSize=12,
            leading=14,
            textColor=self.THEME['header'],
            alignment=TA_CENTER
        ))
        # Cell style for wrapped table cells
        styles.add(ParagraphStyle(
            'Cell',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8.5,
            leading=11,
            textColor=self.THEME['text'],
            alignment=TA_JUSTIFY,
            spaceAfter=0
        ))
        return styles

    def _wrap_cells(self, rows, header_style='KPI', cell_style='Cell'):
        """Convert each cell to a Paragraph to enable word wrapping."""
        wrapped = []
        for r_i, row in enumerate(rows):
            wrow = []
            for val in row:
                txt = self._sanitize_label(val)
                style = self.styles[header_style] if r_i == 0 else self.styles[cell_style]
                wrow.append(Paragraph(txt, style))
            wrapped.append(wrow)
        return wrapped

    def _draw_header_footer(self, canvas, doc):
        canvas.saveState()
        w, h = A4

        # Header line
        canvas.setStrokeColor(self.THEME['line'])
        canvas.setLineWidth(0.6)
        canvas.line(40, h - 60, w - 40, h - 60)

        # Header text
        canvas.setFont('Helvetica-Bold', 10)
        title = f"{self.company_info.get('longName', self.symbol)} Investment Analysis"
        canvas.setFillColor(self.THEME['header'])
        canvas.drawString(40, h - 50, title)

        # Footer line
        canvas.setStrokeColor(self.THEME['line'])
        canvas.setLineWidth(0.6)
        canvas.line(40, 50, w - 40, 50)

        # Footer text
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(self.THEME['muted'])
        canvas.drawString(40, 36, f"Created on {datetime.now().strftime('%B %d, %Y')}")
        canvas.drawRightString(w - 40, 36, f"Page {doc.page}")

        canvas.restoreState()

    def _hr(self, thickness=0.8, color=None, spaceBefore=8, spaceAfter=12):
        return HRFlowable(
            width='100%',
            thickness=thickness,
            color=color or self.THEME['line'],
            spaceBefore=spaceBefore,
            spaceAfter=spaceAfter
        )

    def _striped_table_style(self, header_bg=None):
        header_bg = header_bg or self.THEME['accent']
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), header_bg),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LINEBELOW', (0, 0), (-1, 0), 0.8, self.THEME['accent_dark']),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [self.THEME['bg_row'], colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.3, self.THEME['line']),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ])

    # -------- Formatting helpers --------
    def _fmt_num(self, x, decimals=2):
        try:
            x = float(x)
        except Exception:
            return str(x)
        absx = abs(x)
        if absx >= 1e12:
            return f"{x/1e12:.{decimals}f}T"
        if absx >= 1e9:
            return f"{x/1e9:.{decimals}f}B"
        if absx >= 1e6:
            return f"{x/1e6:.{decimals}f}M"
        if absx >= 1e3:
            return f"{x:,.{decimals}f}"
        return f"{x:.{decimals}f}"

    def _fmt_pct(self, x, decimals=2):
        try:
            return f"{float(x):.{decimals}f}%"
        except Exception:
            return str(x)

    def _sanitize_label(self, s):
        if s is None:
            return "N/A"
        s = str(s)
        return ''.join(ch for ch in s if ch.isprintable())

    def _clean_df(self, df, max_rows=18, max_cols=6):
        if df is None or df.empty:
            return None
        d = df.copy()
        # Drop duplicates / unnamed
        d = d[~d.index.duplicated(keep='first')]
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [' | '.join([self._sanitize_label(c) for c in col]).strip() for col in d.columns]
        else:
            d.columns = [self._sanitize_label(c) for c in d.columns]
        if isinstance(d.index, pd.MultiIndex):
            d.index = [' | '.join([self._sanitize_label(i) for i in idx]).strip() for idx in d.index]
        else:
            d.index = [self._sanitize_label(i) for i in d.index]

        def _fmt_col(c):
            try:
                ts = pd.to_datetime(c)
                if pd.notnull(ts):
                    return ts.strftime("%Y-%m")
            except Exception:
                pass
            return self._sanitize_label(c)

        d.columns = [_fmt_col(c) for c in d.columns]
        d = d.iloc[:max_rows, :max_cols]

        for c in d.columns:
            try:
                d[c] = pd.to_numeric(d[c], errors='coerce')
                d[c] = d[c].apply(lambda v: '—' if pd.isna(v) else self._fmt_num(v))
            except Exception:
                d[c] = d[c].astype(str)
        return d

    # ================== Data acquisition and calculations ==================
    def fetch_data(self):
        """Fetch all necessary data for the report"""
        try:
            # Company information and fast info
            self.company_info = self.stock.info if hasattr(self.stock, "info") else {}
            self.fast_info = getattr(self.stock, "fast_info", {}) or {}

            # Historical data
            hist_5y = self.stock.history(period="5y", auto_adjust=False)
            hist_10y = self.stock.history(period="10y", auto_adjust=False)

            # Financial statements (normalized)
            def _norm(df):
                if df is None or df.empty:
                    return df
                df = df[~df.index.duplicated(keep='first')]
                df = df.loc[~df.index.astype(str).str.contains('^Unnamed', case=False, na=False)]
                return df

            income_t = _norm(self.stock.income_stmt)
            income_q = _norm(self.stock.quarterly_income_stmt)
            bs_t = _norm(self.stock.balance_sheet)
            bs_q = _norm(self.stock.quarterly_balance_sheet)
            cf_t = _norm(self.stock.cash_flow)
            cf_q = _norm(self.stock.quarterly_cash_flow)

            # Dividends and splits
            dividends = self.stock.dividends
            splits = self.stock.splits

            # Analyst info
            rec_trend = self.stock.recommendations if hasattr(self.stock, "recommendations") else None
            cal = self.stock.calendar if hasattr(self.stock, "calendar") else None
            earnings = self.stock.earnings if hasattr(self.stock, "earnings") else None
            q_earnings = self.stock.quarterly_earnings if hasattr(self.stock, "quarterly_earnings") else None
            sustainability = getattr(self.stock, "sustainability", None)

            # Key metrics derived
            self.financial_data = {
                'hist_5y': hist_5y,
                'hist_10y': hist_10y,
                'income_t': income_t,
                'income_q': income_q,
                'bs_t': bs_t,
                'bs_q': bs_q,
                'cf_t': cf_t,
                'cf_q': cf_q,
                'dividends': dividends,
                'splits': splits,
                'recommendations': rec_trend,
                'calendar': cal,
                'earnings': earnings,
                'quarterly_earnings': q_earnings,
                'sustainability': sustainability,
                'key_metrics': self._calculate_key_metrics(hist_5y, income_t, bs_t, cf_t, dividends),
            }

            # Benchmark and factor prep
            self._fetch_competitor_data()
            self._prepare_factor_and_benchmark()

            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def _calculate_key_metrics(self, hist_data, income_stmt, balance_sheet, cash_flow, dividends):
        metrics = {}
        try:
            if hist_data is None or hist_data.empty:
                return {
                    'current_price': 0, 'year_high': 0, 'year_low': 0,
                    'annual_return': 0, 'revenue_growth': 0,
                    'volatility': 0, 'sharpe': 0, 'sortino': 0,
                    'max_drawdown': 0, 'dividend_yield': 0
                }

            current_price = float(hist_data['Close'].iloc[-1])
            year_slice = hist_data.iloc[-252:] if len(hist_data) >= 252 else hist_data
            year_high = float(year_slice['High'].max())
            year_low = float(year_slice['Low'].min())

            daily_ret = hist_data['Close'].pct_change().dropna()
            annual_vol = float(daily_ret.std() * np.sqrt(252) * 100)
            if len(hist_data) >= 252:
                price_1y_ago = float(hist_data['Close'].iloc[-252])
                annual_return = ((current_price / price_1y_ago) - 1) * 100 if price_1y_ago else 0.0
            else:
                annual_return = float(daily_ret.mean() * 252 * 100)

            sharpe = float((daily_ret.mean() / (daily_ret.std() or 1e-9)) * np.sqrt(252))
            downside = daily_ret[daily_ret < 0]
            sortino = float((daily_ret.mean() / (downside.std() or 1e-9)) * np.sqrt(252)) if len(downside) > 0 else 0.0

            roll_max = hist_data['Close'].cummax()
            dd = hist_data['Close'] / roll_max - 1.0
            max_drawdown = float(dd.min() * 100)

            dividend_yield = 0.0
            if dividends is not None and not dividends.empty:
                last_12m = dividends[dividends.index >= (dividends.index.max() - pd.DateOffset(years=1))].sum()
                dividend_yield = float((last_12m / current_price) * 100) if current_price else 0.0

            revenue_growth = 0.0
            if hasattr(income_stmt, 'empty') and not income_stmt.empty and len(income_stmt.columns) >= 2:
                if 'Total Revenue' in income_stmt.index:
                    recent_revenue = float(income_stmt.loc['Total Revenue'].iloc[0])
                    prev_revenue = float(income_stmt.loc['Total Revenue'].iloc[1])
                    revenue_growth = ((recent_revenue / prev_revenue) - 1) * 100 if prev_revenue else 0.0

            metrics = {
                'current_price': current_price,
                'year_high': year_high,
                'year_low': year_low,
                'annual_return': annual_return,
                'revenue_growth': revenue_growth,
                'volatility': annual_vol,
                'sharpe': sharpe,
                'sortino': sortino,
                'max_drawdown': max_drawdown,
                'dividend_yield': dividend_yield
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics = {
                'current_price': 0, 'year_high': 0, 'year_low': 0,
                'annual_return': 0, 'revenue_growth': 0,
                'volatility': 0, 'sharpe': 0, 'sortino': 0,
                'max_drawdown': 0, 'dividend_yield': 0
            }
        return metrics

    def _fetch_competitor_data(self):
        industry_competitors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META'],
            'Finance': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'BP'],
            'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'MCD'],
            'Industrial': ['CAT', 'BA', 'GE', 'HON', 'UNP', 'UPS'],
            'Communication': ['VZ', 'T', 'CMCSA', 'TMUS', 'DIS', 'NFLX']
        }
        default_competitors = ['SPY', 'QQQ', 'DIA']

        try:
            industry = self.company_info.get('industry', '') or ''
            sector = self.company_info.get('sector', '') or ''

            competitors = []
            for key, stocks in industry_competitors.items():
                if key.lower() in industry.lower() or key.lower() in sector.lower():
                    competitors = [s for s in stocks if s != self.symbol][:5]
                    break

            if not competitors:
                competitors = default_competitors

            competitor_metrics = {}
            for comp in competitors:
                try:
                    comp_stock = yf.Ticker(comp)
                    info = comp_stock.info if hasattr(comp_stock, "info") else {}
                    comp_hist = comp_stock.history(period="5y")
                    comp_1y = comp_stock.history(period="1y")
                    ann_ret = 0.0
                    if not comp_1y.empty:
                        ann_ret = ((float(comp_1y['Close'].iloc[-1]) / float(comp_1y['Close'].iloc[0])) - 1) * 100
                    vol = comp_hist['Close'].pct_change().std() * np.sqrt(252) * 100 if not comp_hist.empty else 0.0
                    competitor_metrics[comp] = {
                        'annual_return': float(ann_ret),
                        'volatility': float(vol),
                        'current_price': float(comp_1y['Close'].iloc[-1]) if not comp_1y.empty else 0.0,
                        'market_cap': info.get('marketCap', None),
                        'pe': info.get('trailingPE', None),
                        'ps': info.get('priceToSalesTrailing12Months', None),
                        'pb': info.get('priceToBook', None)
                    }
                except Exception:
                    continue

            self.competitor_data = competitor_metrics

        except Exception as e:
            print(f"Error fetching competitor data: {e}")
            self.competitor_data = {}

    def _prepare_factor_and_benchmark(self):
        try:
            hist = self.financial_data.get('hist_5y')
            if hist is None or hist.empty:
                self.factor_data = {}
                return

            spy = yf.Ticker('SPY')
            spy_hist = spy.history(period='5y')
            if spy_hist is None or spy_hist.empty:
                self.factor_data = {}
                return

            r = hist['Close'].pct_change().dropna()
            r_spy = spy_hist['Close'].pct_change().dropna()
            df = pd.concat([r, r_spy], axis=1, join='inner')
            df.columns = ['r', 'mkt']
            if len(df) > 10:
                cov = np.cov(df['r'], df['mkt'])[0, 1]
                var_mkt = np.var(df['mkt'])
                beta = cov / (var_mkt or 1e-9)
                alpha = (df['r'].mean() - beta * df['mkt'].mean()) * 252
            else:
                beta = 0.0
                alpha = 0.0

            rolling_vol = r.rolling(252).std() * np.sqrt(252) * 100
            rolling_corr = df['r'].rolling(252).corr(df['mkt'])

            monthly = r.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_df = monthly.to_frame('ret')
            monthly_df['Year'] = monthly_df.index.year
            monthly_df['Month'] = monthly_df.index.month
            pivot = monthly_df.pivot_table(index='Year', columns='Month', values='ret')

            var_95 = float(np.percentile(r, 5) * 100)

            self.factor_data = {
                'beta': float(beta),
                'alpha_ann': float(alpha * 100),
                'rolling_vol': rolling_vol,
                'rolling_corr': rolling_corr,
                'monthly_pivot': pivot,
                'spy_hist': spy_hist,
                'var_95': var_95
            }
        except Exception as e:
            print(f"Error preparing factor/benchmark data: {e}")
            self.factor_data = {}

    # ================== Charting helpers ==================
    def _save_current_fig(self, charts, key, dpi=300):
        # Nudge margins to avoid clipping rotated labels and long titles
        plt.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.22)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        charts[key] = buf
        plt.close()

    def _apply_time_axis_clean(self, axes):
        if isinstance(axes, (list, np.ndarray)):
            bottom = axes[-1]
            bottom.xaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
            for a in axes:
                a.tick_params(axis='x', labelrotation=45)
        else:
            ax = axes
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)

    def _chart_price_with_mas(self, charts):
        data = self.financial_data['hist_5y']
        if data is None or data.empty:
            return
        s = data['Close']
        ma50 = s.rolling(50).mean()
        ma200 = s.rolling(200).mean()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(s.index, s, color='#2E86C1', linewidth=1.8, label='Close')
        ax.plot(ma50.index, ma50, color='#16A085', linewidth=1.3, label='MA50')
        ax.plot(ma200.index, ma200, color='#7F8C8D', linewidth=1.3, label='MA200')
        ax.fill_between(s.index, s, ma200, where=(s < ma200), color='#C0392B', alpha=0.08)
        ax.set_title(f'{self.symbol} Price with MA50/MA200', fontsize=14, color='#2C3E50', pad=12, fontweight='bold')
        ax.set_ylabel('Price ($)'); ax.set_xlabel('')
        ax.grid(True, axis='y', alpha=0.8, color='#E8EEF4')
        ax.legend(frameon=False)
        self._apply_time_axis_clean(ax)
        self._save_current_fig(charts, 'price_mas')

    def _chart_rsi_macd(self, charts):
        data = self.financial_data['hist_5y']
        if data is None or data.empty:
            return
        close = data['Close']
        delta = close.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = (-delta.clip(upper=0)).rolling(14).mean()
        rs = up / (down.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                                 gridspec_kw={'height_ratios': [2, 1, 1]})
        axes[0].plot(close.index, close, color='#2E86C1', linewidth=1.6)
        axes[0].set_title(f'{self.symbol} Close, RSI and MACD', fontsize=14, color='#2C3E50', pad=10, fontweight='bold')
        axes[0].grid(True, axis='y', alpha=0.8, color='#E8EEF4')

        axes[1].plot(rsi.index, rsi, color='#16A085', linewidth=1.2)
        axes[1].axhline(70, color='#C0392B', linestyle='--', linewidth=0.8)
        axes[1].axhline(30, color='#27AE60', linestyle='--', linewidth=0.8)
        axes[1].set_ylabel('RSI')
        axes[1].grid(True, axis='y', alpha=0.5, color='#E8EEF4')

        axes[2].plot(macd.index, macd, color='#34495E', linewidth=1.2, label='MACD')
        axes[2].plot(signal.index, signal, color='#8E44AD', linewidth=1.2, label='Signal')
        axes[2].bar(hist.index, hist, color=np.where(hist >= 0, '#27AE60', '#C0392B'), alpha=0.5)
        axes[2].legend(frameon=False)
        axes[2].grid(True, axis='y', alpha=0.5, color='#E8EEF4')

        self._apply_time_axis_clean(axes)
        self._save_current_fig(charts, 'rsi_macd')

    def _chart_returns_distribution(self, charts):
        data = self.financial_data['hist_5y']
        if data is None or data.empty:
            return
        returns = data['Close'].pct_change().dropna() * 100
        plt.figure(figsize=(10, 5))
        plt.hist(returns, bins=60, alpha=0.9, color='#8E44AD', edgecolor='#FFFFFF', linewidth=0.5)
        plt.axvline(returns.mean(), color='#C0392B', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
        pct99 = np.percentile(returns, 99)
        pct01 = np.percentile(returns, 1)
        plt.xlim(pct01, pct99)
        plt.title(f'{self.symbol} Daily Returns Distribution', fontsize=13, color='#2C3E50', pad=12, fontweight='bold')
        plt.xlabel('Daily Returns (%)'); plt.ylabel('Frequency')
        plt.legend(frameon=False)
        plt.grid(True, axis='y', alpha=0.8, color='#E8EEF4')
        self._save_current_fig(charts, 'returns_dist')

    def _chart_rolling_vol_corr(self, charts):
        rv = self.factor_data.get('rolling_vol')
        rc = self.factor_data.get('rolling_corr')
        if rv is None or rc is None or rv.dropna().empty or rc.dropna().empty:
            return
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(rv.index, rv, color='#C0392B', linewidth=1.5)
        axes[0].set_title(f'Rolling 1Y Volatility (%)', fontsize=13, color='#2C3E50', pad=10, fontweight='bold')
        axes[0].grid(True, axis='y', alpha=0.8, color='#E8EEF4')
        axes[1].plot(rc.index, rc, color='#16A085', linewidth=1.5)
        axes[1].set_title('Rolling 1Y Correlation with SPY', fontsize=13, color='#2C3E50', pad=10, fontweight='bold')
        axes[1].grid(True, axis='y', alpha=0.8, color='#E8EEF4')
        self._apply_time_axis_clean(axes)
        self._save_current_fig(charts, 'rolling_vol_corr')

    def _chart_seasonality_heatmap(self, charts):
        pivot = self.factor_data.get('monthly_pivot')
        if pivot is None or pivot.empty:
            return
        n_years = len(pivot.index)
        annot = n_years <= 6
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot * 100,
            cmap='RdYlGn',
            annot=annot,
            fmt='.1f' if annot else '',
            cbar_kws={'label': 'Return %'},
            linewidths=0.4,
            linecolor='#E8EEF4',
            annot_kws={'size': 8}
        )
        plt.title(f'{self.symbol} Monthly Returns Heatmap', fontsize=13, color='#2C3E50', pad=12, fontweight='bold')
        plt.xlabel('Month'); plt.ylabel('Year')
        self._save_current_fig(charts, 'seasonality_heatmap')

    def _chart_drawdown(self, charts):
        data = self.financial_data['hist_5y']
        if data is None or data.empty:
            return
        close = data['Close']
        roll_max = close.cummax()
        dd = close / roll_max - 1.0
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.fill_between(dd.index, dd * 100, 0, color='#C0392B', alpha=0.4)
        ax.set_title(f'{self.symbol} Drawdown (%)', fontsize=13, color='#2C3E50', pad=12, fontweight='bold')
        ax.set_ylabel('Drawdown %'); ax.set_xlabel('')
        ax.grid(True, axis='y', alpha=0.8, color='#E8EEF4')
        self._apply_time_axis_clean(ax)
        self._save_current_fig(charts, 'drawdown')

    def _chart_competitor_bars(self, charts):
        if not self.competitor_data:
            return
        comps = list(self.competitor_data.keys())
        rets = [self.competitor_data[c]['annual_return'] for c in comps]
        vol = [self.competitor_data[c]['volatility'] for c in comps]
        x = np.arange(len(comps))
        width = 0.38
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.bar(x - width/2, rets, width, label='Annual Return (%)', color='#3498DB')
        ax.bar(x + width/2, vol, width, label='Volatility (%)', color='#95A5A6')
        ax.set_xticks(x)
        ax.set_xticklabels(comps, rotation=45, ha='right')
        ax.set_title('Peer Comparison: Return vs Volatility', fontsize=13, color='#2C3E50', pad=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.8, color='#E8EEF4')
        ax.legend(frameon=False)
        self._save_current_fig(charts, 'comp_peers_bars')

    def _chart_valuation_peers(self, charts):
        if not self.competitor_data:
            return
        comps = list(self.competitor_data.keys())
        pe = [self.competitor_data[c]['pe'] if self.competitor_data[c]['pe'] is not None else np.nan for c in comps]
        ps = [self.competitor_data[c]['ps'] if self.competitor_data[c]['ps'] is not None else np.nan for c in comps]
        pb = [self.competitor_data[c]['pb'] if self.competitor_data[c]['pb'] is not None else np.nan for c in comps]
        labels = ['P/E', 'P/S', 'P/B']
        metrics = [pe, ps, pb]
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, m in enumerate(metrics):
            vals = pd.Series(m).astype(float)
            ax.plot(comps, vals, marker='o', linewidth=1.4, label=labels[i])
        ax.set_title('Peer Valuation Multiples', fontsize=13, color='#2C3E50', pad=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.8, color='#E8EEF4')
        ax.legend(frameon=False)
        ax.tick_params(axis='x', labelrotation=45)
        self._save_current_fig(charts, 'peer_valuation')

    def _chart_dividend_history(self, charts):
        div = self.financial_data.get('dividends')
        if div is None or div.empty:
            return
        yearly = div.resample('Y').sum()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(yearly.index.year, yearly.values, color='#27AE60', alpha=0.85)
        ax.set_title('Dividend History (Annual Sum)', fontsize=13, color='#2C3E50', pad=12, fontweight='bold')
        ax.set_ylabel('Dividend $'); ax.set_xlabel('Year')
        ax.grid(True, axis='y', alpha=0.6, color='#E8EEF4')
        self._save_current_fig(charts, 'dividend_history')

    def _chart_monte_carlo(self, charts, n_sims=250, horizon_days=252):
        data = self.financial_data['hist_5y']
        if data is None or data.empty:
            return
        close = data['Close'].dropna()
        if len(close) < 50:
            return
        returns = close.pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        last_price = float(close.iloc[-1])

        np.random.seed(42)
        sims = np.zeros((horizon_days, n_sims))
        for i in range(n_sims):
            rand = np.random.normal(mu, sigma, horizon_days)
            path = last_price * np.cumprod(1 + rand)
            sims[:, i] = path

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sims, color='#BDC3C7', alpha=0.25)
        ax.plot(np.mean(sims, axis=1), color='#16A085', linewidth=2, label='Mean Path')
        ax.set_title(f'{self.symbol} Monte Carlo Price Simulation (1Y, {n_sims} runs)', fontsize=13, color='#2C3E50', pad=12, fontweight='bold')
        ax.set_ylabel('Price ($)'); ax.set_xlabel('Days')
        ax.grid(True, axis='y', alpha=0.8, color='#E8EEF4')
        ax.legend(frameon=False)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x):,}'))
        self._save_current_fig(charts, 'monte_carlo')

    def _chart_revenue_eps(self, charts):
        inc = self.financial_data.get('income_t')
        qearn = self.financial_data.get('quarterly_earnings')
        if inc is not None and not inc.empty and 'Total Revenue' in inc.index:
            rev = inc.loc['Total Revenue'].T.sort_index()
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(rev.index, (pd.to_numeric(rev.values.flatten(), errors='coerce') / 1e9), color='#2E86C1', linewidth=2)
            ax.set_title('Total Revenue (Trailing, billions)', fontsize=13, color='#2C3E50', pad=12, fontweight='bold')
            ax.set_ylabel('Revenue (B$)'); ax.set_xlabel('')
            ax.grid(True, axis='y', alpha=0.7, color='#E8EEF4')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
            ax.tick_params(axis='x', labelrotation=45)
            self._save_current_fig(charts, 'revenue_trend')
        if qearn is not None and not qearn.empty and 'Earnings' in qearn.columns:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.bar(qearn.index, qearn['Earnings'] / 1e9, color='#8E44AD', alpha=0.8)
            ax.set_title('Quarterly Earnings (billions)', fontsize=13, color='#2C3E50', pad=12, fontweight='bold')
            ax.set_ylabel('Earnings (B$)'); ax.set_xlabel('')
            ax.grid(True, axis='y', alpha=0.7, color='#E8EEF4')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
            ax.tick_params(axis='x', labelrotation=45)
            self._save_current_fig(charts, 'earnings_quarterly')

    # ================== Build all charts ==================
    def create_charts(self):
        charts = {}
        plt.rcParams.update({
            'axes.edgecolor': '#DCE3EA',
            'axes.labelcolor': '#2C3E50',
            'xtick.color': '#7F8C8D',
            'ytick.color': '#7F8C8D',
            'grid.color': '#E8EEF4',
            'grid.linestyle': '-',
            'grid.linewidth': 0.7,
            'figure.autolayout': False
        })
        self._chart_price_with_mas(charts)
        self._chart_rsi_macd(charts)
        self._chart_returns_distribution(charts)
        self._chart_rolling_vol_corr(charts)
        self._chart_drawdown(charts)
        self._chart_competitor_bars(charts)
        self._chart_valuation_peers(charts)
        self._chart_dividend_history(charts)
        self._chart_seasonality_heatmap(charts)
        self._chart_monte_carlo(charts, n_sims=250)
        self._chart_revenue_eps(charts)
        return charts

    # ================== Tables and sections ==================
    def _kpi_table(self, styles):
        km = self.financial_data['key_metrics']
        items = [
            ['Metric', 'Value'],
            ['Current Price', f"${self._fmt_num(km['current_price'])}"],
            ['52-Week High', f"${self._fmt_num(km['year_high'])}"],
            ['52-Week Low', f"${self._fmt_num(km['year_low'])}"],
            ['Annual Return', self._fmt_pct(km['annual_return'])],
            ['Volatility (ann.)', self._fmt_pct(km['volatility'])],
            ['Sharpe (rf=0)', self._fmt_num(km['sharpe'])],
            ['Sortino (rf=0)', self._fmt_num(km['sortino'])],
            ['Max Drawdown', self._fmt_pct(km['max_drawdown'])],
            ['Dividend Yield', self._fmt_pct(km['dividend_yield'])],
            ['Revenue Growth (YoY)', self._fmt_pct(km['revenue_growth'])],
            ['Beta vs SPY', self._fmt_num(self.factor_data.get('beta', 0))],
            ['VaR 95% (daily)', self._fmt_pct(self.factor_data.get('var_95', 0))]
        ]
        data = self._wrap_cells(items, header_style='KPI', cell_style='Cell')
        table = Table(data, colWidths=[2.8*inch, 3.2*inch], hAlign='LEFT', repeatRows=1)
        table.setStyle(self._striped_table_style())
        table.setStyle(TableStyle([('WORDWRAP', (0,0), (-1,-1), 'CJK')]))
        return table

    def _company_overview_table(self):
        i = self.company_info
        rows = [
            ['Attribute', 'Value'],
            ['Company', i.get('longName', self.symbol)],
            ['Ticker', self.symbol],
            ['Sector', i.get('sector', 'N/A')],
            ['Industry', i.get('industry', 'N/A')],
            ['Country', i.get('country', 'N/A')],
            ['Employees', self._fmt_num(i.get('fullTimeEmployees', 'N/A'), 0)],
            ['Market Cap', f"${self._fmt_num(i.get('marketCap', 0), 2)}"],
            ['Enterprise Value', f"${self._fmt_num(i.get('enterpriseValue', 0), 2)}"],
            ['P/E (TTM)', self._fmt_num(i.get('trailingPE')) if i.get('trailingPE') else 'N/A'],
            ['P/S (TTM)', self._fmt_num(i.get('priceToSalesTrailing12Months')) if i.get('priceToSalesTrailing12Months') else 'N/A'],
            ['P/B', self._fmt_num(i.get('priceToBook')) if i.get('priceToBook') else 'N/A'],
            ['EV/EBITDA', self._fmt_num(i.get('enterpriseToEbitda')) if i.get('enterpriseToEbitda') else 'N/A'],
            ['Profit Margin', self._fmt_pct(i.get('profitMargins', 0)*100) if i.get('profitMargins') is not None else 'N/A'],
            ['Operating Margin', self._fmt_pct(i.get('operatingMargins', 0)*100) if i.get('operatingMargins') is not None else 'N/A'],
            ['ROE', self._fmt_pct(i.get('returnOnEquity', 0)*100) if i.get('returnOnEquity') is not None else 'N/A'],
            ['ROA', self._fmt_pct(i.get('returnOnAssets', 0)*100) if i.get('returnOnAssets') is not None else 'N/A'],
            ['Debt/Equity', self._fmt_num(i.get('debtToEquity')) if i.get('debtToEquity') else 'N/A'],
            ['Current Ratio', self._fmt_num(i.get('currentRatio')) if i.get('currentRatio') else 'N/A'],
        ]
        data = self._wrap_cells(rows, header_style='KPI', cell_style='Cell')
        t = Table(data, colWidths=[2.8*inch, 3.2*inch], hAlign='LEFT', repeatRows=1)
        t.setStyle(self._striped_table_style(header_bg=self.THEME['header']))
        t.setStyle(TableStyle([('WORDWRAP', (0,0), (-1,-1), 'CJK')]))
        return t

    def _fin_stat_tables(self):
        flows = []

        def _table_from_df(df, title, cols=6):
            d = self._clean_df(df, max_rows=18, max_cols=cols)
            flows.append(Paragraph(title, self.styles['H2']))
            if d is None:
                flows.append(Paragraph("Not available.", self.styles['Small']))
                flows.append(Spacer(1, 6))
                return

            header = ['Metric'] + list(d.columns)
            rows = [header] + [[idx] + list(d.loc[idx].values) for idx in d.index]
            data = self._wrap_cells(rows, header_style='KPI', cell_style='Cell')

            try:
                docw = self._current_doc.width
            except Exception:
                docw = 450
            first = 2.2 * inch
            rest_cols = min(len(d.columns), cols)
            per = (docw - first) / max(rest_cols, 1)

            t = Table(
                data,
                colWidths=[first] + [per]*rest_cols,
                hAlign='LEFT',
                repeatRows=1,
                splitByRow=1
            )
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.THEME['accent']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [self.THEME['bg_row'], colors.white]),
                ('GRID', (0, 0), (-1, -1), 0.3, self.THEME['line']),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('WORDWRAP', (0,0), (-1,-1), 'CJK'),
            ]))
            flows.append(t)
            flows.append(Spacer(1, 8))

        _table_from_df(self.financial_data.get('income_t'), "Income statement (annual, summary)", cols=6)
        _table_from_df(self.financial_data.get('income_q'), "Income statement (quarterly, summary)", cols=6)
        _table_from_df(self.financial_data.get('bs_t'), "Balance sheet (annual, summary)", cols=6)
        _table_from_df(self.financial_data.get('bs_q'), "Balance sheet (quarterly, summary)", cols=6)
        _table_from_df(self.financial_data.get('cf_t'), "Cash flow (annual, summary)", cols=6)
        _table_from_df(self.financial_data.get('cf_q'), "Cash flow (quarterly, summary)", cols=6)
        return flows

    def _peer_table(self):
        if not self.competitor_data:
            return Paragraph("Peer data not available.", self.styles['Small'])
        rows = [['Ticker', 'Price', 'Ann. Return %', 'Vol %', 'Mkt Cap', 'P/E', 'P/S', 'P/B']]
        for k, v in self.competitor_data.items():
            rows.append([
                k,
                f"${self._fmt_num(v.get('current_price', 0))}",
                self._fmt_num(v.get('annual_return', 0)),
                self._fmt_num(v.get('volatility', 0)),
                f"${self._fmt_num(v.get('market_cap', 0))}",
                self._fmt_num(v.get('pe')) if v.get('pe') is not None else 'N/A',
                self._fmt_num(v.get('ps')) if v.get('ps') is not None else 'N/A',
                self._fmt_num(v.get('pb')) if v.get('pb') is not None else 'N/A',
            ])
        data = self._wrap_cells(rows, header_style='KPI', cell_style='Cell')
        t = Table(
            data,
            colWidths=[0.9*inch, 0.9*inch, 1.1*inch, 0.9*inch, 1.4*inch, 0.7*inch, 0.7*inch, 0.7*inch],
            hAlign='LEFT', repeatRows=1
        )
        t.setStyle(self._striped_table_style(header_bg=self.THEME['header']))
        t.setStyle(TableStyle([('WORDWRAP', (0,0), (-1,-1), 'CJK')]))
        return t

    def _scenario_table(self):
        km = self.financial_data['key_metrics']
        base_vol = km.get('volatility', 0)
        base_ret = km.get('annual_return', 0)
        price = km.get('current_price', 0) or 1
        scenarios = [
            ['Scenario', 'Assumed Ann. Return', 'Assumed Vol', '1Y Price (95% CI approx)'],
        ]
        for name, adj_ret, adj_vol in [
            ['Bear', base_ret - 15, base_vol + 10],
            ['Base', base_ret, base_vol],
            ['Bull', base_ret + 15, max(base_vol - 5, 5)],
        ]:
            mu_d = (adj_ret / 100) / 252
            sigma_d = (adj_vol / 100) / np.sqrt(252) if adj_vol else 0.0001
            mu_1y = (1 + mu_d) ** 252 - 1
            sd_1y = sigma_d * np.sqrt(252)
            low = price * (1 + mu_1y - 2*sd_1y)
            high = price * (1 + mu_1y + 2*sd_1y)
            scenarios.append([name, f"{adj_ret:.1f}%", f"{adj_vol:.1f}%", f"${low:.2f} to ${high:.2f}"])
        data = self._wrap_cells(scenarios, header_style='KPI', cell_style='Cell')
        t = Table(data, colWidths=[1.2*inch, 1.6*inch, 1.2*inch, 2.0*inch], hAlign='LEFT', repeatRows=1)
        t.setStyle(self._striped_table_style(header_bg=self.THEME['warn']))
        t.setStyle(TableStyle([('WORDWRAP', (0,0), (-1,-1), 'CJK')]))
        return t

    def _portfolio_fit_table(self):
        km = self.financial_data['key_metrics']
        beta = self.factor_data.get('beta', 0)
        risk_badge = "Low"
        vol = km.get('volatility', 0)
        if vol > 30: risk_badge = "High"
        elif vol > 20: risk_badge = "Medium"
        rows = [
            ['Dimension', 'Indicator'],
            ['Risk Level', f"{risk_badge} (Vol {vol:.1f}%)"],
            ['Market Sensitivity', f"Beta {beta:.2f} vs SPY"],
            ['Income Profile', f"Dividend Yield {km.get('dividend_yield', 0):.2f}%"],
            ['Drawdown Profile', f"Max Drawdown {km.get('max_drawdown', 0):.1f}%"],
            ['Return Quality', f"Sharpe {km.get('sharpe', 0):.2f}, Sortino {km.get('sortino', 0):.2f}"],
        ]
        data = self._wrap_cells(rows, header_style='KPI', cell_style='Cell')
        t = Table(data, colWidths=[2.4*inch, 3.6*inch], hAlign='LEFT', repeatRows=1)
        t.setStyle(self._striped_table_style(header_bg=self.THEME['accent']))
        t.setStyle(TableStyle([('WORDWRAP', (0,0), (-1,-1), 'CJK')]))
        return t

    def _image_if(self, charts, key, w, h):
        if key in charts:
            img = Image(charts[key], width=w, height=h)
            # Restrict to available frame size
            if self._current_doc is not None:
                max_w = self._current_doc.width
                max_h = self._current_doc.height * 0.65
                img._restrictSize(max_w, max_h)
            return img
        return None

    # ================== PDF generation ==================
    def generate_report(self):
        """Generate a comprehensive multi-page PDF report with a modern aesthetic"""
        if not self.fetch_data():
            print("Failed to fetch data. Cannot generate report.")
            return False

        charts = self.create_charts()

        # Use BaseDocTemplate for custom header/footer
        doc = BaseDocTemplate(
            self.filename,
            pagesize=A4,
            leftMargin=40, rightMargin=40,
            topMargin=72, bottomMargin=60
        )
        self._current_doc = doc
        doc.allowSplitting = 1

        frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height - 10, id='normal')
        template = PageTemplate(id='with-header', frames=[frame], onPage=self._draw_header_footer)
        doc.addPageTemplates([template])

        self.styles = self._build_styles()
        story = []

        # Cover
        company_name = self.company_info.get('longName', self.symbol)
        story.append(Spacer(1, 40))
        story.append(Paragraph(company_name, self.styles['Subtitle']))
        story.append(Paragraph(f"{self.symbol} Investment Analysis Report", self.styles['TitleXL']))
        story.append(Paragraph(datetime.now().strftime('%B %d, %Y'), self.styles['Small']))
        story.append(Spacer(1, 30))
        story.append(self._hr(thickness=1.2, color=self.THEME['accent']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Executive summary", self.styles['H1']))
        story.append(self._kpi_table(self.styles))
        story.append(PageBreak())

        # Company overview
        story.append(Paragraph("Company overview", self.styles['H1']))
        story.append(Paragraph("A concise snapshot of the company’s business profile, sector context, and fundamental footprint.", self.styles['Body']))
        story.append(self._company_overview_table())
        story.append(Spacer(1, 8))
        story.append(Paragraph("Business description", self.styles['H2']))
        story.append(Paragraph(self.company_info.get('longBusinessSummary', 'No business summary available.'), self.styles['Body']))
        story.append(PageBreak())

        # Price and trend
        story.append(Paragraph("Price trend and moving averages", self.styles['H1']))
        pimg = self._image_if(charts, 'price_mas', 6.2*inch, 3.6*inch)
        if pimg: story.append(pimg)
        story.append(Spacer(1, 10))
        story.append(Paragraph("The 50-day and 200-day moving averages help contextualize momentum and longer-term trend regimes.", self.styles['Body']))
        story.append(PageBreak())

        # RSI and MACD
        story.append(Paragraph("RSI and MACD", self.styles['H1']))
        rimg = self._image_if(charts, 'rsi_macd', 6.2*inch, 4.8*inch)
        if rimg: story.append(rimg)
        story.append(Spacer(1, 10))
        story.append(Paragraph("RSI gauges momentum extremes while MACD highlights trend inflections. Signals should be aligned with broader context.", self.styles['Body']))
        story.append(PageBreak())

        # Returns distribution and drawdown
        story.append(Paragraph("Distribution of returns and drawdown profile", self.styles['H1']))
        d1 = self._image_if(charts, 'returns_dist', 6.2*inch, 2.9*inch)
        if d1: story.append(d1)
        story.append(Spacer(1, 8))
        d2 = self._image_if(charts, 'drawdown', 6.2*inch, 2.6*inch)
        if d2: story.append(d2)
        story.append(Spacer(1, 10))
        story.append(Paragraph("Fat tails and depth/duration of drawdowns influence risk budgeting and position sizing.", self.styles['Body']))
        story.append(PageBreak())

        # Rolling volatility and correlation
        story.append(Paragraph("Rolling volatility and correlation with SPY", self.styles['H1']))
        rvimg = self._image_if(charts, 'rolling_vol_corr', 6.2*inch, 4.0*inch)
        if rvimg: story.append(rvimg)
        story.append(Spacer(1, 10))
        story.append(Paragraph("Volatility regimes and market linkage inform hedging and tactical allocation decisions.", self.styles['Body']))
        story.append(PageBreak())

        # Seasonality
        story.append(Paragraph("Seasonality analysis", self.styles['H1']))
        simg = self._image_if(charts, 'seasonality_heatmap', 6.0*inch, 3.9*inch)
        if simg: story.append(simg)
        story.append(Spacer(1, 8))
        story.append(Paragraph("Monthly seasonality can guide entry timing but should not override fundamentals or valuations.", self.styles['Body']))
        story.append(PageBreak())

        # Peers overview
        story.append(Paragraph("Peer set comparison", self.styles['H1']))
        story.append(Paragraph("Comparative metrics across selected peers contextualize relative performance and valuation.", self.styles['Body']))
        story.append(self._peer_table())
        cbar = self._image_if(charts, 'comp_peers_bars', 6.2*inch, 3.2*inch)
        if cbar:
            story.append(Spacer(1, 8))
            story.append(cbar)
        vpeers = self._image_if(charts, 'peer_valuation', 6.2*inch, 3.0*inch)
        if vpeers:
            story.append(Spacer(1, 8))
            story.append(vpeers)
        story.append(PageBreak())

        # Fundamental trends
        story.append(Paragraph("Fundamental trends", self.styles['H1']))
        rtrend = self._image_if(charts, 'revenue_trend', 6.2*inch, 2.8*inch)
        if rtrend: story.append(rtrend)
        etrend = self._image_if(charts, 'earnings_quarterly', 6.2*inch, 2.8*inch)
        if etrend:
            story.append(Spacer(1, 8))
            story.append(etrend)
        story.append(Spacer(1, 10))
        story.extend(self._fin_stat_tables())
        story.append(PageBreak())

        # Dividend profile
        story.append(Paragraph("Dividend profile", self.styles['H1']))
        divimg = self._image_if(charts, 'dividend_history', 6.0*inch, 2.6*inch)
        if divimg: story.append(divimg)
        story.append(Spacer(1, 8))
        story.append(Paragraph("Dividend sustainability depends on free cash flow coverage, payout ratio, and earnings stability.", self.styles['Body']))
        story.append(PageBreak())

        # Scenario analysis
        story.append(Paragraph("Scenario analysis (1-year horizon)", self.styles['H1']))
        story.append(self._scenario_table())
        mcimg = self._image_if(charts, 'monte_carlo', 6.2*inch, 3.5*inch)
        if mcimg:
            story.append(Spacer(1, 10))
            story.append(mcimg)
        story.append(Paragraph("The Monte Carlo simulation illustrates a range of plausible price paths based on historical drift and volatility.", self.styles['Body']))
        story.append(PageBreak())

        # Portfolio fit
        story.append(Paragraph("Portfolio fit and role", self.styles['H1']))
        story.append(self._portfolio_fit_table())
        story.append(Spacer(1, 8))
        story.append(Paragraph("Consider position sizing rules, stop-loss disciplines, and diversification benefits relative to your existing holdings.", self.styles['Body']))
        story.append(PageBreak())

        # Risk assessment and governance
        story.append(Paragraph("Risk assessment and governance considerations", self.styles['H1']))
        km = self.financial_data['key_metrics']
        risk_level = "Low"
        if km['volatility'] > 30: risk_level = "High"
        elif km['volatility'] > 20: risk_level = "Medium"
        risk_text = f"""
        • Price volatility: {km['volatility']:.2f}% (annualized) — {risk_level.lower()} risk regime.<br/>
        • Market sensitivity: Beta {self.factor_data.get('beta', 0):.2f} vs SPY; VAR 95% (daily) {self.factor_data.get('var_95', 0):.2f}%.<br/>
        • Drawdown depth: {km['max_drawdown']:.1f}% peak-to-trough in the historical window.<br/>
        • Return quality: Sharpe {km['sharpe']:.2f}, Sortino {km['sortino']:.2f}.<br/>
        • Liquidity: Consider average daily volume and spreads before large allocations.
        """
        story.append(Paragraph(risk_text, self.styles['Body']))
        story.append(PageBreak())

        # Outlook & recommendations
        story.append(Paragraph("Future outlook and recommendations", self.styles['H1']))
        recommendation = "Hold"
        if km['annual_return'] > 15 and km['volatility'] < 25:
            recommendation = "Buy"
        elif km['annual_return'] < -10 or km['volatility'] > 40:
            recommendation = "Sell"
        outlook_text = f"""
        <b>Recommendation:</b> {recommendation}<br/><br/>
        <b>Considerations:</b><br/>
        • Trend and momentum: watch MA50/MA200 alignment and RSI extremes.<br/>
        • Fundamentals: track revenue trajectory, margins, and cash conversion.<br/>
        • Valuation: compare P/E, P/S, and P/B against peers and historical bands.<br/>
        • Risk: volatility regime, drawdown tolerance, and diversification benefits.<br/>
        • Catalysts: earnings announcements, guidance updates, macro events.
        """
        story.append(Paragraph(outlook_text, self.styles['Body']))
        story.append(PageBreak())

        # Appendices
        story.append(Paragraph("Appendix A: Detailed financial statements (selected items)", self.styles['H1']))
        story.extend(self._fin_stat_tables())
        story.append(PageBreak())

        story.append(Paragraph("Appendix B: Methodology and assumptions", self.styles['H1']))
        story.append(Paragraph("""
        Data sources include historical price series and publicly available financial statements where provided. Returns are computed using adjusted close prices when available. Volatility is annualized standard deviation of daily returns. Beta is estimated via covariance with SPY returns over available history. Scenario ranges are illustrative and not predictive. Monte Carlo uses Gaussian innovations—real markets may deviate materially from normality.
        """, self.styles['Body']))
        story.append(PageBreak())

        story.append(Paragraph("Appendix C: Glossary", self.styles['H1']))
        story.append(Paragraph("""
        • Sharpe Ratio: Excess return per unit of total risk (std dev).<br/>
        • Sortino Ratio: Excess return per unit of downside risk.<br/>
        • Drawdown: Peak-to-trough decline during a specific period.<br/>
        • Beta: Sensitivity of asset returns to market returns.<br/>
        • VaR: Value at Risk—loss threshold at a given confidence level.<br/>
        • Seasonality: Recurring return patterns across calendar periods.
        """, self.styles['Body']))
        story.append(PageBreak())

        story.append(Paragraph("Appendix D: Disclaimers", self.styles['H1']))
        story.append(Paragraph("""
        This report is for informational purposes only and does not constitute investment advice. Past performance does not guarantee future results. All investments involve risks, including loss of principal. Investors should conduct their own research or consult a professional advisor to ensure suitability with personal objectives, constraints, and risk tolerance.
        """, self.styles['Body']))

        # Build PDF
        try:
            doc.build(story)
            print(f"Report successfully generated: {self.filename}")
            return True
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return False


def main():
    symbol = input("Enter stock symbol (e.g., AAPL, MSFT, TSLA): ").strip().upper()

    if not symbol:
        print("Please provide a valid stock symbol.")
        return

    print(f"Generating comprehensive investment analysis report for {symbol}...")
    print("Fetching data, creating charts, and assembling a multi-page PDF...")

    report = StockAnalysisReport(symbol)

    if report.generate_report():
        print(f"\n✓ Report successfully created: {report.filename}")
    else:
        print("✗ Failed to generate report. Please check the stock symbol and try again.")


if __name__ == "__main__":
    main()