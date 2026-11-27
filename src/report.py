"""
PDF report generator using ReportLab Platypus.

Produces an empirical model evaluation report at reports/model_report.pdf
in a two-column IEEE-style layout with dataset distributions, model comparison
metrics and charts, residual diagnostics, and permutation-based interpretability.
"""

import math
import os
from html import escape as _html_escape
from typing import Any

from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    HRFlowable,
    KeepTogether,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


def escape(value: object) -> str:
    """Escape XML special characters for ReportLab Paragraph markup."""
    return _html_escape(str(value), quote=False)


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
_DARK = colors.HexColor("#1a1a2e")
_ACCENT = colors.HexColor("#16213e")
_MEDIUM = colors.HexColor("#0f3460")
_LIGHT_BG = colors.HexColor("#f5f5f5")
_BORDER = colors.HexColor("#cccccc")


# ---------------------------------------------------------------------------
# Page template callbacks
# ---------------------------------------------------------------------------
def _header_footer(canvas, doc) -> None:
    """Draw header line and footer with page number on every page."""
    canvas.saveState()
    width, height = letter

    canvas.setStrokeColor(_DARK)
    canvas.setLineWidth(0.5)
    canvas.line(
        doc.leftMargin,
        height - doc.topMargin + 12,
        width - doc.rightMargin,
        height - doc.topMargin + 12,
    )
    canvas.setFont("Times-Italic", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(
        doc.leftMargin,
        height - doc.topMargin + 16,
        "End-to-End ML Pipeline — Empirical Model Evaluation Report",
    )

    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(colors.black)
    canvas.drawCentredString(
        width / 2.0,
        doc.bottomMargin - 20,
        f"— {canvas.getPageNumber()} —",
    )
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _build_bar_chart(labels: list[str], values: list[float]) -> Drawing:
    """Create a horizontal bar chart sized to fit a 2-column IEEE frame."""
    chart_height = max(115, len(labels) * 18 + 20)
    drawing = Drawing(235, chart_height)

    chart = HorizontalBarChart()
    chart.x = 76
    chart.y = 14
    chart.width = 108
    chart.height = chart_height - 24
    chart.data = [values]
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.fontName = "Times-Roman"
    chart.categoryAxis.labels.fontSize = 8
    min_val = min(0.0, min(values)) if values else 0.0
    max_val = max(values) if values else 0.0
    chart.valueAxis.valueMin = min_val * 1.1 if min_val < 0 else 0.0
    chart.valueAxis.valueMax = max_val * 1.25 if max_val > 0 else 1.0
    chart.valueAxis.labels.fontName = "Times-Roman"
    chart.valueAxis.labels.fontSize = 8
    chart.bars[0].fillColor = _MEDIUM
    chart.bars[0].strokeColor = _DARK
    chart.bars[0].strokeWidth = 0.3
    chart.barWidth = 11
    chart.barLabelFormat = "%.4f"
    chart.barLabels.fontName = "Times-Roman"
    chart.barLabels.fontSize = 7.5
    chart.barLabels.boxAnchor = "w"
    chart.barLabels.nudge = 4

    drawing.add(chart)
    return drawing


# ---------------------------------------------------------------------------
# Academic-style table builder
# ---------------------------------------------------------------------------
def _academic_table(
    data: list[list[str]],
    col_widths: list[float],
    *,
    zebra: bool = True,
    font_size: float = 9.0,
) -> Table:
    """Build a table with top/bottom rules, optional zebra striping, and wrapping."""
    styles = getSampleStyleSheet()
    header_style = ParagraphStyle(
        f"TableHeader_{font_size}",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=font_size,
        leading=font_size + 2,
        alignment=TA_CENTER,
    )
    cell_style = ParagraphStyle(
        f"TableCell_{font_size}",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=font_size,
        leading=font_size + 2,
        alignment=TA_CENTER,
    )

    wrapped_data = [
        [
            Paragraph(cell, header_style if r_idx == 0 else cell_style)
            for cell in row
        ]
        for r_idx, row in enumerate(data)
    ]

    tbl = Table(wrapped_data, colWidths=col_widths, hAlign="CENTER", repeatRows=1)
    style_cmds: list[Any] = [
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, _DARK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, _DARK),
        ("LINEBELOW", (0, -1), (-1, -1), 1.2, _DARK),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    if zebra:
        for row_idx in range(1, len(data)):
            if row_idx % 2 == 0:
                style_cmds.append(
                    ("BACKGROUND", (0, row_idx), (-1, row_idx), _LIGHT_BG)
                )
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------
def generate_report(
    model_results: list[dict[str, Any]],
    best_model_name: str,
    feature_names: list[str],
    train_size: int,
    test_size: int,
    output_dir: str = "reports",
    dataset_stats: dict[str, Any] | None = None,
    residual_stats: dict[str, Any] | None = None,
    importance_method: str = "Feature Importance",
) -> str:
    """Build and persist the empirical model evaluation PDF report."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_report.pdf")

    doc = BaseDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=1.0 * inch,
        bottomMargin=1.0 * inch,
    )

    title_height = 1.6 * inch
    col_gap = 0.25 * inch
    col_width = (doc.width - col_gap) / 2

    frame_title = Frame(
        doc.leftMargin,
        doc.bottomMargin + doc.height - title_height,
        doc.width,
        title_height,
        id="title",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    frame_col1_first = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        col_width,
        doc.height - title_height,
        id="col1_first",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    frame_col2_first = Frame(
        doc.leftMargin + col_width + col_gap,
        doc.bottomMargin,
        col_width,
        doc.height - title_height,
        id="col2_first",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    frame_col1 = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        col_width,
        doc.height,
        id="col1",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    frame_col2 = Frame(
        doc.leftMargin + col_width + col_gap,
        doc.bottomMargin,
        col_width,
        doc.height,
        id="col2",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )

    template_first = PageTemplate(
        id="First",
        frames=[frame_title, frame_col1_first, frame_col2_first],
        onPage=_header_footer,
    )
    template_later = PageTemplate(
        id="Later", frames=[frame_col1, frame_col2], onPage=_header_footer
    )
    doc.addPageTemplates([template_first, template_later])

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "AcademicTitle",
        parent=styles["Title"],
        fontName="Times-Bold",
        fontSize=15,
        spaceAfter=6,
        alignment=TA_CENTER,
        textColor=_DARK,
    )
    abstract_heading = ParagraphStyle(
        "AbstractHeading",
        parent=styles["Heading3"],
        fontName="Times-Bold",
        fontSize=10,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    abstract_body = ParagraphStyle(
        "AbstractBody",
        parent=styles["BodyText"],
        fontName="Times-Italic",
        fontSize=9.5,
        alignment=TA_JUSTIFY,
        leftIndent=0.08 * inch,
        rightIndent=0.08 * inch,
        spaceAfter=14,
        leading=13,
    )
    heading1_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading1"],
        fontName="Times-Bold",
        fontSize=10.5,
        spaceBefore=14,
        spaceAfter=6,
        textColor=_DARK,
    )
    heading2_style = ParagraphStyle(
        "SubsectionHeading",
        parent=styles["Heading2"],
        fontName="Times-Bold",
        fontSize=9.5,
        spaceBefore=10,
        spaceAfter=4,
        textColor=_ACCENT,
    )
    body_style = ParagraphStyle(
        "AcademicBody",
        parent=styles["BodyText"],
        fontName="Times-Roman",
        fontSize=9.5,
        alignment=TA_JUSTIFY,
        spaceAfter=7,
        leading=13,
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=8.5,
        alignment=TA_CENTER,
        spaceBefore=5,
        spaceAfter=10,
        textColor=colors.grey,
    )
    bullet_style = ParagraphStyle(
        "BulletBody",
        parent=body_style,
        leftIndent=0.25 * inch,
        bulletFontName="Times-Roman",
        bulletFontSize=10,
        spaceAfter=4,
    )

    elements: list[Any] = []

    total_samples = train_size + test_size
    test_ratio = test_size / total_samples if total_samples else 0.0

    best_result = next(
        res for res in model_results if res["name"] == best_model_name
    )
    best_model_escaped = escape(best_model_name)
    best_metrics = best_result["metrics"]
    r2 = float(best_metrics.get("r2", 0.0))
    mse = float(best_metrics.get("mse", 0.0))
    rmse = float(best_metrics.get("rmse", math.sqrt(mse) if mse > 0 else 0.0))
    mae = best_metrics.get("mae")

    # ==================================================================
    # TITLE
    # ==================================================================
    elements.append(Spacer(1, 0.12 * inch))
    elements.append(
        Paragraph(
            "End-to-End Machine Learning Pipeline: Empirical Model Evaluation Report",
            title_style,
        )
    )
    author_name = os.getenv("REPORT_AUTHOR", "").strip()
    author_email = os.getenv("REPORT_AUTHOR_EMAIL", "").strip()
    if author_name:
        elements.append(
            Paragraph(
                escape(author_name),
                ParagraphStyle(
                    "Author", fontName="Times-Bold", fontSize=11, alignment=TA_CENTER
                ),
            )
        )
    if author_email:
        elements.append(
            Paragraph(
                escape(author_email),
                ParagraphStyle(
                    "Email",
                    fontName="Times-Italic",
                    fontSize=9.5,
                    alignment=TA_CENTER,
                    textColor=colors.grey,
                    spaceAfter=12,
                ),
            )
        )
    if not author_name and not author_email:
        elements.append(
            Paragraph(
                "California Housing Regression Benchmark",
                ParagraphStyle(
                    "ReportSubtitle",
                    fontName="Times-Italic",
                    fontSize=10,
                    alignment=TA_CENTER,
                    textColor=colors.grey,
                    spaceAfter=12,
                ),
            )
        )
    elements.append(NextPageTemplate("Later"))
    elements.append(FrameBreak())

    # ==================================================================
    # ABSTRACT
    # ==================================================================
    elements.append(Paragraph("Abstract", abstract_heading))
    mae_clause = (
        f", and a Mean Absolute Error (MAE) of {float(mae):.4f} "
        f"(${float(mae) * 100000:,.0f})"
        if mae is not None
        else ""
    )
    abstract_text = (
        f"This report evaluates {len(model_results)} regression model(s) trained to "
        f"predict California block-group median house values (in $100,000s). Models "
        f"were trained on {train_size:,} observations and evaluated on a {test_ratio:.0%} "
        f"hold-out split of {test_size:,} observations. Among the evaluated candidates, "
        f"<b>{best_model_escaped}</b> achieved the strongest generalization with a "
        f"Coefficient of Determination (R²) of {r2:.4f}, a Root Mean Squared Error "
        f"(RMSE) of {rmse:.4f} (${rmse * 100000:,.0f}){mae_clause}. The following "
        f"sections document data quality and target censoring, empirical model "
        f"comparisons, residual error diagnostics, hyperparameter settings, and "
        f"hold-out feature sensitivity."
    )
    elements.append(Paragraph(abstract_text, abstract_body))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.black))
    elements.append(Spacer(1, 0.14 * inch))

    # ==================================================================
    # 1. INTRODUCTION
    # ==================================================================
    elements.append(Paragraph("1. Introduction", heading1_style))
    elements.append(
        Paragraph(
            "Predicting residential property values from aggregated census attributes "
            "requires modeling nonlinear interactions between socioeconomic drivers "
            "and spatial coordinates. The California Housing dataset (1990 U.S. Census) "
            "provides 8 numerical predictors per block group.",
            body_style,
        )
    )
    elements.append(
        Paragraph(
            "This pipeline standardizes ingestion, deterministic train/test splitting "
            "(random_state=42), experiment logging in MLflow, model serialization, "
            "validated FastAPI inference, and automated reporting.",
            body_style,
        )
    )

    # ==================================================================
    # 2. DATASET & TARGET DISTRIBUTION
    # ==================================================================
    elements.append(Paragraph("2. Dataset & Target Distribution", heading1_style))
    if dataset_stats is not None:
        t_stats = dataset_stats["target"]
        elements.append(
            Paragraph(
                f"The loaded dataset contains <b>{dataset_stats['total_rows']:,}</b> "
                f"rows across {len(feature_names)} numerical features with "
                f"<b>{dataset_stats['null_count']}</b> missing values, partitioned into "
                f"<b>{train_size:,}</b> training and <b>{test_size:,}</b> hold-out test "
                f"samples.",
                body_style,
            )
        )
        elements.append(
            Paragraph(
                f"The target (median house value in $100,000s) has mean "
                f"<b>{t_stats['mean']:.4f}</b> (SD {t_stats['std']:.4f}), median "
                f"<b>{t_stats['median']:.4f}</b> (IQR {t_stats['q25']:.4f}–"
                f"{t_stats['q75']:.4f}), and range {t_stats['min']:.4f}–"
                f"{t_stats['max']:.4f}. Importantly, <b>{dataset_stats['censored_count']:,}</b> "
                f"observations (<b>{dataset_stats['censored_pct']:.2f}%</b>) are "
                f"right-censored at the census cap of 5.00001 ($500,001), which "
                f"compresses upper-tail predictions.",
                body_style,
            )
        )
        elements.append(Paragraph("2.1 Empirical Feature Summary", heading2_style))
        feat_table_data = [["Feature", "Mean", "Std", "Min", "Max"]]
        for fstat in dataset_stats["features"]:
            feat_table_data.append(
                [
                    escape(fstat["name"]),
                    f"{fstat['mean']:.2f}",
                    f"{fstat['std']:.2f}",
                    f"{fstat['min']:.2f}",
                    f"{fstat['max']:.2f}",
                ]
            )
        feat_table = _academic_table(
            feat_table_data,
            [0.83 * inch, 0.62 * inch, 0.62 * inch, 0.62 * inch, 0.62 * inch],
            font_size=8.0,
        )
        elements.append(
            KeepTogether(
                [
                    Spacer(1, 0.06 * inch),
                    feat_table,
                    Paragraph(
                        "Table 1: Empirical feature distributions across the full dataset.",
                        caption_style,
                    ),
                ]
            )
        )
    else:
        elements.append(
            Paragraph(
                f"The experiment utilized <b>{total_samples:,}</b> observations "
                f"partitioned into <b>{train_size:,}</b> training and "
                f"<b>{test_size:,}</b> test samples across {len(feature_names)} "
                f"numerical attributes.",
                body_style,
            )
        )
        elements.append(Paragraph("2.1 Feature Inventory", heading2_style))
        feat_table_data = [["#", "Feature Name", "Description"]]
        feature_descriptions = {
            "MedInc": "Median income in block group ($10,000s)",
            "HouseAge": "Median house age in block group (years)",
            "AveRooms": "Average rooms per household",
            "AveBedrms": "Average bedrooms per household",
            "Population": "Block group population",
            "AveOccup": "Average household members",
            "Latitude": "Block group latitude coordinate",
            "Longitude": "Block group longitude coordinate",
        }
        for idx, name in enumerate(feature_names, 1):
            desc = feature_descriptions.get(name, "—")
            feat_table_data.append([str(idx), escape(name), escape(desc)])
        feat_table = _academic_table(
            feat_table_data,
            [0.3 * inch, 0.85 * inch, 2.15 * inch],
            font_size=8.5,
        )
        elements.append(
            KeepTogether(
                [
                    Spacer(1, 0.06 * inch),
                    feat_table,
                    Paragraph(
                        "Table 1: Feature inventory for the regression dataset.",
                        caption_style,
                    ),
                ]
            )
        )

    # ==================================================================
    # 3. EMPIRICAL MODEL COMPARISON & DIAGNOSTICS
    # ==================================================================
    elements.append(
        Paragraph("3. Empirical Model Comparison & Diagnostics", heading1_style)
    )
    elements.append(
        Paragraph(
            "Candidate models were evaluated on the identical hold-out test split. "
            "Table 2 summarizes R², RMSE, MAE, and MSE across all runs.",
            body_style,
        )
    )

    sorted_desc = sorted(
        model_results, key=lambda x: x["metrics"].get("r2", 0.0), reverse=True
    )
    comp_table_data = [["Model", "R²", "RMSE", "MAE", "MSE"]]
    for res in sorted_desc:
        m = res["metrics"]
        mse_val = float(m.get("mse", 0.0))
        rmse_val = float(m.get("rmse", math.sqrt(mse_val) if mse_val > 0 else 0.0))
        mae_val = m.get("mae")
        r2_val = float(m.get("r2", 0.0))
        raw_name = escape(res["name"])
        name_marker = f"{raw_name}*" if res["name"] == best_model_name else raw_name
        comp_table_data.append(
            [
                name_marker,
                f"{r2_val:.4f}",
                f"{rmse_val:.4f}",
                f"{float(mae_val):.4f}" if mae_val is not None else "—",
                f"{mse_val:.4f}",
            ]
        )

    comp_table = _academic_table(
        comp_table_data,
        [1.11 * inch, 0.55 * inch, 0.55 * inch, 0.55 * inch, 0.55 * inch],
        font_size=8.0,
    )
    elements.append(
        KeepTogether(
            [
                Spacer(1, 0.06 * inch),
                comp_table,
                Paragraph(
                    "Table 2: Hold-out test metrics ordered by R². Selected model marked (*).",
                    caption_style,
                ),
            ]
        )
    )

    if len(model_results) > 1:
        asc_models = sorted(
            model_results, key=lambda x: x["metrics"].get("r2", 0.0)
        )
        elements.append(
            _build_bar_chart(
                [str(r["name"]) for r in asc_models],
                [float(r["metrics"].get("r2", 0.0)) for r in asc_models],
            )
        )
        elements.append(
            Paragraph(
                "Figure 1: Hold-out R² comparison across evaluated architectures.",
                caption_style,
            )
        )

    elements.append(
        Paragraph("3.1 Best Model Selection & Metric Interpretation", heading2_style)
    )
    if r2 >= 0.8:
        fit_summary = "strong variance explanation on the hold-out split"
    elif r2 >= 0.6:
        fit_summary = "moderate-to-strong variance explanation on the hold-out split"
    else:
        fit_summary = "limited variance explanation on the hold-out split"

    mae_text = (
        f" The MAE of <b>{float(mae):.4f}</b> (${float(mae) * 100000:,.0f}) "
        f"reflects typical absolute error, whereas RMSE penalizes large tail misses."
        if mae is not None
        else ""
    )
    elements.append(
        Paragraph(
            f"<b>{best_model_escaped}</b> was selected with R² = <b>{r2:.4f}</b> "
            f"({fit_summary}). Its RMSE of <b>{rmse:.4f}</b> corresponds to "
            f"<b>${rmse * 100000:,.0f}</b> per block group.{mae_text}",
            body_style,
        )
    )

    if residual_stats is not None:
        elements.append(Paragraph("3.2 Residual & Error Diagnostics", heading2_style))
        ratio = residual_stats["rmse_mae_ratio"]
        tail_note = (
            "values above 1.25 indicate right-skewed error tails from "
            "outliers and high-value blocks"
            if ratio > 1.25
            else "error spread is relatively uniform without severe tail skew"
        )
        elements.append(
            Paragraph(
                f"Test residuals (actual minus predicted, <i>y</i> - <i>y_hat</i>) "
                f"exhibit mean bias <b>{residual_stats['bias']:+.4f}</b> "
                f"(SD {residual_stats['resid_std']:.4f}), median absolute error "
                f"(MedAE) <b>{residual_stats['medae']:.4f}</b> "
                f"(${residual_stats['medae'] * 100000:,.0f}), and 95th-percentile "
                f"absolute error <b>{residual_stats['p95_abs_err']:.4f}</b> "
                f"(${residual_stats['p95_abs_err'] * 100000:,.0f}). The RMSE/MAE "
                f"ratio is <b>{ratio:.2f}</b>; {tail_note}.",
                body_style,
            )
        )
        if (
            residual_stats.get("uncensored_rmse") is not None
            and residual_stats.get("censored_rmse") is not None
        ):
            elements.append(
                Paragraph(
                    f"Stratifying by the census cap shows RMSE = "
                    f"<b>{residual_stats['uncensored_rmse']:.4f}</b> on uncensored "
                    f"blocks (&lt; $500,001) versus RMSE = "
                    f"<b>{residual_stats['censored_rmse']:.4f}</b> on right-censored "
                    f"blocks (&ge; $500,001), confirming that boundary compression "
                    f"drives higher variance in luxury markets.",
                    body_style,
                )
            )

    # ==================================================================
    # 4. BEST MODEL CONFIGURATION
    # ==================================================================
    elements.append(Paragraph("4. Best Model Configuration", heading1_style))
    elements.append(
        Paragraph(
            f"Configured hyperparameters for <b>{best_model_escaped}</b> (logged to "
            f"MLflow for exact replication):",
            body_style,
        )
    )
    param_data = [["Hyperparameter", "Value"]] + [
        [escape(k), escape(v)] for k, v in best_result["params"].items()
    ]
    param_table = _academic_table(
        param_data, [1.45 * inch, 1.80 * inch], font_size=8.5
    )
    elements.append(
        KeepTogether(
            [
                Spacer(1, 0.06 * inch),
                param_table,
                Paragraph(
                    f"Table 3: Hyperparameters for {best_model_escaped}.",
                    caption_style,
                ),
            ]
        )
    )

    # ==================================================================
    # 5. FEATURE IMPORTANCE & INTERPRETABILITY
    # ==================================================================
    elements.append(
        Paragraph("5. Feature Importance & Interpretability", heading1_style)
    )
    feature_importances = best_result.get("feature_importances")
    feature_importances_std = best_result.get("feature_importances_std")

    if feature_names and feature_importances:
        elements.append(
            Paragraph(
                f"Feature sensitivity was quantified via <b>{escape(importance_method)}</b> "
                f"on the hold-out test set. Figure 2 and Table 4 report mean score "
                f"contributions across features.",
                body_style,
            )
        )
        paired = sorted(
            zip(feature_names, feature_importances), key=lambda x: x[1]
        )
        chart = _build_bar_chart(
            [p[0] for p in paired], [float(p[1]) for p in paired]
        )
        elements.append(Spacer(1, 0.06 * inch))
        elements.append(chart)
        fig_num = 2 if len(model_results) > 1 else 1
        elements.append(
            Paragraph(
                f"Figure {fig_num}: {escape(importance_method)} for {best_model_escaped}.",
                caption_style,
            )
        )

        if feature_importances_std and len(feature_importances_std) == len(
            feature_importances
        ):
            desc_with_std = sorted(
                zip(feature_names, feature_importances, feature_importances_std),
                key=lambda x: x[1],
                reverse=True,
            )
            fi_data = [["Rank", "Feature", "Mean ΔR²", "Std"]]
            for rank, (fname, fimp, fstd) in enumerate(desc_with_std, 1):
                fi_data.append(
                    [str(rank), escape(fname), f"{fimp:.4f}", f"± {fstd:.4f}"]
                )
            fi_table = _academic_table(
                fi_data,
                [0.38 * inch, 1.15 * inch, 0.90 * inch, 0.85 * inch],
                font_size=8.0,
            )
        else:
            desc_paired = sorted(
                zip(feature_names, feature_importances),
                key=lambda x: x[1],
                reverse=True,
            )
            fi_data = [["Rank", "Feature", "Importance"]]
            for rank, (fname, fimp) in enumerate(desc_paired, 1):
                fi_data.append([str(rank), escape(fname), f"{fimp:.6f}"])
            fi_table = _academic_table(
                fi_data, [0.45 * inch, 1.45 * inch, 1.35 * inch], font_size=8.5
            )
            desc_with_std = [(fn, fi, 0.0) for fn, fi in desc_paired]

        elements.append(
            KeepTogether(
                [
                    fi_table,
                    Paragraph(
                        "Table 4: Ranked hold-out feature importance scores.",
                        caption_style,
                    ),
                ]
            )
        )

        top1, top1_val, _ = desc_with_std[0]
        if len(desc_with_std) >= 2:
            top2, top2_val, _ = desc_with_std[1]
            top_msg = (
                f"The leading predictors by {escape(importance_method)} are "
                f"<b>{escape(top1)}</b> ({top1_val:.4f}) and "
                f"<b>{escape(top2)}</b> ({top2_val:.4f})."
            )
        else:
            top_msg = (
                f"The leading predictor is <b>{escape(top1)}</b> ({top1_val:.4f})."
            )
        spatial_note = (
            " Because Latitude and Longitude are strongly correlated in California "
            "coastal/inland geography, independent permutation reflects sensitivity "
            "to geographic location rather than isolated coordinate effects."
            if {"Latitude", "Longitude"} & {f[0] for f in desc_with_std[:2]}
            else ""
        )
        elements.append(Paragraph(f"{top_msg}{spatial_note}", body_style))
    else:
        elements.append(
            Paragraph(
                f"Feature importance values were not supplied for "
                f"<b>{best_model_escaped}</b>.",
                body_style,
            )
        )

    # ==================================================================
    # 6. LIMITATIONS & RECOMMENDATIONS
    # ==================================================================
    elements.append(
        Paragraph("6. Limitations & Recommendations", heading1_style)
    )
    elements.append(
        Paragraph(
            "Key structural constraints and operational guardrails for deployment:",
            body_style,
        )
    )
    censoring_note = (
        f"{dataset_stats['censored_pct']:.2f}% of targets are capped at $500,001"
        if dataset_stats is not None
        else "upper-tail target censoring can compress predictions"
    )
    recommendations = [
        f"<b>Historical &amp; Censoring Bias:</b> 1990 census attributes omit inflation, interest rates, and post-1990 urban growth; {censoring_note}, causing underestimation in high-value coastal tracts.",
        "<b>Out-of-Distribution Guardrails:</b> Enforce strict input bounds at the API boundary and route extreme occupancy, age, or upper-tail estimates for manual appraisal review.",
        "<b>Production Monitoring:</b> Log live feature distributions (PSI) and periodic residual drift by county/coordinate region before relying on predictions outside benchmark evaluation.",
    ]
    for rec in recommendations:
        elements.append(Paragraph(f"• {rec}", bullet_style))

    # ==================================================================
    # 7. PIPELINE & REPRODUCIBILITY
    # ==================================================================
    elements.append(
        Paragraph("7. Pipeline and Reproducibility", heading1_style)
    )
    pipeline_items = [
        "<b>Experiment Tracking:</b> MLflow records parameters, metrics, and model artifacts.",
        "<b>Model Serialization:</b> Persisted via <i>joblib</i> for reproducible local loading (trusted artifacts only).",
        "<b>API Serving:</b> FastAPI <i>/predict</i> enforces Pydantic bounds, non-finite float rejection, and restricted CORS.",
        "<b>Frontend &amp; CI:</b> Streamlit interactive client; GitHub Actions runs Ruff, pytest, and Docker builds.",
    ]
    for item in pipeline_items:
        elements.append(Paragraph(f"• {item}", bullet_style))

    # ==================================================================
    # 8. REFERENCES
    # ==================================================================
    elements.append(Paragraph("8. References", heading1_style))
    references = [
        "Pace, R. K., &amp; Barry, R. (1997). Sparse spatial autoregressions. <i>Statistics &amp; Probability Letters</i>, 33(3), 291–297.",
        "Breiman, L. (2001). Random forests (permutation importance). <i>Machine Learning</i>, 45(1), 5–32.",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. <i>JMLR</i>, 12, 2825–2830.",
        "Zaharia, M., et al. (2018). Accelerating the machine learning lifecycle with MLflow. <i>IEEE Data Eng. Bull.</i>, 41(4), 39–45.",
    ]
    for i, ref in enumerate(references, 1):
        elements.append(Paragraph(f"[{i}] {ref}", body_style))

    elements.append(Spacer(1, 0.2 * inch))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=_BORDER))
    elements.append(Spacer(1, 0.08 * inch))
    elements.append(
        Paragraph(
            "End of Report — Generated by End-to-End ML Pipeline",
            ParagraphStyle(
                "EndNote",
                parent=styles["Normal"],
                fontName="Times-Italic",
                fontSize=8.5,
                alignment=TA_CENTER,
                textColor=colors.grey,
            ),
        )
    )

    doc.build(elements)
    return output_path
