from collections.abc import Sequence
import textwrap

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram

from utils.geography import (
    MunicipalityGeography,
    join_municipality_geography,
)
from utils.ip_utils import INCOME_GROUPS
from utils.municipal_profiles import (
    MunicipalityClusterAnalysis,
    MunicipalityIncomeProfile,
    ensure_municipality_income_profile,
    sort_municipalities_by_share,
    validate_income_group_sequence,
)


INCOME_GROUP_COLORS = {
    "Transferencias corrientes": "#009E73",
    "Transferencias de capital": "#CC79A7",
    "Otros ingresos": "#7A7A7A",
    "IPP": "#0072B2",
    "FCM": "#E69F00",
}

OTHER_INCOME_COMPONENT_COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
)
OTHER_INCOME_REMAINDER_COLOR = "#B3B3B3"


def _wrap_plot_label(value: object, *, width: int = 38) -> str:
    """Divide etiquetas largas sin perder el texto completo del hover."""
    return "<br>".join(
        textwrap.wrap(str(value), width=width, break_long_words=False)
    )


def _other_income_component_color(
    rank: int,
    *,
    is_remainder: bool = False,
) -> str:
    if is_remainder:
        return OTHER_INCOME_REMAINDER_COLOR

    return OTHER_INCOME_COMPONENT_COLORS[
        (int(rank) - 1) % len(OTHER_INCOME_COMPONENT_COLORS)
    ]


def _prepare_history(history: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    required_columns = {
        "Nombre Municipio",
        "Ejercicio",
        "grupo_ingreso",
        "ingreso_anual",
        "porcentaje_ingreso",
        "total_anual",
    }
    missing = sorted(required_columns.difference(history.columns))

    if missing:
        raise KeyError(
            "El historial no contiene las columnas requeridas: "
            f"{', '.join(missing)}"
        )

    municipality_names = history["Nombre Municipio"].dropna().unique()

    if len(municipality_names) != 1:
        raise ValueError(
            "El gráfico requiere el historial de exactamente una comuna."
        )

    plot_data = history.copy()
    plot_data["Año"] = plot_data["Ejercicio"].astype(int).astype(str)
    year_order = [
        str(year)
        for year in sorted(plot_data["Ejercicio"].dropna().astype(int).unique())
    ]

    return plot_data, str(municipality_names[0]), year_order


def _format_clp_millions(value: float) -> str:
    millions = f"{value / 1_000_000:,.0f}".replace(",", ".")

    return f"${millions} MM"


def _add_stacked_bar_traces(
    figure: go.Figure,
    plot_data: pd.DataFrame,
    *,
    year_order: list[str],
    group_order: Sequence[str],
    value_column: str,
    hovertemplate: str,
) -> None:
    """Agrega trazas compatibles con renderizadores antiguos de notebooks."""
    year_positions = list(range(len(year_order)))

    for group in group_order:
        group_data = (
            plot_data.loc[plot_data["grupo_ingreso"].eq(group)]
            .set_index("Año")
            .reindex(year_order)
        )
        figure.add_bar(
            name=group,
            x=year_positions,
            y=[float(value) for value in group_data[value_column]],
            customdata=list(year_order),
            width=0.72,
            marker_color=INCOME_GROUP_COLORS[group],
            hovertemplate=hovertemplate,
        )


def _configure_year_axis(
    figure: go.Figure,
    year_order: list[str],
) -> None:
    """Muestra años sobre posiciones numéricas estables en Plotly."""
    year_positions = list(range(len(year_order)))
    figure.update_xaxes(
        type="linear",
        tickmode="array",
        tickvals=year_positions,
        ticktext=year_order,
        range=[-0.5, len(year_order) - 0.5],
        fixedrange=True,
    )


def plot_annual_income_share(
    history: pd.DataFrame,
    *,
    group_order: Sequence[str] = INCOME_GROUPS,
    title: str | None = None,
) -> go.Figure:
    """
    Crea un gráfico 100% apilado de ingresos por año.

    Parameters
    ----------
    history:
        Resultado de ``build_municipality_income_history``.
    group_order:
        Lista o tupla con el orden de los segmentos y la leyenda. Debe
        contener exactamente una vez cada grupo de ``INCOME_GROUPS``.
    title:
        Título opcional. Si se omite, utiliza el nombre de la comuna.

    Returns
    -------
    plotly.graph_objects.Figure
        Figura interactiva con el total anual sobre cada barra.

    Examples
    --------
    >>> history = build_municipality_income_history(
    ...     presupuesto,
    ...     municipality="Renca",
    ... )
    >>> figure = plot_annual_income_share(history)
    >>> figure.show()
    """
    normalized_group_order = validate_income_group_sequence(
        group_order,
        parameter_name="group_order",
        require_all=True,
    )
    plot_data, municipality_name, year_order = _prepare_history(history)
    figure = go.Figure()
    _add_stacked_bar_traces(
        figure,
        plot_data,
        year_order=year_order,
        group_order=normalized_group_order,
        value_column="porcentaje_ingreso",
        hovertemplate=(
            "Año %{customdata}<br>"
            "%{fullData.name}: %{y:.2f}%<extra></extra>"
        ),
    )
    figure.update_layout(
        title=(
            title
            or f"Composición porcentual del ingreso anual de "
            f"{municipality_name} por año"
        ),
        barmode="stack",
        height=600,
        xaxis_title="Año",
        yaxis_title="Porcentaje del ingreso anual",
        legend_title_text="Grupo de ingreso",
        legend_traceorder="normal",
        hovermode="x unified",
        template="plotly_white",
    )
    _configure_year_axis(figure, year_order)
    figure.update_yaxes(range=[0, 110], ticksuffix="%")
    totals = (
        plot_data[["Ejercicio", "total_anual"]]
        .drop_duplicates()
        .sort_values("Ejercicio")
    )

    year_positions = {year: position for position, year in enumerate(year_order)}

    for row in totals.itertuples(index=False):
        figure.add_annotation(
            x=year_positions[str(int(row.Ejercicio))],
            y=103,
            text=_format_clp_millions(float(row.total_anual)),
            showarrow=False,
            yanchor="bottom",
        )

    return figure


def plot_annual_income_share_lines(
    history: pd.DataFrame,
    *,
    group_order: Sequence[str] = INCOME_GROUPS,
    title: str | None = None,
) -> go.Figure:
    """Crea líneas con la participación anual de cada grupo de ingreso."""
    normalized_group_order = validate_income_group_sequence(
        group_order,
        parameter_name="group_order",
        require_all=True,
    )
    plot_data, municipality_name, year_order = _prepare_history(history)
    year_positions = list(range(len(year_order)))
    figure = go.Figure()

    for group in normalized_group_order:
        group_data = (
            plot_data.loc[plot_data["grupo_ingreso"].eq(group)]
            .set_index("Año")
            .reindex(year_order)
        )
        figure.add_scatter(
            name=group,
            x=year_positions,
            y=[
                float(value)
                for value in group_data["porcentaje_ingreso"]
            ],
            customdata=list(year_order),
            mode="lines+markers",
            line={"color": INCOME_GROUP_COLORS[group], "width": 2.5},
            marker={"color": INCOME_GROUP_COLORS[group], "size": 7},
            connectgaps=False,
            hovertemplate=(
                "Año %{customdata}<br>"
                "%{fullData.name}: %{y:.2f}%<extra></extra>"
            ),
        )

    maximum_share = float(plot_data["porcentaje_ingreso"].max())
    y_axis_maximum = min(
        100.0,
        max(10.0, float(np.ceil(maximum_share * 1.1 / 10) * 10)),
    )
    figure.update_layout(
        title=(
            title
            or f"Evolución porcentual de los grupos de ingreso de "
            f"{municipality_name} por año"
        ),
        height=600,
        xaxis_title="Año",
        yaxis_title="Porcentaje del ingreso anual",
        legend_title_text="Grupo de ingreso",
        legend_traceorder="normal",
        hovermode="x unified",
        template="plotly_white",
    )
    _configure_year_axis(figure, year_order)
    figure.update_yaxes(range=[0, y_axis_maximum], ticksuffix="%")

    return figure


def _plot_periodic_income_share_lines(
    history: pd.DataFrame,
    *,
    group_order: Sequence[str] = INCOME_GROUPS,
    title: str | None = None,
    period_column: str,
    period_prefix: str,
    period_name: str,
    period_adjective: str,
) -> go.Figure:
    """Crea líneas con la participación por período y grupo de ingreso."""
    required_columns = {
        "Nombre Municipio",
        "Ejercicio",
        period_column,
        "grupo_ingreso",
        "porcentaje_ingreso",
    }
    missing = sorted(required_columns.difference(history.columns))

    if missing:
        raise KeyError(
            f"El historial {period_adjective} no contiene las columnas "
            "requeridas: "
            f"{', '.join(missing)}"
        )

    normalized_group_order = validate_income_group_sequence(
        group_order,
        parameter_name="group_order",
        require_all=True,
    )
    municipality_names = history["Nombre Municipio"].dropna().unique()

    if len(municipality_names) != 1:
        raise ValueError(
            "El gráfico requiere el historial de exactamente una comuna."
        )

    plot_data = history.copy()
    periods = sorted(
        {
            (int(year), int(period))
            for year, period in zip(
                plot_data["Ejercicio"],
                plot_data[period_column],
                strict=True,
            )
        }
    )
    period_positions = list(range(len(periods)))
    period_labels = [
        f"{period_prefix}{period}<br>{year}" for year, period in periods
    ]
    period_customdata = [
        [year, f"{period_prefix}{period}"] for year, period in periods
    ]
    period_index = pd.MultiIndex.from_tuples(
        periods,
        names=["Ejercicio", period_column],
    )
    figure = go.Figure()

    for group in normalized_group_order:
        group_data = (
            plot_data.loc[plot_data["grupo_ingreso"].eq(group)]
            .set_index(["Ejercicio", period_column])
            .reindex(period_index)
        )
        figure.add_scatter(
            name=group,
            x=period_positions,
            y=[
                float(value)
                for value in group_data["porcentaje_ingreso"]
            ],
            customdata=period_customdata,
            mode="lines+markers",
            line={"color": INCOME_GROUP_COLORS[group], "width": 2.5},
            marker={"color": INCOME_GROUP_COLORS[group], "size": 7},
            connectgaps=False,
            hovertemplate=(
                "Año %{customdata[0]}<br>"
                "%{customdata[1]}<br>"
                "%{fullData.name}: %{y:.2f}%<extra></extra>"
            ),
        )

    maximum_share = float(plot_data["porcentaje_ingreso"].max())
    y_axis_maximum = min(
        100.0,
        max(10.0, float(np.ceil(maximum_share * 1.1 / 10) * 10)),
    )
    first_year, last_year = periods[0][0], periods[-1][0]
    figure.update_layout(
        title=(
            title
            or f"Evolución porcentual {period_adjective} de los grupos de "
            f"ingreso "
            f"de {municipality_names[0]} ({first_year}–{last_year})"
        ),
        height=600,
        xaxis_title=period_name,
        yaxis_title=f"Porcentaje del ingreso {period_adjective}",
        legend_title_text="Grupo de ingreso",
        legend_traceorder="normal",
        hovermode="x unified",
        template="plotly_white",
    )
    figure.update_xaxes(
        type="linear",
        tickmode="array",
        tickvals=period_positions,
        ticktext=period_labels,
        range=[-0.5, len(periods) - 0.5],
        fixedrange=True,
    )
    figure.update_yaxes(range=[0, y_axis_maximum], ticksuffix="%")

    return figure


def plot_quarterly_income_share_lines(
    history: pd.DataFrame,
    *,
    group_order: Sequence[str] = INCOME_GROUPS,
    title: str | None = None,
) -> go.Figure:
    """Crea líneas con la participación trimestral por grupo de ingreso."""
    return _plot_periodic_income_share_lines(
        history,
        group_order=group_order,
        title=title,
        period_column="Trimestre",
        period_prefix="T",
        period_name="Trimestre",
        period_adjective="trimestral",
    )


def plot_four_month_income_share_lines(
    history: pd.DataFrame,
    *,
    group_order: Sequence[str] = INCOME_GROUPS,
    title: str | None = None,
) -> go.Figure:
    """Crea líneas con la participación cuatrimestral por grupo."""
    return _plot_periodic_income_share_lines(
        history,
        group_order=group_order,
        title=title,
        period_column="Cuatrimestre",
        period_prefix="C",
        period_name="Cuatrimestre",
        period_adjective="cuatrimestral",
    )


def plot_semiannual_income_share_lines(
    history: pd.DataFrame,
    *,
    group_order: Sequence[str] = INCOME_GROUPS,
    title: str | None = None,
) -> go.Figure:
    """Crea líneas con la participación semestral por grupo."""
    return _plot_periodic_income_share_lines(
        history,
        group_order=group_order,
        title=title,
        period_column="Semestre",
        period_prefix="S",
        period_name="Semestre",
        period_adjective="semestral",
    )


def _plot_periodic_income_amount_lines(
    history: pd.DataFrame,
    *,
    amount_column: str,
    period_column: str,
    period_prefix: str,
    period_name: str,
    period_adjective: str,
    group_order: Sequence[str] = INCOME_GROUPS,
    title: str | None = None,
) -> go.Figure:
    """Crea líneas de montos por período y grupo de ingreso."""
    required_columns = {
        "Nombre Municipio",
        "Ejercicio",
        period_column,
        "grupo_ingreso",
        amount_column,
    }
    missing = sorted(required_columns.difference(history.columns))

    if missing:
        raise KeyError(
            f"El historial {period_adjective} no contiene las columnas "
            "requeridas: "
            f"{', '.join(missing)}"
        )

    normalized_group_order = validate_income_group_sequence(
        group_order,
        parameter_name="group_order",
        require_all=True,
    )
    municipality_names = history["Nombre Municipio"].dropna().unique()

    if len(municipality_names) != 1:
        raise ValueError(
            "El gráfico requiere el historial de exactamente una comuna."
        )

    plot_data = history.copy()
    periods = sorted(
        {
            (int(year), int(period))
            for year, period in zip(
                plot_data["Ejercicio"],
                plot_data[period_column],
                strict=True,
            )
        }
    )
    period_positions = list(range(len(periods)))
    period_labels = [
        f"{period_prefix}{period}<br>{year}" for year, period in periods
    ]
    period_customdata = [
        [year, f"{period_prefix}{period}"] for year, period in periods
    ]
    period_index = pd.MultiIndex.from_tuples(
        periods,
        names=["Ejercicio", period_column],
    )
    figure = go.Figure()

    for group in normalized_group_order:
        group_data = (
            plot_data.loc[plot_data["grupo_ingreso"].eq(group)]
            .set_index(["Ejercicio", period_column])
            .reindex(period_index)
        )
        figure.add_scatter(
            name=group,
            x=period_positions,
            y=[
                float(value) / 1_000_000_000
                for value in group_data[amount_column]
            ],
            customdata=period_customdata,
            mode="lines+markers",
            line={"color": INCOME_GROUP_COLORS[group], "width": 2.5},
            marker={"color": INCOME_GROUP_COLORS[group], "size": 7},
            connectgaps=False,
            hovertemplate=(
                "Año %{customdata[0]}<br>"
                "%{customdata[1]}<br>"
                "%{fullData.name}: %{y:,.2f} mil MM CLP<extra></extra>"
            ),
        )

    amounts = plot_data[amount_column].astype(float).div(1_000_000_000)
    minimum_amount = float(amounts.min())
    maximum_amount = float(amounts.max())
    amount_range = maximum_amount - minimum_amount
    padding = max(
        amount_range * 0.08,
        max(abs(minimum_amount), abs(maximum_amount)) * 0.02,
        0.1,
    )
    y_axis_minimum = min(0.0, minimum_amount - padding)
    y_axis_maximum = max(0.0, maximum_amount + padding)
    first_year, last_year = periods[0][0], periods[-1][0]
    figure.update_layout(
        title=(
            title
            or f"Ingresos por grupo y período {period_adjective} de "
            f"{municipality_names[0]} ({first_year}–{last_year})"
        ),
        height=600,
        xaxis_title=period_name,
        yaxis_title="Ingresos percibidos (miles de millones de CLP)",
        legend_title_text="Grupo de ingreso",
        legend_traceorder="normal",
        hovermode="x unified",
        template="plotly_white",
    )
    figure.update_xaxes(
        type="linear",
        tickmode="array",
        tickvals=period_positions,
        ticktext=period_labels,
        range=[-0.5, len(periods) - 0.5],
        fixedrange=True,
    )
    figure.update_yaxes(range=[y_axis_minimum, y_axis_maximum])

    return figure


def plot_quarterly_income_amount_lines(
    history: pd.DataFrame,
    *,
    group_order: Sequence[str] = INCOME_GROUPS,
    title: str | None = None,
) -> go.Figure:
    """Crea líneas con los ingresos trimestrales por grupo."""
    return _plot_periodic_income_amount_lines(
        history,
        amount_column="ingreso_trimestral",
        period_column="Trimestre",
        period_prefix="T",
        period_name="Trimestre",
        period_adjective="trimestral",
        group_order=group_order,
        title=title,
    )


def plot_four_month_income_amount_lines(
    history: pd.DataFrame,
    *,
    group_order: Sequence[str] = INCOME_GROUPS,
    title: str | None = None,
) -> go.Figure:
    """Crea líneas con los ingresos cuatrimestrales por grupo."""
    return _plot_periodic_income_amount_lines(
        history,
        amount_column="ingreso_cuatrimestral",
        period_column="Cuatrimestre",
        period_prefix="C",
        period_name="Cuatrimestre",
        period_adjective="cuatrimestral",
        group_order=group_order,
        title=title,
    )


def plot_semiannual_income_amount_lines(
    history: pd.DataFrame,
    *,
    group_order: Sequence[str] = INCOME_GROUPS,
    title: str | None = None,
) -> go.Figure:
    """Crea líneas con los ingresos semestrales por grupo."""
    return _plot_periodic_income_amount_lines(
        history,
        amount_column="ingreso_semestral",
        period_column="Semestre",
        period_prefix="S",
        period_name="Semestre",
        period_adjective="semestral",
        group_order=group_order,
        title=title,
    )


def plot_other_income_semester_breakdown(
    breakdown: pd.DataFrame,
    *,
    title: str | None = None,
) -> go.Figure:
    """Descompone el monto semestral de ``Otros ingresos`` en barras."""
    required_columns = {
        "Nombre Municipio",
        "Ejercicio",
        "Semestre",
        "componente",
        "orden_componente",
        "ingreso_semestral",
        "participacion_otros_ingresos",
        "total_otros_ingresos",
    }
    missing = sorted(required_columns.difference(breakdown.columns))

    if missing:
        raise KeyError(
            "El desglose de Otros ingresos no contiene las columnas "
            f"requeridas: {', '.join(missing)}"
        )

    municipality_names = breakdown["Nombre Municipio"].dropna().unique()

    if len(municipality_names) != 1:
        raise ValueError(
            "El gráfico requiere el desglose de exactamente una comuna."
        )

    plot_data = breakdown.copy()
    periods = sorted(
        {
            (int(year), int(semester))
            for year, semester in zip(
                plot_data["Ejercicio"],
                plot_data["Semestre"],
                strict=True,
            )
        }
    )
    period_index = pd.MultiIndex.from_tuples(
        periods,
        names=["Ejercicio", "Semestre"],
    )
    period_positions = list(range(len(periods)))
    component_order = (
        plot_data[["componente", "orden_componente"]]
        .drop_duplicates()
        .sort_values(["orden_componente", "componente"], kind="stable")
    )
    figure = go.Figure()

    for component_row in component_order.itertuples(index=False):
        component = str(component_row.componente)
        rank = int(component_row.orden_componente)
        is_remainder = component == "Resto de otros ingresos"
        component_data = (
            plot_data.loc[plot_data["componente"].eq(component)]
            .set_index(["Ejercicio", "Semestre"])
            .reindex(period_index)
        )
        amounts = component_data["ingreso_semestral"].fillna(0)
        shares = component_data["participacion_otros_ingresos"]
        figure.add_bar(
            name=_wrap_plot_label(component),
            x=period_positions,
            y=[float(value) / 1_000_000_000 for value in amounts],
            customdata=[
                [year, f"S{semester}", component, float(share)]
                for (year, semester), share in zip(
                    periods,
                    shares.fillna(0),
                    strict=True,
                )
            ],
            width=0.72,
            marker_color=_other_income_component_color(
                rank,
                is_remainder=is_remainder,
            ),
            hovertemplate=(
                "Año %{customdata[0]} · %{customdata[1]}<br>"
                "%{customdata[2]}<br>"
                "Monto: %{y:,.2f} mil MM CLP<br>"
                "Participación en Otros ingresos: "
                "%{customdata[3]:.1f}%<extra></extra>"
            ),
        )

    total_data = (
        plot_data.groupby(
            ["Ejercicio", "Semestre"],
            observed=True,
        )["total_otros_ingresos"]
        .first()
        .reindex(period_index)
    )
    figure.add_scatter(
        name="Total Otros ingresos",
        x=period_positions,
        y=[float(value) / 1_000_000_000 for value in total_data],
        customdata=[
            [year, f"S{semester}"] for year, semester in periods
        ],
        mode="lines+markers",
        line={"color": "#222222", "width": 2.5},
        marker={"color": "#222222", "size": 7, "symbol": "diamond"},
        hovertemplate=(
            "Año %{customdata[0]} · %{customdata[1]}<br>"
            "Total Otros ingresos: %{y:,.2f} mil MM CLP"
            "<extra></extra>"
        ),
    )

    first_year, last_year = periods[0][0], periods[-1][0]
    figure.update_layout(
        title=(
            title
            or "Descomposición semestral de Otros ingresos de "
            f"{municipality_names[0]} ({first_year}–{last_year})"
        ),
        barmode="relative",
        height=720,
        xaxis_title="Semestre",
        yaxis_title="Ingresos percibidos (miles de millones de CLP)",
        legend_title_text="Componente de Otros ingresos",
        legend_traceorder="normal",
        hovermode="x unified",
        template="plotly_white",
    )
    figure.update_xaxes(
        type="linear",
        tickmode="array",
        tickvals=period_positions,
        ticktext=[
            f"S{semester}<br>{year}" for year, semester in periods
        ],
        range=[-0.5, len(periods) - 0.5],
        fixedrange=True,
    )
    figure.update_yaxes(zeroline=True, zerolinecolor="#666666")

    return figure


def plot_other_income_semester_variability(
    variability: pd.DataFrame,
    *,
    top_n: int = 6,
    title: str | None = None,
) -> go.Figure:
    """Ordena las cuentas por su diferencia promedio entre S1 y S2."""
    if top_n < 1:
        raise ValueError("top_n debe ser al menos 1.")

    required_columns = {
        "Nombre Municipio",
        "cuenta",
        "codigo_cuenta",
        "ranking_variabilidad",
        "variacion_promedio_abs_s1_s2",
        "diferencia_promedio_s1_s2",
        "participacion_variabilidad",
    }
    missing = sorted(required_columns.difference(variability.columns))

    if missing:
        raise KeyError(
            "El ranking de Otros ingresos no contiene las columnas "
            f"requeridas: {', '.join(missing)}"
        )

    municipality_names = variability["Nombre Municipio"].dropna().unique()

    if len(municipality_names) != 1:
        raise ValueError(
            "El gráfico requiere el ranking de exactamente una comuna."
        )

    plot_data = (
        variability.sort_values("ranking_variabilidad", kind="stable")
        .head(top_n)
        .copy()
    )
    labels = [
        _wrap_plot_label(account, width=44)
        for account in plot_data["cuenta"]
    ]
    colors = [
        _other_income_component_color(int(rank))
        for rank in plot_data["ranking_variabilidad"]
    ]
    figure = go.Figure()
    figure.add_bar(
        x=[
            float(value) / 1_000_000_000
            for value in plot_data["variacion_promedio_abs_s1_s2"]
        ],
        y=labels,
        orientation="h",
        marker_color=colors,
        customdata=[
            [
                account,
                code,
                float(signed_difference) / 1_000_000_000,
                float(share),
            ]
            for account, code, signed_difference, share in zip(
                plot_data["cuenta"],
                plot_data["codigo_cuenta"],
                plot_data["diferencia_promedio_s1_s2"],
                plot_data["participacion_variabilidad"],
                strict=True,
            )
        ],
        hovertemplate=(
            "%{customdata[0]} (%{customdata[1]})<br>"
            "Variación promedio |S1 − S2|: "
            "%{x:,.2f} mil MM CLP<br>"
            "Diferencia promedio S1 − S2: "
            "%{customdata[2]:,.2f} mil MM CLP<br>"
            "Parte de la variabilidad: %{customdata[3]:.1f}%"
            "<extra></extra>"
        ),
    )
    figure.update_layout(
        title=(
            title
            or "Partidas de Otros ingresos con mayor variación semestral "
            f"en {municipality_names[0]}"
        ),
        height=max(480, len(plot_data) * 82),
        xaxis_title=(
            "Variación promedio |S1 − S2| "
            "(miles de millones de CLP)"
        ),
        yaxis_title="",
        showlegend=False,
        template="plotly_white",
        margin={"l": 285, "r": 40, "t": 80, "b": 80},
    )
    figure.update_yaxes(autorange="reversed", automargin=True)

    return figure


def plot_municipality_income_share(
    income: pd.DataFrame | MunicipalityIncomeProfile,
    *,
    group_order: Sequence[str] = INCOME_GROUPS,
    sort_priority: Sequence[str] | None = None,
    sort_ascending: bool = False,
    title: str | None = None,
) -> go.Figure:
    """
    Crea un gráfico 100% apilado de ingresos entre comunas.

    Parameters
    ----------
    income:
        Ingresos clasificados con municipio, grupo y ``Percibidopag Total``.
        El DataFrame puede contener varias filas por municipio y grupo.
    group_order:
        Lista o tupla completa que controla los segmentos y la leyenda.
    sort_priority:
        Lista o tupla parcial de grupos. Las comunas se ordenan
        lexicográficamente por sus porcentajes en ese orden de prioridad.
        Si se omite, se utiliza el orden alfabético de las comunas.
    sort_ascending:
        Si es ``False``, las participaciones mayores aparecen primero.
    title:
        Título opcional del gráfico.

    Returns
    -------
    plotly.graph_objects.Figure
        Figura horizontal interactiva con porcentajes y montos en el hover.

    Examples
    --------
    >>> figure = plot_municipality_income_share(
    ...     income,
    ...     group_order=(
    ...         "Transferencias corrientes",
    ...         "Transferencias de capital",
    ...         "Otros ingresos",
    ...         "IPP",
    ...         "FCM",
    ...     ),
    ...     sort_priority=("IPP", "FCM"),
    ... )
    >>> figure.show()
    """
    normalized_group_order = validate_income_group_sequence(
        group_order,
        parameter_name="group_order",
        require_all=True,
    )
    profile = ensure_municipality_income_profile(income)
    municipality_order = sort_municipalities_by_share(
        profile,
        sort_priority=sort_priority,
        ascending=sort_ascending,
    )
    amounts = profile.amounts
    percentages = profile.shares
    figure = go.Figure()

    for group in normalized_group_order:
        group_amounts = amounts.loc[municipality_order, group]
        group_percentages = percentages.loc[municipality_order, group]
        figure.add_bar(
            name=group,
            x=[float(value) for value in group_percentages],
            y=list(municipality_order),
            customdata=[
                [float(value) / 1_000_000_000]
                for value in group_amounts
            ],
            orientation="h",
            width=0.72,
            marker_color=INCOME_GROUP_COLORS[group],
            hovertemplate=(
                "%{fullData.name}: %{x:.2f}%<br>"
                "Monto: %{customdata[0]:,.2f} mil MM CLP"
                "<extra></extra>"
            ),
        )

    figure.update_layout(
        title=(
            title
            or "Composición porcentual del ingreso anual por comuna"
        ),
        barmode="stack",
        height=max(700, len(municipality_order) * 25),
        xaxis_title="Porcentaje del ingreso anual",
        yaxis_title="Comuna",
        legend_title_text="Grupo de ingreso",
        legend_traceorder="normal",
        hovermode="y unified",
        template="plotly_white",
    )
    figure.update_xaxes(range=[0, 100], ticksuffix="%")
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=municipality_order,
        autorange="reversed",
    )

    return figure


def plot_municipality_income_amount(
    income: pd.DataFrame | MunicipalityIncomeProfile,
    *,
    group_order: Sequence[str] = INCOME_GROUPS,
    sort_ascending: bool = False,
    title: str | None = None,
) -> go.Figure:
    """Compara montos absolutos apilados entre comunas."""
    normalized_group_order = validate_income_group_sequence(
        group_order,
        parameter_name="group_order",
        require_all=True,
    )
    profile = ensure_municipality_income_profile(income)
    order_data = profile.totals.rename("total").reset_index()
    municipality_order = (
        order_data.sort_values(
            ["total", "Nombre Municipio"],
            ascending=[sort_ascending, True],
            kind="stable",
        )["Nombre Municipio"]
        .astype(str)
        .tolist()
    )
    figure = go.Figure()

    for group in normalized_group_order:
        group_amounts = profile.amounts.loc[municipality_order, group].div(
            1_000_000_000
        )
        figure.add_bar(
            name=group,
            x=[float(value) for value in group_amounts],
            y=list(municipality_order),
            orientation="h",
            width=0.72,
            marker_color=INCOME_GROUP_COLORS[group],
            hovertemplate=(
                "%{fullData.name}: %{x:,.2f} mil MM CLP<extra></extra>"
            ),
        )

    figure.update_layout(
        title=(
            title or "Ingresos anuales por comuna y grupo de ingreso"
        ),
        barmode="stack",
        height=max(700, len(municipality_order) * 25),
        xaxis_title="Ingresos percibidos (miles de millones de CLP)",
        yaxis_title="Comuna",
        legend_title_text="Grupo de ingreso",
        legend_traceorder="normal",
        hovermode="y unified",
        template="plotly_white",
    )
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=municipality_order,
        autorange="reversed",
    )

    return figure


def plot_annual_income_amount(
    history: pd.DataFrame,
    *,
    title: str | None = None,
) -> go.Figure:
    """
    Crea un gráfico apilado de los montos anuales de una comuna.

    Parameters
    ----------
    history:
        Resultado de ``build_municipality_income_history``.
    title:
        Título opcional. Si se omite, utiliza el nombre de la comuna.

    Returns
    -------
    plotly.graph_objects.Figure
        Figura expresada en miles de millones de CLP.

    Examples
    --------
    >>> history = build_municipality_income_history(
    ...     presupuesto,
    ...     municipality=8728,
    ... )
    >>> plot_annual_income_amount(history).show()
    """
    plot_data, municipality_name, year_order = _prepare_history(history)
    plot_data["ingreso_miles_millones"] = (
        plot_data["ingreso_anual"] / 1_000_000_000
    )
    figure = go.Figure()
    _add_stacked_bar_traces(
        figure,
        plot_data,
        year_order=year_order,
        group_order=INCOME_GROUPS,
        value_column="ingreso_miles_millones",
        hovertemplate=(
            "Año %{customdata}<br>"
            "%{fullData.name}: %{y:,.2f} mil MM CLP<extra></extra>"
        ),
    )
    figure.update_layout(
        title=(
            title
            or f"Ingresos anuales de {municipality_name} por grupo"
        ),
        barmode="stack",
        height=600,
        xaxis_title="Año",
        yaxis_title="Ingresos anuales percibidos (miles de millones de CLP)",
        legend_title_text="Grupo de ingreso",
        legend_traceorder="normal",
        hovermode="x unified",
        template="plotly_white",
    )
    _configure_year_axis(figure, year_order)
    totals = (
        plot_data[["Ejercicio", "total_anual"]]
        .drop_duplicates()
        .sort_values("Ejercicio")
    )
    maximum_total = float(totals["total_anual"].max()) / 1_000_000_000
    annotation_offset = maximum_total * 0.02 if maximum_total else 0.1

    year_positions = {year: position for position, year in enumerate(year_order)}

    for row in totals.itertuples(index=False):
        total_billions = float(row.total_anual) / 1_000_000_000
        figure.add_annotation(
            x=year_positions[str(int(row.Ejercicio))],
            y=total_billions + annotation_offset,
            text=_format_clp_millions(float(row.total_anual)),
            showarrow=False,
            yanchor="bottom",
        )

    figure.update_yaxes(
        range=[0, maximum_total * 1.12 if maximum_total else 1]
    )

    return figure


def _shared_income_log_scale(
    map_data: pd.DataFrame,
) -> tuple[tuple[float, float], list[float]]:
    columns = [
        "ingreso_total_mil_mm_clp",
        "ingreso_no_ip_mil_mm_clp",
    ]
    missing = sorted(set(columns).difference(map_data.columns))

    if missing:
        raise KeyError(
            "Faltan métricas para construir la escala común: "
            f"{', '.join(missing)}"
        )

    values = np.concatenate(
        [map_data[column].to_numpy(dtype=float) for column in columns]
    )
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(
            "Los mapas logarítmicos requieren valores finitos y positivos."
        )

    exponent_min = int(np.floor(np.log10(values.min()))) - 1
    exponent_max = int(np.ceil(np.log10(values.max()))) + 1
    candidates = sorted(
        factor * 10**exponent
        for exponent in range(exponent_min, exponent_max + 1)
        for factor in (1, 2, 5)
    )
    scale_min = max(value for value in candidates if value <= values.min())
    scale_max = min(value for value in candidates if value >= values.max())
    ticks = [
        value for value in candidates if scale_min <= value <= scale_max
    ]

    return (float(np.log10(scale_min)), float(np.log10(scale_max))), ticks


def plot_municipality_income_map(
    map_data: pd.DataFrame,
    geography: MunicipalityGeography,
    *,
    metric: str,
    title: str | None = None,
    subtitle: str | None = None,
) -> go.Figure:
    """Crea un coroplético de ingreso total o ingreso no IP."""
    metric_config = {
        "total": (
            "ingreso_total_mil_mm_clp",
            "Ingresos totales percibidos por comuna",
        ),
        "no_ip": (
            "ingreso_no_ip_mil_mm_clp",
            "Ingresos no IP por comuna",
        ),
    }

    if metric not in metric_config:
        raise ValueError("metric debe ser 'total' o 'no_ip'.")

    metric_column, default_title = metric_config[metric]
    required_columns = {
        "CUT_COM",
        "Nombre Municipio",
        metric_column,
        "ingreso_total_mil_mm_clp",
        "ingreso_no_ip_mil_mm_clp",
        "porcentaje_no_ip",
    }
    missing = sorted(required_columns.difference(map_data.columns))

    if missing:
        raise KeyError(
            "Faltan columnas para construir el mapa: "
            f"{', '.join(missing)}"
        )

    color_range, scale_ticks = _shared_income_log_scale(map_data)
    plot_data = map_data.copy()
    plot_data["valor_log"] = np.log10(plot_data[metric_column])
    figure = px.choropleth(
        plot_data,
        geojson=geography.geojson,
        locations="CUT_COM",
        featureidkey="properties.CUT_COM",
        color="valor_log",
        hover_name="Nombre Municipio",
        custom_data=[
            "ingreso_total_mil_mm_clp",
            "ingreso_no_ip_mil_mm_clp",
            "porcentaje_no_ip",
        ],
        color_continuous_scale="Viridis",
        range_color=color_range,
        projection="mercator",
        fitbounds="locations",
        basemap_visible=False,
        title=title or default_title,
        subtitle=subtitle,
    )
    figure.update_traces(
        marker_line_color="#f5f5f5",
        marker_line_width=0.7,
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Ingreso total: %{customdata[0]:,.2f} mil MM CLP<br>"
            "Ingreso no IP: %{customdata[1]:,.2f} mil MM CLP<br>"
            "Participación no IP: %{customdata[2]:.1f}%"
            "<extra></extra>"
        ),
    )
    figure.update_coloraxes(
        colorbar={
            "title": {"text": "Miles de millones<br>de CLP (log)"},
            "tickmode": "array",
            "tickvals": np.log10(scale_ticks).tolist(),
            "ticktext": [f"{value:g}" for value in scale_ticks],
        }
    )
    figure.update_geos(visible=False, fitbounds="locations")
    figure.update_layout(
        height=720,
        margin={"l": 10, "r": 10, "t": 100, "b": 10},
        template="plotly_white",
    )

    return figure


def plot_cluster_selection(
    analysis: MunicipalityClusterAnalysis,
    *,
    title: str | None = None,
) -> go.Figure:
    """Muestra el silhouette de cada cantidad candidata de clústeres."""
    scores = analysis.cluster_validation_scores
    selected = scores.loc[
        scores["Número de clústeres"].eq(
            analysis.selected_cluster_count
        )
    ].iloc[0]
    figure = go.Figure()
    figure.add_scatter(
        x=scores["Número de clústeres"].astype(int).tolist(),
        y=scores["Silhouette"].astype(float).tolist(),
        mode="lines+markers",
        name="Candidatos",
        line={"color": "#4C78A8"},
        marker={"size": 9},
        hovertemplate=(
            "%{x} clústeres<br>Silhouette: %{y:.3f}<extra></extra>"
        ),
    )
    figure.add_scatter(
        x=[analysis.selected_cluster_count],
        y=[float(selected["Silhouette"])],
        mode="markers",
        name="Solución seleccionada",
        marker={"color": "#E45756", "size": 14, "symbol": "diamond"},
        hovertemplate=(
            "Seleccionado: %{x} clústeres<br>"
            "Silhouette: %{y:.3f}<extra></extra>"
        ),
    )
    figure.update_layout(
        title=title or "Selección de la cantidad de clústeres",
        xaxis_title="Número de clústeres",
        yaxis_title="Silhouette promedio",
        hovermode="x unified",
        template="plotly_white",
        height=480,
    )
    figure.update_xaxes(dtick=1)

    return figure


def plot_cluster_validation(
    analysis: MunicipalityClusterAnalysis,
    *,
    title: str | None = None,
) -> go.Figure:
    """Compara tres criterios internos para las particiones candidatas."""
    scores = analysis.cluster_validation_scores
    selected_count = analysis.selected_cluster_count
    selected = scores.loc[
        scores["Número de clústeres"].eq(selected_count)
    ].iloc[0]
    metrics = (
        ("Silhouette", "Mayor es mejor"),
        ("Calinski-Harabasz", "Mayor es mejor"),
        ("Davies-Bouldin", "Menor es mejor"),
    )
    figure = make_subplots(
        rows=1,
        cols=len(metrics),
        subplot_titles=[
            f"{metric}<br><sup>{criterion}</sup>"
            for metric, criterion in metrics
        ],
        horizontal_spacing=0.08,
    )

    for column, (metric, _) in enumerate(metrics, start=1):
        figure.add_scatter(
            x=scores["Número de clústeres"].astype(int).tolist(),
            y=scores[metric].astype(float).tolist(),
            mode="lines+markers",
            name="Candidatos",
            legendgroup="candidates",
            showlegend=column == 1,
            line={"color": "#4C78A8"},
            marker={"size": 8},
            hovertemplate=(
                f"%{{x}} clústeres<br>{metric}: %{{y:.3f}}"
                "<extra></extra>"
            ),
            row=1,
            col=column,
        )
        figure.add_scatter(
            x=[selected_count],
            y=[float(selected[metric])],
            mode="markers",
            name="Solución seleccionada por silhouette",
            legendgroup="selected",
            showlegend=column == 1,
            marker={"color": "#E45756", "size": 13, "symbol": "diamond"},
            hovertemplate=(
                f"Seleccionado: %{{x}} clústeres<br>{metric}: %{{y:.3f}}"
                "<extra></extra>"
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(
            title_text="Número de clústeres",
            dtick=1,
            row=1,
            col=column,
        )
        figure.update_yaxes(title_text=metric, row=1, col=column)

    figure.update_layout(
        title=title or "Validación interna de las tipologías comunales",
        template="plotly_white",
        hovermode="closest",
        height=480,
        width=1180,
    )

    return figure


def plot_municipality_silhouette(
    analysis: MunicipalityClusterAnalysis,
    *,
    title: str | None = None,
) -> go.Figure:
    """Muestra cohesión y separación para cada comuna seleccionada."""
    plot_data = analysis.assignments.sort_values(
        ["cluster_id", "Silhouette", "Nombre Municipio"],
        ascending=[True, True, True],
        kind="stable",
    ).copy()
    municipality_order = plot_data["Nombre Municipio"].astype(str).tolist()
    figure = px.bar(
        plot_data,
        x="Silhouette",
        y="Nombre Municipio",
        color="Clúster",
        orientation="h",
        category_orders={
            "Nombre Municipio": municipality_order,
            "Clúster": [
                f"Clúster {cluster_id}"
                for cluster_id in range(
                    1,
                    analysis.selected_cluster_count + 1,
                )
            ],
        },
        color_discrete_sequence=px.colors.qualitative.Safe,
        hover_data={
            "cluster_id": False,
            "Silhouette": ":.3f",
            "Ingreso total (mil MM CLP)": ":,.2f",
        },
        title=title or "Silhouette por comuna",
    )
    figure.add_vline(x=0, line_color="#444444", line_width=1)
    figure.update_layout(
        xaxis_title="Silhouette",
        yaxis_title="Comuna",
        template="plotly_white",
        height=max(720, len(plot_data) * 24 + 180),
    )

    return figure


def plot_cluster_pca(
    analysis: MunicipalityClusterAnalysis,
    *,
    title: str | None = None,
) -> go.Figure:
    """Proyecta los perfiles CLR a dos componentes para exploración visual."""
    projection = analysis.assignments.copy()
    explained_cp1, explained_cp2 = analysis.pca_explained_variance_ratio
    figure = px.scatter(
        projection,
        x="CP1",
        y="CP2",
        color="Clúster",
        category_orders={
            "Clúster": [
                f"Clúster {cluster_id}"
                for cluster_id in range(
                    1,
                    analysis.selected_cluster_count + 1,
                )
            ]
        },
        color_discrete_sequence=px.colors.qualitative.Safe,
        hover_name="Nombre Municipio",
        hover_data={
            "CP1": ":.3f",
            "CP2": ":.3f",
            "Silhouette": ":.3f",
            "Ingreso total (mil MM CLP)": ":,.2f",
            "cluster_id": False,
        },
        title=title or "Proyección PCA de los perfiles CLR",
    )
    figure.update_traces(marker={"size": 11, "line": {"width": 0.5}})
    figure.update_layout(
        xaxis_title=f"CP1 ({explained_cp1:.1%} de varianza)",
        yaxis_title=f"CP2 ({explained_cp2:.1%} de varianza)",
        template="plotly_white",
        height=650,
    )

    return figure


def plot_clustered_income_heatmap(
    analysis: MunicipalityClusterAnalysis,
    *,
    title: str | None = None,
) -> go.Figure:
    """Muestra la composición comunal en el orden del dendrograma."""
    municipality_order = list(analysis.ordered_municipalities)
    shares = analysis.profile.shares.loc[municipality_order, INCOME_GROUPS]
    cluster_by_municipality = analysis.assignments.set_index(
        "Nombre Municipio"
    )["Clúster"]
    cluster_labels = cluster_by_municipality.loc[municipality_order].tolist()
    customdata = [
        [cluster_label] * len(INCOME_GROUPS)
        for cluster_label in cluster_labels
    ]
    figure = go.Figure(
        data=go.Heatmap(
            z=shares.to_numpy(dtype=float).tolist(),
            x=list(INCOME_GROUPS),
            y=municipality_order,
            customdata=customdata,
            colorscale="Viridis",
            zmin=0,
            zmax=float(shares.to_numpy(dtype=float).max()),
            colorbar={"title": {"text": "Participación (%)"}},
            hovertemplate=(
                "<b>%{y}</b><br>%{customdata}<br>"
                "%{x}: %{z:.2f}%<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=(
            title
            or "Perfiles de ingreso ordenados por similitud jerárquica"
        ),
        xaxis_title="Grupo de ingreso",
        yaxis_title="Comuna",
        height=max(760, len(municipality_order) * 21),
        template="plotly_white",
        margin={"l": 125, "r": 30, "t": 80, "b": 90},
    )
    figure.update_yaxes(autorange="reversed")

    return figure


def plot_income_cluster_dendrogram(
    analysis: MunicipalityClusterAnalysis,
    *,
    title: str | None = None,
) -> go.Figure:
    """Representa el árbol jerárquico utilizado para formar los clústeres."""
    municipality_names = analysis.profile.shares.index.astype(str).tolist()
    tree = dendrogram(
        analysis.linkage_matrix,
        labels=municipality_names,
        orientation="left",
        no_plot=True,
    )
    figure = go.Figure()

    for leaf_coordinates, distances in zip(
        tree["icoord"],
        tree["dcoord"],
        strict=True,
    ):
        figure.add_scatter(
            x=[float(value) for value in distances],
            y=[float(value) for value in leaf_coordinates],
            mode="lines",
            line={"color": "#526D82", "width": 1.2},
            hoverinfo="skip",
            showlegend=False,
        )

    leaf_labels = list(tree["ivl"])
    figure.update_layout(
        title=title or "Dendrograma de perfiles de ingreso municipal",
        xaxis_title="Distancia Ward sobre composición CLR",
        yaxis_title="Comuna",
        height=max(800, len(leaf_labels) * 21),
        template="plotly_white",
        margin={"l": 125, "r": 30, "t": 80, "b": 50},
    )
    figure.update_yaxes(
        tickmode="array",
        tickvals=[5 + 10 * position for position in range(len(leaf_labels))],
        ticktext=leaf_labels,
        autorange="reversed",
    )

    return figure


def plot_cluster_profile_summary(
    analysis: MunicipalityClusterAnalysis,
    *,
    title: str | None = None,
) -> go.Figure:
    """Compara la composición promedio de los clústeres seleccionados."""
    summary = analysis.summary.sort_values("cluster_id")
    figure = go.Figure()

    for group in INCOME_GROUPS:
        figure.add_bar(
            name=group,
            x=summary["Clúster"].astype(str).tolist(),
            y=summary[group].astype(float).tolist(),
            customdata=summary["Número de comunas"].astype(int).tolist(),
            marker_color=INCOME_GROUP_COLORS[group],
            hovertemplate=(
                "%{x}<br>%{fullData.name}: %{y:.2f}%<br>"
                "%{customdata} comunas<extra></extra>"
            ),
        )

    figure.update_layout(
        title=title or "Composición promedio por clúster",
        barmode="stack",
        xaxis_title="Clúster",
        yaxis_title="Participación promedio",
        legend_title_text="Grupo de ingreso",
        template="plotly_white",
        hovermode="x unified",
        height=560,
    )
    figure.update_yaxes(range=[0, 100], ticksuffix="%")

    return figure


def plot_income_cluster_map(
    analysis: MunicipalityClusterAnalysis,
    geography: MunicipalityGeography,
    *,
    title: str | None = None,
    subtitle: str | None = None,
) -> go.Figure:
    """Representa la pertenencia a clúster sobre los polígonos comunales."""
    cluster_data = analysis.assignments.merge(
        analysis.profile.shares.reset_index(),
        on="Nombre Municipio",
        how="left",
        validate="one_to_one",
    )
    map_data = join_municipality_geography(cluster_data, geography)
    cluster_order = [
        f"Clúster {cluster_id}"
        for cluster_id in range(1, analysis.selected_cluster_count + 1)
    ]
    figure = px.choropleth(
        map_data,
        geojson=geography.geojson,
        locations="CUT_COM",
        featureidkey="properties.CUT_COM",
        color="Clúster",
        category_orders={"Clúster": cluster_order},
        color_discrete_sequence=px.colors.qualitative.Safe,
        hover_name="Nombre Municipio",
        custom_data=[
            "Ingreso total (mil MM CLP)",
            "IPP",
            "FCM",
            "Transferencias corrientes",
            "Transferencias de capital",
            "Otros ingresos",
        ],
        projection="mercator",
        fitbounds="locations",
        basemap_visible=False,
        title=title or "Clústeres de composición del ingreso municipal",
        subtitle=subtitle,
    )
    figure.update_traces(
        marker_line_color="#f5f5f5",
        marker_line_width=0.7,
        hovertemplate=(
            "<b>%{hovertext}</b><br>%{fullData.name}<br>"
            "Ingreso total: %{customdata[0]:,.2f} mil MM CLP<br>"
            "IPP: %{customdata[1]:.1f}% · FCM: %{customdata[2]:.1f}%<br>"
            "Transf. corrientes: %{customdata[3]:.1f}%<br>"
            "Transf. capital: %{customdata[4]:.1f}%<br>"
            "Otros: %{customdata[5]:.1f}%<extra></extra>"
        ),
    )
    figure.update_geos(visible=False, fitbounds="locations")
    figure.update_layout(
        height=720,
        legend_title_text="Clúster",
        margin={"l": 10, "r": 10, "t": 100, "b": 10},
        template="plotly_white",
    )

    return figure
