from collections.abc import Collection
from numbers import Integral
import unicodedata

import pandas as pd

from utils.data_schema import MONTH_NAMES
from utils.ip_utils import INCOME_GROUPS, get_income_group


def _require_columns(
    dataframe: pd.DataFrame,
    columns: Collection[str],
    *,
    operation: str,
) -> None:
    missing = sorted(set(columns).difference(dataframe.columns))

    if missing:
        raise KeyError(
            f"No se puede {operation}; faltan las columnas: "
            f"{', '.join(missing)}"
        )


def _normalize_municipality_name(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).strip().casefold())

    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )


def _municipal_income_mask(presupuesto: pd.DataFrame) -> pd.Series:
    service_name = (
        presupuesto["Nombre Serv Incorporado"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )
    service_is_municipal = service_name.isin(
        ["gestión municipal", "gestion municipal"]
    )

    if "Cod Serv Incorporado" in presupuesto.columns:
        service_is_municipal |= presupuesto["Cod Serv Incorporado"].eq(1)

    return presupuesto["Cod Tipo Cuenta"].eq(1) & service_is_municipal


def resolve_municipality(
    presupuesto: pd.DataFrame,
    municipality: str | int,
) -> tuple[int, str]:
    """
    Resuelve un nombre o código municipal a su código y nombre canónico.

    Parameters
    ----------
    presupuesto:
        Base presupuestaria normalizada.
    municipality:
        Nombre o código de la comuna.

    Returns
    -------
    tuple[int, str]
        Código municipal y nombre disponible en el año más reciente.

    Examples
    --------
    Resolver por nombre, sin depender de mayúsculas ni acentos:

    >>> resolve_municipality(presupuesto, "renca")
    (8728, 'Renca')

    Resolver directamente por código:

    >>> resolve_municipality(presupuesto, 8728)
    (8728, 'Renca')
    """
    _require_columns(
        presupuesto,
        {"Cod Municipio", "Nombre Municipio", "Ejercicio"},
        operation="resolver la comuna",
    )

    catalog = (
        presupuesto.loc[
            :,
            ["Cod Municipio", "Nombre Municipio", "Ejercicio"],
        ]
        .dropna(subset=["Cod Municipio", "Nombre Municipio"])
        .drop_duplicates()
        .copy()
    )

    if isinstance(municipality, str):
        requested_name = _normalize_municipality_name(municipality)

        if not requested_name:
            raise ValueError("El nombre de la comuna no puede estar vacío.")

        normalized_names = catalog["Nombre Municipio"].map(
            _normalize_municipality_name
        )
        matches = catalog.loc[normalized_names.eq(requested_name)]
    elif isinstance(municipality, Integral) and not isinstance(municipality, bool):
        matches = catalog.loc[
            catalog["Cod Municipio"].eq(int(municipality))
        ]
    else:
        raise TypeError(
            "municipality debe ser el nombre de una comuna o su código entero."
        )

    if matches.empty:
        raise ValueError(f"No se encontró la comuna: {municipality!r}.")

    municipality_codes = matches["Cod Municipio"].astype(int).unique()

    if len(municipality_codes) != 1:
        raise ValueError(
            f"La comuna {municipality!r} coincide con múltiples códigos: "
            f"{municipality_codes.tolist()}."
        )

    municipality_code = int(municipality_codes[0])
    canonical_rows = catalog.loc[
        catalog["Cod Municipio"].astype(int).eq(municipality_code)
    ].sort_values("Ejercicio", na_position="first")
    canonical_name = str(canonical_rows.iloc[-1]["Nombre Municipio"]).strip()

    return municipality_code, canonical_name


def filter_municipality_income(
    presupuesto: pd.DataFrame,
    *,
    municipality: str | int,
    years: Collection[int] | None = None,
) -> pd.DataFrame:
    """
    Selecciona los ingresos de Gestión Municipal de una comuna.

    Parameters
    ----------
    presupuesto:
        Base presupuestaria normalizada.
    municipality:
        Nombre o código de la comuna.
    years:
        Años que deben conservarse. Si es ``None``, conserva todos.

    Returns
    -------
    pd.DataFrame
        Filas de ingresos de la comuna solicitada.
    """
    _require_columns(
        presupuesto,
        {
            "Cod Municipio",
            "Nombre Municipio",
            "Cod Tipo Cuenta",
            "Nombre Serv Incorporado",
            "Ejercicio",
        },
        operation="filtrar los ingresos municipales",
    )
    municipality_code, municipality_name = resolve_municipality(
        presupuesto,
        municipality,
    )
    mask = (
        presupuesto["Cod Municipio"].eq(municipality_code)
        & _municipal_income_mask(presupuesto)
    )

    if years is not None:
        requested_years = {int(year) for year in years}
        mask &= presupuesto["Ejercicio"].isin(requested_years)

    result = presupuesto.loc[mask].copy()

    if result.empty:
        raise ValueError(
            "No se encontraron ingresos de Gestión Municipal para "
            f"{municipality_name}."
        )

    result["Nombre Municipio"] = municipality_name

    return result


def build_municipal_income_scope(
    presupuesto: pd.DataFrame,
    *,
    region: str | None = None,
    years: Collection[int] | None = None,
) -> pd.DataFrame:
    """
    Selecciona y clasifica ingresos municipales para análisis comparativos.

    A diferencia de ``filter_municipality_income``, conserva todas las comunas
    del ámbito solicitado. Esto permite construir perfiles, mapas y clústeres
    sin repetir filtros dentro de notebooks.
    """
    _require_columns(
        presupuesto,
        {
            "Ejercicio",
            "Región",
            "Cod Tipo Cuenta",
            "Nombre Serv Incorporado",
            "Nombre Municipio",
            "Percibidopag Total",
        },
        operation="construir el ámbito de ingresos municipales",
    )
    mask = _municipal_income_mask(presupuesto)

    if region is not None:
        requested_region = str(region).strip().casefold()
        mask &= (
            presupuesto["Región"]
            .astype("string")
            .str.strip()
            .str.casefold()
            .eq(requested_region)
        )

    if years is not None:
        requested_years = {int(year) for year in years}
        mask &= presupuesto["Ejercicio"].isin(requested_years)

    result = presupuesto.loc[mask].copy()

    if result.empty:
        raise ValueError(
            "No se encontraron ingresos municipales para el ámbito solicitado."
        )

    result["grupo_ingreso"] = result.apply(get_income_group, axis=1)
    unclassified = result["grupo_ingreso"].isna()

    if unclassified.any():
        raise ValueError(
            "Existen "
            f"{int(unclassified.sum())} filas que no pudieron clasificarse."
        )

    return result


def infer_year_coverage(presupuesto: pd.DataFrame) -> pd.DataFrame:
    """
    Infiere si cada ejercicio contiene un año completo o parcial.

    La inferencia se calcula sobre todos los ingresos de Gestión Municipal,
    no sobre una comuna específica. Para cada ejercicio se identifica el
    último mes con algún valor distinto de cero en las columnas mensuales de
    ``Percibidopag``.

    Parameters
    ----------
    presupuesto:
        Base presupuestaria normalizada.

    Returns
    -------
    pd.DataFrame
        Una fila por ejercicio con las columnas ``last_reported_month``,
        ``last_reported_month_number``, ``year_status`` e ``is_complete``.

    Examples
    --------
    Inspeccionar la cobertura de todos los archivos cargados:

    >>> coverage = infer_year_coverage(presupuesto)
    >>> coverage[["Ejercicio", "last_reported_month", "year_status"]]

    Consultar solamente los años parciales:

    >>> coverage.loc[coverage["year_status"].eq("partial")]

    Notes
    -----
    La fuente no incluye una bandera explícita de cierre anual. Por eso esta
    función utiliza actividad en diciembre como evidencia de un año completo.
    Un año sin columnas mensuales o sin actividad se marca como ``unknown``.
    """
    _require_columns(
        presupuesto,
        {"Ejercicio", "Cod Tipo Cuenta", "Nombre Serv Incorporado"},
        operation="inferir la cobertura anual",
    )
    monthly_columns = [
        (month_number, month_name, f"Percibidopag {month_name}")
        for month_number, month_name in enumerate(MONTH_NAMES, start=1)
        if f"Percibidopag {month_name}" in presupuesto.columns
    ]
    scope_mask = _municipal_income_mask(presupuesto)
    scope_columns = [
        "Ejercicio",
        *[column for _, _, column in monthly_columns],
    ]
    scope = presupuesto.loc[scope_mask, scope_columns]
    years = sorted(scope["Ejercicio"].dropna().astype(int).unique())

    if monthly_columns:
        activity = (
            scope[[column for _, _, column in monthly_columns]]
            .fillna(0)
            .ne(0)
            .groupby(scope["Ejercicio"], observed=True)
            .any()
        )
    else:
        activity = pd.DataFrame(index=years)

    records = []

    for year in years:
        active_months = [
            (month_number, month_name)
            for month_number, month_name, column in monthly_columns
            if year in activity.index and bool(activity.loc[year, column])
        ]

        if active_months:
            last_month_number, last_month_name = active_months[-1]
            year_status = (
                "complete" if last_month_number == 12 else "partial"
            )
            is_complete = last_month_number == 12
        else:
            last_month_number = None
            last_month_name = None
            year_status = "unknown"
            is_complete = pd.NA

        records.append(
            {
                "Ejercicio": year,
                "last_reported_month": last_month_name,
                "last_reported_month_number": last_month_number,
                "year_status": year_status,
                "is_complete": is_complete,
            }
        )

    coverage = pd.DataFrame.from_records(records)

    if coverage.empty:
        return pd.DataFrame(
            columns=[
                "Ejercicio",
                "last_reported_month",
                "last_reported_month_number",
                "year_status",
                "is_complete",
            ]
        )

    coverage["Ejercicio"] = coverage["Ejercicio"].astype("Int64")
    coverage["last_reported_month"] = coverage[
        "last_reported_month"
    ].astype("string")
    coverage["last_reported_month_number"] = coverage[
        "last_reported_month_number"
    ].astype("Int64")
    coverage["year_status"] = coverage["year_status"].astype("string")
    coverage["is_complete"] = coverage["is_complete"].astype("boolean")

    return coverage


def build_municipality_income_history(
    presupuesto: pd.DataFrame,
    *,
    municipality: str | int,
    complete_years_only: bool = True,
    exclude_years: Collection[int] = (),
) -> pd.DataFrame:
    """
    Construye el historial anual de ingresos de una comuna.

    Filtra los ingresos de Gestión Municipal, los clasifica en grupos
    mutuamente excluyentes y calcula el monto, total y porcentaje anual.

    Parameters
    ----------
    presupuesto:
        Base presupuestaria normalizada.
    municipality:
        Nombre o código de la comuna. Por ejemplo, ``"Renca"`` o ``8728``.
    complete_years_only:
        Si es ``True``, excluye automáticamente años parciales o desconocidos.
    exclude_years:
        Años adicionales que deben excluirse explícitamente.

    Returns
    -------
    pd.DataFrame
        Historial anual con grupo de ingreso, monto, porcentaje, total y
        estado de cobertura.

    Examples
    --------
    Consultar Renca por nombre:

    >>> history = build_municipality_income_history(
    ...     presupuesto,
    ...     municipality="Renca",
    ... )

    Consultar la comuna mediante su código:

    >>> history = build_municipality_income_history(
    ...     presupuesto,
    ...     municipality=8728,
    ... )

    Excluir años adicionales:

    >>> history = build_municipality_income_history(
    ...     presupuesto,
    ...     municipality="Renca",
    ...     exclude_years={2021, 2022},
    ... )

    Incluir años parciales:

    >>> history = build_municipality_income_history(
    ...     presupuesto,
    ...     municipality="Renca",
    ...     complete_years_only=False,
    ... )

    Notes
    -----
    La completitud se determina usando toda la actividad mensual disponible en
    ``presupuesto``. Si ``data_loader`` ya filtró una comuna, la inferencia
    describe esa comuna. Las exclusiones explícitas siempre tienen prioridad.
    """
    _require_columns(
        presupuesto,
        {
            "Percibidopag Total",
            "Cod Subtítulo",
            "Cod Subasignación",
        },
        operation="construir el historial municipal",
    )
    municipality_code, municipality_name = resolve_municipality(
        presupuesto,
        municipality,
    )
    coverage = infer_year_coverage(presupuesto)
    selected_years = set(coverage["Ejercicio"].dropna().astype(int))

    if complete_years_only:
        selected_years &= set(
            coverage.loc[
                coverage["year_status"].eq("complete"),
                "Ejercicio",
            ].astype(int)
        )

    selected_years -= {int(year) for year in (exclude_years or ())}

    if not selected_years:
        raise ValueError(
            "No quedan años disponibles después de aplicar la cobertura "
            "y las exclusiones solicitadas."
        )

    municipal_income = filter_municipality_income(
        presupuesto,
        municipality=municipality_code,
        years=selected_years,
    )
    municipal_income["grupo_ingreso"] = municipal_income.apply(
        get_income_group,
        axis=1,
    )
    unclassified = municipal_income["grupo_ingreso"].isna()

    if unclassified.any():
        raise ValueError(
            "Existen "
            f"{int(unclassified.sum())} filas que no pudieron clasificarse."
        )

    annual_income = municipal_income.groupby(
        ["Ejercicio", "grupo_ingreso"],
        observed=True,
    )["Percibidopag Total"].sum()
    complete_grid = pd.MultiIndex.from_product(
        [sorted(selected_years), INCOME_GROUPS],
        names=["Ejercicio", "grupo_ingreso"],
    )
    history = (
        annual_income.reindex(complete_grid, fill_value=0)
        .rename("ingreso_anual")
        .reset_index()
    )
    history["total_anual"] = history.groupby("Ejercicio")[
        "ingreso_anual"
    ].transform("sum")
    history["porcentaje_ingreso"] = (
        history["ingreso_anual"]
        .div(history["total_anual"].replace(0, pd.NA))
        .mul(100)
        .fillna(0)
    )
    history.insert(0, "Nombre Municipio", municipality_name)
    history.insert(0, "Cod Municipio", municipality_code)
    history = history.merge(
        coverage[
            [
                "Ejercicio",
                "last_reported_month",
                "year_status",
                "is_complete",
            ]
        ],
        on="Ejercicio",
        how="left",
        validate="many_to_one",
    )

    return history[
        [
            "Cod Municipio",
            "Nombre Municipio",
            "Ejercicio",
            "grupo_ingreso",
            "ingreso_anual",
            "porcentaje_ingreso",
            "total_anual",
            "last_reported_month",
            "year_status",
            "is_complete",
        ]
    ]


def build_municipalities_income_history(
    presupuesto: pd.DataFrame,
    *,
    municipalities: Collection[str | int],
    complete_years_only: bool = True,
    exclude_years: Collection[int] = (),
) -> pd.DataFrame:
    """Construye y concatena historiales anuales para varias comunas."""
    if isinstance(municipalities, (str, bytes)) or not isinstance(
        municipalities,
        Collection,
    ):
        raise TypeError(
            "municipalities debe ser una colección de nombres o códigos."
        )
    if not municipalities:
        raise ValueError("municipalities no puede estar vacío.")

    municipality_codes: list[int] = []
    seen_codes: set[int] = set()

    for municipality in municipalities:
        municipality_code, _ = resolve_municipality(
            presupuesto,
            municipality,
        )
        if municipality_code not in seen_codes:
            municipality_codes.append(municipality_code)
            seen_codes.add(municipality_code)

    histories = []

    for municipality_code in municipality_codes:
        municipality_data = presupuesto.loc[
            presupuesto["Cod Municipio"].eq(municipality_code)
        ].copy()
        histories.append(
            build_municipality_income_history(
                municipality_data,
                municipality=municipality_code,
                complete_years_only=complete_years_only,
                exclude_years=exclude_years,
            )
        )

    return pd.concat(histories, ignore_index=True)


def _build_municipality_periodic_income_history(
    presupuesto: pd.DataFrame,
    *,
    municipality: str | int,
    start_year: int | None = None,
    complete_years_only: bool = True,
    exclude_years: Collection[int] = (),
    months_per_period: int,
    period_column: str,
    period_prefix: str,
    income_column: str,
    total_column: str,
    period_adjective: str,
) -> pd.DataFrame:
    """Construye una composición periódica desde las columnas mensuales."""
    monthly_columns = [
        f"Percibidopag {month_name}" for month_name in MONTH_NAMES
    ]
    _require_columns(
        presupuesto,
        {
            "Cod Subtítulo",
            "Cod Subasignación",
            *monthly_columns,
        },
        operation=f"construir el historial {period_adjective} municipal",
    )
    municipality_code, municipality_name = resolve_municipality(
        presupuesto,
        municipality,
    )
    coverage = infer_year_coverage(presupuesto)
    selected_years = set(coverage["Ejercicio"].dropna().astype(int))

    if complete_years_only:
        selected_years &= set(
            coverage.loc[
                coverage["year_status"].eq("complete"),
                "Ejercicio",
            ].astype(int)
        )

    if start_year is not None:
        selected_years = {
            year for year in selected_years if year >= int(start_year)
        }

    selected_years -= {int(year) for year in (exclude_years or ())}

    if not selected_years:
        raise ValueError(
            "No quedan años disponibles para construir el historial "
            f"{period_adjective}."
        )

    municipal_income = filter_municipality_income(
        presupuesto,
        municipality=municipality_code,
        years=selected_years,
    )
    municipal_income["grupo_ingreso"] = municipal_income.apply(
        get_income_group,
        axis=1,
    )
    unclassified = municipal_income["grupo_ingreso"].isna()

    if unclassified.any():
        raise ValueError(
            "Existen "
            f"{int(unclassified.sum())} filas que no pudieron clasificarse."
        )

    period_count = len(MONTH_NAMES) // months_per_period
    period_frames = []

    for period in range(1, period_count + 1):
        first_month = (period - 1) * months_per_period
        period_columns = monthly_columns[
            first_month:first_month + months_per_period
        ]
        period_data = municipal_income[
            ["Ejercicio", "grupo_ingreso"]
        ].copy()
        period_data[period_column] = period
        period_data[income_column] = (
            municipal_income[period_columns]
            .sum(axis=1, min_count=1)
            .fillna(0)
        )
        period_frames.append(period_data)

    periodic_income = (
        pd.concat(period_frames, ignore_index=True)
        .groupby(
            ["Ejercicio", period_column, "grupo_ingreso"],
            observed=True,
        )[income_column]
        .sum()
    )
    complete_grid = pd.MultiIndex.from_product(
        [
            sorted(selected_years),
            range(1, period_count + 1),
            INCOME_GROUPS,
        ],
        names=["Ejercicio", period_column, "grupo_ingreso"],
    )
    history = (
        periodic_income.reindex(complete_grid, fill_value=0)
        .rename(income_column)
        .reset_index()
    )
    history[total_column] = history.groupby(
        ["Ejercicio", period_column],
        observed=True,
    )[income_column].transform("sum")
    history["porcentaje_ingreso"] = (
        history[income_column]
        .div(history[total_column].replace(0, pd.NA))
        .mul(100)
        .fillna(0)
    )
    history["Periodo"] = (
        history["Ejercicio"].astype(int).astype(str)
        + f" {period_prefix}"
        + history[period_column].astype(int).astype(str)
    )
    history.insert(0, "Nombre Municipio", municipality_name)
    history.insert(0, "Cod Municipio", municipality_code)
    history = history.merge(
        coverage[
            [
                "Ejercicio",
                "last_reported_month",
                "year_status",
                "is_complete",
            ]
        ],
        on="Ejercicio",
        how="left",
        validate="many_to_one",
    )

    return history[
        [
            "Cod Municipio",
            "Nombre Municipio",
            "Ejercicio",
            period_column,
            "Periodo",
            "grupo_ingreso",
            income_column,
            "porcentaje_ingreso",
            total_column,
            "last_reported_month",
            "year_status",
            "is_complete",
        ]
    ].sort_values(["Ejercicio", period_column, "grupo_ingreso"])


def build_municipality_quarterly_income_history(
    presupuesto: pd.DataFrame,
    *,
    municipality: str | int,
    start_year: int | None = None,
    complete_years_only: bool = True,
    exclude_years: Collection[int] = (),
) -> pd.DataFrame:
    """Construye la composición trimestral de ingresos de una comuna."""
    return _build_municipality_periodic_income_history(
        presupuesto,
        municipality=municipality,
        start_year=start_year,
        complete_years_only=complete_years_only,
        exclude_years=exclude_years,
        months_per_period=3,
        period_column="Trimestre",
        period_prefix="T",
        income_column="ingreso_trimestral",
        total_column="total_trimestral",
        period_adjective="trimestral",
    )


def build_municipality_four_month_income_history(
    presupuesto: pd.DataFrame,
    *,
    municipality: str | int,
    start_year: int | None = None,
    complete_years_only: bool = True,
    exclude_years: Collection[int] = (),
) -> pd.DataFrame:
    """Construye la composición cuatrimestral de ingresos de una comuna."""
    return _build_municipality_periodic_income_history(
        presupuesto,
        municipality=municipality,
        start_year=start_year,
        complete_years_only=complete_years_only,
        exclude_years=exclude_years,
        months_per_period=4,
        period_column="Cuatrimestre",
        period_prefix="C",
        income_column="ingreso_cuatrimestral",
        total_column="total_cuatrimestral",
        period_adjective="cuatrimestral",
    )


def build_municipality_semiannual_income_history(
    presupuesto: pd.DataFrame,
    *,
    municipality: str | int,
    start_year: int | None = None,
    complete_years_only: bool = True,
    exclude_years: Collection[int] = (),
) -> pd.DataFrame:
    """Construye la composición semestral de ingresos de una comuna."""
    return _build_municipality_periodic_income_history(
        presupuesto,
        municipality=municipality,
        start_year=start_year,
        complete_years_only=complete_years_only,
        exclude_years=exclude_years,
        months_per_period=6,
        period_column="Semestre",
        period_prefix="S",
        income_column="ingreso_semestral",
        total_column="total_semestral",
        period_adjective="semestral",
    )


def _latest_meaningful_account_name(
    rows: pd.DataFrame,
    column: str,
) -> str | None:
    if column not in rows.columns:
        return None

    values = rows[column].dropna().astype(str).str.strip()
    values = values.loc[values.ne("")]
    values = values.loc[
        ~values.map(_normalize_municipality_name).isin(
            {"sin desagregacion", "nan", "none"}
        )
    ]

    return values.iloc[-1] if not values.empty else None


def _account_label_tokens(value: str) -> set[str]:
    normalized = _normalize_municipality_name(value)
    tokenized = "".join(
        character if character.isalnum() else " "
        for character in normalized
    )

    return set(tokenized.split())


def _other_income_account_labels(
    other_income: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = ["Cod Subtítulo", "Cod Subasignación"]
    records = []

    for account_key, rows in other_income.sort_values("Ejercicio").groupby(
        key_columns,
        dropna=False,
        sort=True,
    ):
        subtitle_code, subassignment_code = account_key
        assignment = _latest_meaningful_account_name(
            rows,
            "Nombre Asignación",
        )
        subassignment = _latest_meaningful_account_name(
            rows,
            "Nombre Subasignación",
        )
        item = _latest_meaningful_account_name(rows, "Nombre Ítem")
        subtitle = _latest_meaningful_account_name(
            rows,
            "Nombre Subtítulo",
        )

        if assignment and subassignment:
            normalized_assignment = _normalize_municipality_name(assignment)
            normalized_subassignment = _normalize_municipality_name(
                subassignment
            )
            assignment_tokens = _account_label_tokens(assignment)
            subassignment_tokens = _account_label_tokens(subassignment)
            shorter_token_count = min(
                len(assignment_tokens),
                len(subassignment_tokens),
            )
            overlap = len(assignment_tokens & subassignment_tokens)
            labels_are_near_duplicates = (
                bool(shorter_token_count)
                and overlap / shorter_token_count >= 0.7
            )
            account_name = assignment

            if (
                normalized_subassignment not in normalized_assignment
                and not labels_are_near_duplicates
            ):
                account_name = f"{assignment} — {subassignment}"
        else:
            account_name = subassignment or assignment or item or subtitle

        account_code = (
            f"{int(subtitle_code):02d}-"
            f"{int(subassignment_code):07d}"
        )
        records.append(
            {
                "Cod Subtítulo": subtitle_code,
                "Cod Subasignación": subassignment_code,
                "codigo_cuenta": account_code,
                "cuenta": account_name or f"Cuenta {account_code}",
            }
        )

    labels = pd.DataFrame.from_records(records)
    duplicated_labels = labels["cuenta"].duplicated(keep=False)
    labels.loc[duplicated_labels, "cuenta"] = (
        labels.loc[duplicated_labels, "cuenta"]
        + " ("
        + labels.loc[duplicated_labels, "codigo_cuenta"]
        + ")"
    )

    return labels


def analyze_other_income_semester_variability(
    presupuesto: pd.DataFrame,
    *,
    municipality: str | int,
    start_year: int | None = None,
    top_n: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descompone ``Otros ingresos`` y mide su variación entre semestres."""
    if top_n < 1:
        raise ValueError("top_n debe ser al menos 1.")

    monthly_columns = [
        f"Percibidopag {month_name}" for month_name in MONTH_NAMES
    ]
    _require_columns(
        presupuesto,
        {
            "Cod Subtítulo",
            "Cod Subasignación",
            *monthly_columns,
        },
        operation="analizar la variabilidad semestral de Otros ingresos",
    )
    municipality_code, municipality_name = resolve_municipality(
        presupuesto,
        municipality,
    )
    coverage = infer_year_coverage(presupuesto)
    selected_years = set(
        coverage.loc[
            coverage["year_status"].eq("complete"),
            "Ejercicio",
        ].astype(int)
    )

    if start_year is not None:
        selected_years = {
            year for year in selected_years if year >= int(start_year)
        }

    if not selected_years:
        raise ValueError(
            "No quedan años completos para analizar Otros ingresos."
        )

    municipal_income = filter_municipality_income(
        presupuesto,
        municipality=municipality_code,
        years=selected_years,
    )
    municipal_income["grupo_ingreso"] = municipal_income.apply(
        get_income_group,
        axis=1,
    )
    other_income = municipal_income.loc[
        municipal_income["grupo_ingreso"].eq("Otros ingresos")
    ].copy()

    if other_income.empty:
        raise ValueError(
            f"{municipality_name} no tiene filas clasificadas como "
            "Otros ingresos en el período solicitado."
        )

    key_columns = ["Cod Subtítulo", "Cod Subasignación"]
    semester_frames = []

    for semester, month_names in (
        (1, MONTH_NAMES[:6]),
        (2, MONTH_NAMES[6:]),
    ):
        semester_data = other_income[
            ["Ejercicio", *key_columns]
        ].copy()
        semester_data["Semestre"] = semester
        semester_data["ingreso_semestral"] = (
            other_income[
                [f"Percibidopag {month_name}" for month_name in month_names]
            ]
            .sum(axis=1, min_count=1)
            .fillna(0)
        )
        semester_frames.append(semester_data)

    semester_income = (
        pd.concat(semester_frames, ignore_index=True)
        .groupby(
            ["Ejercicio", "Semestre", *key_columns],
            observed=True,
            dropna=False,
        )["ingreso_semestral"]
        .sum()
        .reset_index()
    )
    account_keys = semester_income[key_columns].drop_duplicates()
    complete_grid = pd.DataFrame.from_records(
        [
            {
                "Ejercicio": year,
                "Semestre": semester,
                "Cod Subtítulo": account[0],
                "Cod Subasignación": account[1],
            }
            for year in sorted(selected_years)
            for semester in (1, 2)
            for account in account_keys.itertuples(index=False, name=None)
        ]
    )
    semester_income = complete_grid.merge(
        semester_income,
        on=["Ejercicio", "Semestre", *key_columns],
        how="left",
        validate="one_to_one",
    )
    semester_income["ingreso_semestral"] = semester_income[
        "ingreso_semestral"
    ].fillna(0)
    labels = _other_income_account_labels(other_income)
    semester_income = semester_income.merge(
        labels,
        on=key_columns,
        how="left",
        validate="many_to_one",
    )
    semester_income["Periodo"] = (
        semester_income["Ejercicio"].astype(int).astype(str)
        + " S"
        + semester_income["Semestre"].astype(int).astype(str)
    )

    semester_wide = semester_income.pivot_table(
        index=["Ejercicio", *key_columns],
        columns="Semestre",
        values="ingreso_semestral",
        fill_value=0,
    ).reindex(columns=[1, 2], fill_value=0)
    semester_differences = semester_wide[1] - semester_wide[2]
    variability = (
        semester_differences.rename("diferencia_s1_s2")
        .reset_index()
        .assign(
            variacion_abs_s1_s2=lambda data: data[
                "diferencia_s1_s2"
            ].abs()
        )
        .groupby(key_columns, observed=True, dropna=False)
        .agg(
            variacion_promedio_abs_s1_s2=(
                "variacion_abs_s1_s2",
                "mean",
            ),
            diferencia_promedio_s1_s2=("diferencia_s1_s2", "mean"),
            variacion_maxima_abs_s1_s2=(
                "variacion_abs_s1_s2",
                "max",
            ),
        )
        .reset_index()
    )
    account_totals = (
        semester_income.groupby(
            key_columns,
            observed=True,
            dropna=False,
        )["ingreso_semestral"]
        .sum()
        .rename("ingreso_total")
        .reset_index()
    )
    variability = (
        variability.merge(
            account_totals,
            on=key_columns,
            validate="one_to_one",
        )
        .merge(labels, on=key_columns, validate="one_to_one")
        .sort_values(
            [
                "variacion_promedio_abs_s1_s2",
                "ingreso_total",
                "codigo_cuenta",
            ],
            ascending=[False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    variability["ranking_variabilidad"] = (
        variability.index.to_series().add(1).astype(int)
    )
    total_variability = float(
        variability["variacion_promedio_abs_s1_s2"].sum()
    )
    variability["participacion_variabilidad"] = (
        variability["variacion_promedio_abs_s1_s2"]
        .div(total_variability if total_variability else 1)
        .mul(100)
    )
    selected_keys = {
        tuple(account)
        for account in variability.head(top_n)[key_columns].itertuples(
            index=False,
            name=None,
        )
    }
    rank_by_key = {
        tuple(account): int(rank)
        for account, rank in zip(
            variability[key_columns].itertuples(index=False, name=None),
            variability["ranking_variabilidad"],
            strict=True,
        )
    }
    semester_income["ranking_variabilidad"] = [
        rank_by_key[tuple(account)]
        for account in semester_income[key_columns].itertuples(
            index=False,
            name=None,
        )
    ]
    semester_income["componente"] = [
        account_name
        if tuple(account) in selected_keys
        else "Resto de otros ingresos"
        for account, account_name in zip(
            semester_income[key_columns].itertuples(index=False, name=None),
            semester_income["cuenta"],
            strict=True,
        )
    ]
    semester_income["orden_componente"] = semester_income[
        "ranking_variabilidad"
    ].where(
        semester_income["componente"].ne("Resto de otros ingresos"),
        top_n + 1,
    )
    breakdown = (
        semester_income.groupby(
            [
                "Ejercicio",
                "Semestre",
                "Periodo",
                "componente",
                "orden_componente",
            ],
            observed=True,
        )["ingreso_semestral"]
        .sum()
        .reset_index()
        .sort_values(
            ["Ejercicio", "Semestre", "orden_componente"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    totals_by_period = breakdown.groupby(
        ["Ejercicio", "Semestre"],
        observed=True,
    )["ingreso_semestral"].transform("sum")
    breakdown["total_otros_ingresos"] = totals_by_period
    breakdown["participacion_otros_ingresos"] = (
        breakdown["ingreso_semestral"]
        .div(totals_by_period.where(totals_by_period.ne(0)))
        .mul(100)
    )
    breakdown.insert(0, "Nombre Municipio", municipality_name)
    variability["seleccionada"] = variability[
        "ranking_variabilidad"
    ].le(top_n)
    variability.insert(0, "Nombre Municipio", municipality_name)

    return breakdown, variability
