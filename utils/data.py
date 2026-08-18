"""Carga presupuestaria normalizada siempre al contrato canónico 2025."""

from collections.abc import Collection
import json
import os
from pathlib import Path
import re
import unicodedata

import pandas as pd

from utils.data_schema import (
    AUDIT_COLUMNS,
    CANONICAL_COLUMNS,
    CANONICAL_SCHEMA_VERSION,
    CODE_COLUMNS,
    IDENTIFIER_COLUMNS,
    MONTH_NAMES,
    NORMALIZATION_VERSION,
    SOURCE_FIELD_MAP,
    TEXT_COLUMNS,
    VALUE_COLUMNS,
)
from utils.income_mapping import (
    CANONICAL_GROUP_COLUMN,
    CROSSWALK_PATH,
    add_income_mapping,
)


DATA_ENV = "DATA_FOLDER"
CACHE_ENV = "NORMALIZED_DATA_FOLDER"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FOLDER = PROJECT_ROOT / "data"
DEFAULT_CACHE_FOLDER = (
    PROJECT_ROOT / "data_normalized" / "schema_2025"
)

# Las claves son nombres encontrados en fuentes históricas y los valores
# corresponden exactamente a la convención del archivo 2025.
COLUMN_ALIASES = {
    "Region": "Región",
    "Cod Servicio": "Cod Serv Incorporado",
    "Nombre Servicio": "Nombre Serv Incorporado",
    "Cod Area": "Cod Subárea",
    "Nombre Area": "Nombre Subárea",
    "Cod Subarea": "Cod Subárea",
    "Nombre Subarea": "Nombre Subárea",
    "Cod Subtitulo": "Cod Subtítulo",
    "Nombre Subtitulo": "Nombre Subtítulo",
    "Cod Item": "Cod Ítem",
    "Nombre Item": "Nombre Ítem",
    "Cod Asignacion": "Cod Asignación",
    "Nombre Asignacion": "Nombre Asignación",
    "Cod Subasignacion": "Cod Subasignación",
    "Nombre Subasignacion": "Nombre Subasignación",
    "Cod Subsubasignacion": "Cod Subsubasignación",
    "Nombre Subsubasignacion": "Nombre Subsubasignación",
    "Ppto inicial": "Ppto Inicial",
    "Ppto actualizado": "Ppto Actualizado",
    "Devengado Acum": "Devengado Total",
    "Devengado acum": "Devengado Total",
    "Percibidopag Acum": "Percibidopag Total",
    "Percibidopag acum": "Percibidopag Total",
    "Porpercibir Acum": "Porpercibir Total",
    "Porpercibir acum": "Porpercibir Total",
    **{
        f"Modif Ppto {month}": f"Ppto Modif {month}"
        for month in MONTH_NAMES
    },
}

SERVICE_ALIASES = {
    "GESTIONMUNICIPAL": (1, "Gestión Municipal"),
    "EDUCACION": (2, "Educación"),
    "AREAEDUCACION": (2, "Educación"),
    "SALUD": (3, "Salud"),
    "AREASALUD": (3, "Salud"),
    "AREADESALUD": (3, "Salud"),
    "CEMENTERIO": (4, "Cementerios"),
    "CEMENTERIOS": (4, "Cementerios"),
    "AREACEMENTERIO": (4, "Cementerios"),
    "AREACEMENTERIOS": (4, "Cementerios"),
}
ACCOUNT_TYPE_NAMES = {
    1: "INGRESOS",
    2: "GASTOS",
}


def _normalized_key(value: object) -> str:
    text = unicodedata.normalize(
        "NFKD",
        str(value).strip().casefold() if pd.notna(value) else "",
    )
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    return re.sub(r"[^a-z0-9]+", "", text).upper()


MUNICIPALITY_NAME_ALIASES = {
    _normalized_key("Llay-Llay"): _normalized_key("Llaillay"),
    _normalized_key("Marchigue"): _normalized_key("Marchihue"),
    _normalized_key("Padre de las Casas"): _normalized_key(
        "Padre Las Casas"
    ),
    _normalized_key("Trehuaco"): _normalized_key("Treguaco"),
    _normalized_key("Qiquen"): _normalized_key("Ñiquén"),
}


def _normalize_integer_codes(
    values: pd.Series,
    *,
    column: str,
) -> pd.Series:
    """Convierte códigos enteros, aceptando miles escritos con punto."""
    normalized = values.astype("string").str.strip()
    thousands_mask = normalized.str.fullmatch(
        r"[+-]?\d{1,3}(?:[.,]\d{3})+",
        na=False,
    )
    normalized.loc[thousands_mask] = normalized.loc[
        thousands_mask
    ].str.replace(r"[.,]", "", regex=True)
    numeric = pd.to_numeric(normalized, errors="coerce")
    non_integer_mask = numeric.notna() & numeric.mod(1).ne(0)

    if non_integer_mask.any():
        examples = values.loc[non_integer_mask].astype(str).unique()[:3]
        raise ValueError(
            f"La columna {column!r} contiene códigos no enteros: "
            f"{examples.tolist()}."
        )

    return numeric.astype("Int64")


def _sum_numeric_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    numeric_values = dataframe.loc[:, columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    return numeric_values.sum(axis=1, min_count=1)


def _fill_annual_totals(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Completa totales ausentes usando la convención de columnas 2025."""
    derived_columns = {
        "Ppto Actualizado": [
            "Ppto Inicial",
            *[f"Ppto Modif {month}" for month in MONTH_NAMES],
        ],
        "Devengado Total": [
            f"Devengado {month}" for month in MONTH_NAMES
        ],
        "Percibidopag Total": [
            f"Percibidopag {month}" for month in MONTH_NAMES
        ],
        "Porpercibir Total": [
            f"Porpercibir {month}" for month in MONTH_NAMES
        ],
    }
    filled_columns: list[str] = []

    for total_column, component_columns in derived_columns.items():
        if total_column in dataframe.columns:
            existing_total = pd.to_numeric(
                dataframe[total_column],
                errors="coerce",
            )
            dataframe[total_column] = existing_total

            if existing_total.notna().all():
                continue
        else:
            existing_total = None

        available_columns = [
            column for column in component_columns if column in dataframe.columns
        ]

        if not available_columns:
            continue

        derived_total = _sum_numeric_columns(dataframe, available_columns)

        if existing_total is not None:
            missing_before = int(existing_total.isna().sum())
            dataframe[total_column] = existing_total.fillna(derived_total)

            if missing_before:
                filled_columns.append(total_column)
        else:
            dataframe[total_column] = derived_total
            filled_columns.append(total_column)

    return dataframe, filled_columns


def _capture_source_fields(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()

    for canonical_column, source_column in SOURCE_FIELD_MAP.items():
        if canonical_column in result.columns:
            result[source_column] = result[canonical_column].astype("string")
        else:
            result[source_column] = pd.Series(
                pd.NA,
                index=result.index,
                dtype="string",
            )

    return result


def _normalize_service(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    service_key = result["Nombre Serv Incorporado"].map(_normalized_key)
    mapped_code = service_key.map(
        {key: value[0] for key, value in SERVICE_ALIASES.items()}
    )
    mapped_name = service_key.map(
        {key: value[1] for key, value in SERVICE_ALIASES.items()}
    )
    mapped = mapped_code.notna()
    result["service_mapping_status"] = pd.Series(
        "unmapped",
        index=result.index,
        dtype="string",
    )
    result.loc[mapped, "Cod Serv Incorporado"] = mapped_code.loc[mapped]
    result.loc[mapped, "Nombre Serv Incorporado"] = mapped_name.loc[mapped]
    result.loc[mapped, "service_mapping_status"] = "mapped_by_name"

    return result


def _normalize_account_type(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Usa los nombres de tipo de cuenta observados en el archivo 2025."""
    result = dataframe.copy()
    result["Nombre Tipo Cuenta"] = result["Nombre Tipo Cuenta"].astype(
        "string"
    )

    for account_code, account_name in ACCOUNT_TYPE_NAMES.items():
        mask = result["Cod Tipo Cuenta"].eq(account_code)
        result.loc[mask, "Nombre Tipo Cuenta"] = account_name

    return result


def _ensure_canonical_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()

    for column in CANONICAL_COLUMNS:
        if column not in result.columns:
            if column in TEXT_COLUMNS:
                result[column] = pd.Series(
                    pd.NA,
                    index=result.index,
                    dtype="string",
                )
            else:
                result[column] = pd.NA

    for column in TEXT_COLUMNS:
        result[column] = result[column].astype("string")

    result["Cod Subárea"] = result["Cod Subárea"].fillna(0)
    missing_subarea_name = (
        result["Nombre Subárea"].isna()
        | result["Nombre Subárea"].astype("string").str.strip().eq("")
    )
    result.loc[missing_subarea_name, "Nombre Subárea"] = "Sin Subárea"

    return result


def _apply_canonical_dtypes(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()

    for column in CODE_COLUMNS:
        result[column] = _normalize_integer_codes(
            result[column],
            column=column,
        )

    for column in TEXT_COLUMNS:
        result[column] = result[column].astype("string").str.strip()

    for column in VALUE_COLUMNS:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).astype("Float64")

    for column in AUDIT_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    string_audit_columns = [
        column for column in AUDIT_COLUMNS if column != "source_row"
    ]

    for column in string_audit_columns:
        result[column] = result[column].astype("string")

    result["source_row"] = _normalize_integer_codes(
        result["source_row"],
        column="source_row",
    )

    return result[[*CANONICAL_COLUMNS, *AUDIT_COLUMNS]]


def _blank_legacy_account_hierarchy(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Evita presentar códigos 2002-2007 como si fueran códigos modernos."""
    result = dataframe.copy()
    legacy = result["Ejercicio"].lt(2008)
    account_columns = [
        "Cod Subtítulo",
        "Nombre Subtítulo",
        "Cod Ítem",
        "Nombre Ítem",
        "Cod Asignación",
        "Nombre Asignación",
        "Cod Subasignación",
        "Nombre Subasignación",
        "Cod Subsubasignación",
        "Nombre Subsubasignación",
    ]
    result.loc[legacy, account_columns] = pd.NA

    return result


def load_presupuesto_normalized(path_file: str | Path) -> pd.DataFrame:
    """
    Carga un Excel y lo adapta siempre al contrato estructural 2025.

    Los códigos y nombres originales se conservan en columnas ``source_*``.
    Para 2002-2007 se homologa el grupo de ingreso, pero los códigos modernos
    detallados quedan nulos porque no existe una equivalencia uno-a-uno.

    Parameters
    ----------
    path_file:
        Ruta al archivo presupuestario Excel.

    Returns
    -------
    pandas.DataFrame
        Archivo normalizado al contrato 2025, todavía sin aplicar el catálogo
        municipal moderno. ``data_loader`` realiza esa etapa al combinar años.

    Examples
    --------
    >>> normalized = load_presupuesto_normalized(
    ...     "data/BD_Presupuestaria_2010.xlsx"
    ... )
    >>> normalized["canonical_schema_version"].unique().tolist()
    ['2025']
    """
    path_file = Path(path_file)
    dataframe = pd.read_excel(
        path_file,
        dtype={"Ejercicio": "string"},
    )
    dataframe.columns = dataframe.columns.astype(str).str.strip()
    dataframe = dataframe.rename(columns=COLUMN_ALIASES)

    unexpected_columns = sorted(
        set(dataframe.columns).difference(CANONICAL_COLUMNS)
    )

    if unexpected_columns:
        raise ValueError(
            f"{path_file.name} contiene columnas sin regla de "
            f"normalización: {unexpected_columns}."
        )

    if dataframe.columns.duplicated().any():
        duplicated = dataframe.columns[dataframe.columns.duplicated()].tolist()
        raise ValueError(
            f"{path_file.name} genera columnas duplicadas al normalizar: "
            f"{duplicated}."
        )

    dataframe = _capture_source_fields(dataframe)
    dataframe["source_file"] = path_file.name
    dataframe["source_row"] = pd.Series(
        range(2, len(dataframe) + 2),
        index=dataframe.index,
        dtype="Int64",
    )
    dataframe = _ensure_canonical_columns(dataframe)

    for column in CODE_COLUMNS:
        dataframe[column] = _normalize_integer_codes(
            dataframe[column],
            column=column,
        )

    years = sorted(dataframe["Ejercicio"].dropna().astype(int).unique())

    if not years:
        raise ValueError(f"{path_file.name} no contiene un ejercicio válido.")

    dataframe["source_schema_version"] = (
        "legacy_pre_2008" if max(years) < 2008 else "modern_2008_plus"
    )
    dataframe["canonical_schema_version"] = CANONICAL_SCHEMA_VERSION
    dataframe["normalization_version"] = NORMALIZATION_VERSION
    dataframe["municipality_mapping_status"] = "source_only"
    dataframe = _normalize_account_type(dataframe)
    dataframe = _normalize_service(dataframe)
    dataframe, derived_columns = _fill_annual_totals(dataframe)
    dataframe = add_income_mapping(dataframe)
    dataframe = _blank_legacy_account_hierarchy(dataframe)
    dataframe = _apply_canonical_dtypes(dataframe)
    dataframe.attrs["derived_total_columns"] = derived_columns

    return dataframe


def build_municipality_catalog(
    canonical_2025: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el catálogo municipal canónico desde el archivo 2025."""
    catalog = (
        canonical_2025.loc[
            canonical_2025["Ejercicio"].eq(2025),
            ["Cod Municipio", "Nombre Municipio", "Región"],
        ]
        .dropna(subset=["Cod Municipio", "Nombre Municipio"])
        .drop_duplicates()
        .copy()
    )

    if catalog.empty:
        raise ValueError(
            "No fue posible construir el catálogo municipal desde 2025."
        )

    catalog["municipality_key"] = catalog["Nombre Municipio"].map(
        _normalized_key
    )
    conflicts = catalog.groupby("municipality_key")["Cod Municipio"].nunique()

    if conflicts.gt(1).any():
        keys = conflicts.loc[conflicts.gt(1)].index.tolist()
        raise ValueError(
            "El catálogo 2025 contiene nombres municipales ambiguos: "
            f"{keys}."
        )

    return catalog.drop_duplicates("municipality_key").reset_index(drop=True)


def apply_municipality_catalog(
    dataframe: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Reemplaza nombre, código y región por su identidad municipal 2025."""
    result = dataframe.copy()
    source_key = result["source_nombre_municipio"].map(_normalized_key)
    aliased_key = source_key.replace(MUNICIPALITY_NAME_ALIASES)
    catalog_by_key = catalog.set_index("municipality_key")
    canonical_code = aliased_key.map(catalog_by_key["Cod Municipio"])
    canonical_name = aliased_key.map(catalog_by_key["Nombre Municipio"])
    canonical_region = aliased_key.map(catalog_by_key["Región"])
    mapped = canonical_code.notna()
    source_code = pd.to_numeric(
        result["source_cod_municipio"],
        errors="coerce",
    )
    alias_used = source_key.ne(aliased_key)
    code_changed = source_code.ne(canonical_code)
    result["municipality_mapping_status"] = pd.Series(
        "unmapped",
        index=result.index,
        dtype="string",
    )
    result.loc[mapped, "Cod Municipio"] = canonical_code.loc[mapped]
    result.loc[mapped, "Nombre Municipio"] = canonical_name.loc[mapped]
    result.loc[mapped, "Región"] = canonical_region.loc[mapped]
    result.loc[mapped, "municipality_mapping_status"] = "matched_by_name"
    result.loc[mapped & ~code_changed, "municipality_mapping_status"] = (
        "canonical_code"
    )
    result.loc[mapped & alias_used, "municipality_mapping_status"] = (
        "matched_by_alias"
    )

    return _apply_canonical_dtypes(result)


def validate_normalized_file(
    dataframe: pd.DataFrame,
    *,
    strict: bool = True,
) -> dict[str, object]:
    """Valida el contrato 2025 y retorna indicadores de auditoría."""
    missing = sorted(set([*CANONICAL_COLUMNS, *AUDIT_COLUMNS]) - set(dataframe))

    if missing:
        raise ValueError(
            "El dataframe no cumple el contrato 2025; faltan: "
            f"{', '.join(missing)}."
        )

    schema_values = set(
        dataframe["canonical_schema_version"].dropna().astype(str).unique()
    )

    if schema_values != {CANONICAL_SCHEMA_VERSION}:
        raise ValueError(
            "Se encontraron versiones canónicas inesperadas: "
            f"{sorted(schema_values)}."
        )

    income_scope = (
        dataframe["Cod Tipo Cuenta"].eq(1)
        & dataframe["Cod Serv Incorporado"].eq(1)
    )
    missing_income_group = int(
        dataframe.loc[income_scope, CANONICAL_GROUP_COLUMN].isna().sum()
    )
    municipality_unmapped = int(
        dataframe["municipality_mapping_status"].eq("unmapped").sum()
    )
    service_unmapped = int(
        dataframe["service_mapping_status"].eq("unmapped").sum()
    )
    missing_account_type = int(dataframe["Cod Tipo Cuenta"].isna().sum())
    missing_collected_total = int(
        dataframe["Percibidopag Total"].isna().sum()
    )

    if strict and missing_income_group:
        raise ValueError(
            f"Quedan {missing_income_group} ingresos municipales sin grupo."
        )

    if strict and municipality_unmapped:
        examples = (
            dataframe.loc[
                dataframe["municipality_mapping_status"].eq("unmapped"),
                "source_nombre_municipio",
            ]
            .dropna()
            .unique()[:5]
            .tolist()
        )
        raise ValueError(
            f"Quedan {municipality_unmapped} filas sin municipio canónico: "
            f"{examples}."
        )

    if strict and missing_account_type == len(dataframe):
        raise ValueError("El archivo no contiene códigos de tipo de cuenta.")

    if strict and missing_collected_total == len(dataframe):
        raise ValueError(
            "El archivo no contiene valores de Percibidopag mensuales ni "
            "totales."
        )

    years = sorted(dataframe["Ejercicio"].dropna().astype(int).unique())

    return {
        "source_file": str(dataframe["source_file"].iloc[0]),
        "rows": len(dataframe),
        "years": ", ".join(str(year) for year in years),
        "source_schema_version": str(
            dataframe["source_schema_version"].iloc[0]
        ),
        "derived_total_columns": ", ".join(
            dataframe.attrs.get("derived_total_columns", [])
        ),
        "mapped_income_rows": int(
            dataframe.loc[income_scope, CANONICAL_GROUP_COLUMN].notna().sum()
        ),
        "review_income_rows": int(
            dataframe.loc[income_scope, "income_mapping_status"]
            .eq("review")
            .sum()
        ),
        "missing_income_group_rows": missing_income_group,
        "unmapped_municipality_rows": municipality_unmapped,
        "unmapped_service_rows": service_unmapped,
        "missing_account_type_rows": missing_account_type,
        "missing_collected_total_rows": missing_collected_total,
    }


def _source_signature(
    source: Path,
    canonical_source: Path,
) -> dict[str, object]:
    source_stat = source.stat()
    canonical_stat = canonical_source.stat()
    crosswalk_stat = CROSSWALK_PATH.stat()

    return {
        "normalization_version": NORMALIZATION_VERSION,
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "catalog_size": canonical_stat.st_size,
        "catalog_mtime_ns": canonical_stat.st_mtime_ns,
        "crosswalk_size": crosswalk_stat.st_size,
        "crosswalk_mtime_ns": crosswalk_stat.st_mtime_ns,
    }


def _cache_paths(source: Path, cache_folder: Path) -> tuple[Path, Path]:
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", source.stem).strip("-")

    return (
        cache_folder / f"{safe_name}.pkl.gz",
        cache_folder / f"{safe_name}.json",
    )


def _read_cache(
    source: Path,
    cache_folder: Path,
    canonical_source: Path,
) -> pd.DataFrame | None:
    cache_path, manifest_path = _cache_paths(source, cache_folder)

    if not cache_path.exists() or not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if manifest != _source_signature(source, canonical_source):
            return None

        return pd.read_pickle(cache_path, compression="gzip")
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _write_cache(
    dataframe: pd.DataFrame,
    source: Path,
    cache_folder: Path,
    canonical_source: Path,
) -> None:
    cache_folder.mkdir(parents=True, exist_ok=True)
    cache_path, manifest_path = _cache_paths(source, cache_folder)
    temporary_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    dataframe.to_pickle(
        temporary_cache,
        compression={"method": "gzip", "compresslevel": 1},
    )
    temporary_manifest.write_text(
        json.dumps(
            _source_signature(source, canonical_source),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_cache.replace(cache_path)
    temporary_manifest.replace(manifest_path)


def _file_year_from_name(path: Path) -> int | None:
    match = re.search(r"(?<!\d)(20\d{2})(?!\d)", path.name)

    return int(match.group(1)) if match else None


def _filter_municipality(
    dataframe: pd.DataFrame,
    municipality: str | int | None,
) -> pd.DataFrame:
    if municipality is None:
        return dataframe

    if isinstance(municipality, str):
        requested = _normalized_key(municipality)
        mask = dataframe["Nombre Municipio"].map(_normalized_key).eq(requested)
    elif isinstance(municipality, int) and not isinstance(municipality, bool):
        mask = dataframe["Cod Municipio"].eq(municipality)
    else:
        raise TypeError("municipality debe ser nombre, código entero o None.")

    result = dataframe.loc[mask].copy()

    if result.empty:
        raise ValueError(f"No se encontró la comuna {municipality!r}.")

    return result


def _normalize_municipality_collection(
    municipalities: Collection[str | int] | None,
) -> tuple[str | int, ...] | None:
    if municipalities is None:
        return None
    if isinstance(municipalities, (str, bytes)) or not isinstance(
        municipalities,
        Collection,
    ):
        raise TypeError(
            "municipalities debe ser una colección de nombres o códigos."
        )

    normalized: list[str | int] = []
    seen: set[tuple[str, str | int]] = set()

    for municipality in municipalities:
        if isinstance(municipality, str):
            normalized_name = _normalized_key(municipality)
            if not normalized_name:
                raise ValueError(
                    "Los nombres incluidos en municipalities no pueden "
                    "estar vacíos."
                )
            key: tuple[str, str | int] = ("name", normalized_name)
        elif isinstance(municipality, int) and not isinstance(
            municipality,
            bool,
        ):
            municipality = int(municipality)
            key = ("code", municipality)
        else:
            raise TypeError(
                "municipalities solamente puede contener nombres o códigos "
                "enteros."
            )

        if key not in seen:
            normalized.append(municipality)
            seen.add(key)

    if not normalized:
        raise ValueError("municipalities no puede estar vacío.")

    return tuple(normalized)


def _filter_municipalities(
    dataframe: pd.DataFrame,
    municipalities: Collection[str | int],
    *,
    require_all: bool = False,
) -> pd.DataFrame:
    requested_names = {
        _normalized_key(value)
        for value in municipalities
        if isinstance(value, str)
    }
    requested_codes = {
        int(value)
        for value in municipalities
        if isinstance(value, int) and not isinstance(value, bool)
    }
    normalized_names = dataframe["Nombre Municipio"].map(_normalized_key)
    mask = normalized_names.isin(requested_names)
    mask |= dataframe["Cod Municipio"].isin(requested_codes)
    result = dataframe.loc[mask].copy()

    if result.empty:
        raise ValueError("No se encontró ninguna de las comunas solicitadas.")

    if require_all:
        found_names = set(normalized_names.loc[mask])
        found_codes = set(
            dataframe.loc[mask, "Cod Municipio"].dropna().astype(int)
        )
        missing = [
            value
            for value in municipalities
            if (
                isinstance(value, str)
                and _normalized_key(value) not in found_names
            )
            or (
                isinstance(value, int)
                and not isinstance(value, bool)
                and int(value) not in found_codes
            )
        ]
        if missing:
            missing_labels = ", ".join(repr(value) for value in missing)
            raise ValueError(
                "No se encontraron las comunas solicitadas: "
                f"{missing_labels}."
            )

    return result


def get_normalization_report(presupuesto: pd.DataFrame) -> pd.DataFrame:
    """
    Recupera el reporte producido por ``data_loader``.

    Examples
    --------
    >>> presupuesto = data_loader(municipality="Renca")
    >>> report = get_normalization_report(presupuesto)
    >>> report[["source_file", "years", "review_income_rows"]]
    """
    records = presupuesto.attrs.get("normalization_report", [])

    return pd.DataFrame.from_records(records)


def data_loader(
    *,
    data_folder: str | Path | None = None,
    years: Collection[int] | None = None,
    municipality: str | int | None = None,
    municipalities: Collection[str | int] | None = None,
    refresh: bool = False,
    validate: bool = True,
    use_cache: bool = True,
    cache_folder: str | Path | None = None,
) -> pd.DataFrame:
    """
    Carga todos los presupuestos usando siempre el contrato canónico 2025.

    No existe un selector de esquema: cualquier archivo histórico se adapta a
    2025 antes de concatenarse. El caché local se invalida cuando cambia el
    Excel, el catálogo 2025, el crosswalk o la versión de normalización.

    Parameters
    ----------
    data_folder:
        Carpeta de Excel. Por defecto usa ``DATA_FOLDER`` o ``./data``.
    years:
        Ejercicios opcionales que deben cargarse.
    municipality:
        Nombre o código canónico opcional para reducir el resultado.
    municipalities:
        Colección opcional de nombres o códigos canónicos. No puede combinarse
        con ``municipality``.
    refresh:
        Si es ``True``, reconstruye el caché desde los Excel.
    validate:
        Ejecuta validaciones estrictas del contrato y de las homologaciones.
    use_cache:
        Activa el caché local confiable en formato pickle comprimido.
    cache_folder:
        Ubicación alternativa del caché.

    Returns
    -------
    pandas.DataFrame
        Presupuestos normalizados al esquema 2025.

    Examples
    --------
    Cargar todo:

    >>> presupuesto = data_loader()

    Cargar solamente Renca y un rango de años:

    >>> presupuesto_renca = data_loader(
    ...     municipality="Renca",
    ...     years=range(2008, 2026),
    ... )

    Cargar varias comunas en una sola pasada:

    >>> presupuesto_cluster = data_loader(
    ...     municipalities=["Buin", "Renca", "Ñuñoa"],
    ... )

    Forzar una reconstrucción tras cambiar reglas:

    >>> presupuesto = data_loader(refresh=True)
    """
    if municipality is not None and municipalities is not None:
        raise ValueError(
            "municipality y municipalities son mutuamente excluyentes."
        )
    normalized_municipalities = _normalize_municipality_collection(
        municipalities
    )
    resolved_data_folder = Path(
        data_folder
        or os.getenv(DATA_ENV, str(DEFAULT_DATA_FOLDER))
    )
    resolved_cache_folder = Path(
        cache_folder
        or os.getenv(CACHE_ENV, str(DEFAULT_CACHE_FOLDER))
    )

    if not resolved_data_folder.exists():
        raise FileNotFoundError(
            f"La carpeta de datos no existe: {resolved_data_folder}"
        )

    files = sorted(resolved_data_folder.glob("*.xlsx"))

    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos .xlsx en {resolved_data_folder}"
        )

    canonical_candidates = [
        file for file in files if _file_year_from_name(file) == 2025
    ]

    if len(canonical_candidates) != 1:
        raise ValueError(
            "Se necesita exactamente un archivo fuente del ejercicio 2025; "
            f"se encontraron {len(canonical_candidates)}."
        )

    canonical_source = canonical_candidates[0]
    canonical_2025 = None if refresh or not use_cache else _read_cache(
        canonical_source,
        resolved_cache_folder,
        canonical_source,
    )
    canonical_cache_hit = canonical_2025 is not None

    if canonical_2025 is None:
        canonical_2025 = load_presupuesto_normalized(canonical_source)
        catalog = build_municipality_catalog(canonical_2025)
        canonical_2025 = apply_municipality_catalog(canonical_2025, catalog)

        if use_cache:
            _write_cache(
                canonical_2025,
                canonical_source,
                resolved_cache_folder,
                canonical_source,
            )
    else:
        catalog = build_municipality_catalog(canonical_2025)

    if normalized_municipalities is not None:
        _filter_municipalities(
            canonical_2025,
            normalized_municipalities,
            require_all=True,
        )

    requested_years = (
        {int(year) for year in years} if years is not None else None
    )
    dataframes: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []
    print(
        f"Normalizando {len(files)} archivos al contrato 2025 desde "
        f"{resolved_data_folder.resolve()}"
    )

    for file in files:
        file_year = _file_year_from_name(file)

        if requested_years is not None and file_year not in requested_years:
            continue

        cached = False

        if file == canonical_source:
            normalized = canonical_2025
            cached = canonical_cache_hit
        else:
            normalized = None if refresh or not use_cache else _read_cache(
                file,
                resolved_cache_folder,
                canonical_source,
            )
            cached = normalized is not None

            if normalized is None:
                normalized = load_presupuesto_normalized(file)
                normalized = apply_municipality_catalog(normalized, catalog)

                if use_cache:
                    _write_cache(
                        normalized,
                        file,
                        resolved_cache_folder,
                        canonical_source,
                    )

        report = validate_normalized_file(
            normalized,
            strict=validate,
        )
        report["cache_status"] = "hit" if cached else "rebuilt"
        reports.append(report)
        try:
            if normalized_municipalities is None:
                selected = _filter_municipality(normalized, municipality)
            else:
                selected = _filter_municipalities(
                    normalized,
                    normalized_municipalities,
                )
        except ValueError:
            if municipality is None and normalized_municipalities is None:
                raise

            requested = (
                municipality
                if normalized_municipalities is None
                else normalized_municipalities
            )
            print(
                f"  - {file.name}: comuna(s) {requested!r} ausente(s); "
                "archivo omitido"
            )
            continue

        dataframes.append(selected)
        print(
            f"  - {file.name}: {len(selected):,} filas "
            f"({report['years']}; {report['cache_status']})"
        )

    if not dataframes:
        raise ValueError("No hay archivos que coincidan con los años solicitados.")

    presupuesto = pd.concat(dataframes, ignore_index=True, sort=False)
    presupuesto = _apply_canonical_dtypes(presupuesto)
    presupuesto.attrs["normalization_report"] = reports
    presupuesto.attrs["canonical_schema_version"] = CANONICAL_SCHEMA_VERSION
    loaded_years = sorted(
        presupuesto["Ejercicio"].dropna().astype(int).unique()
    )
    print(
        f"Carga completa: {len(presupuesto):,} filas | "
        f"ejercicios {loaded_years} | esquema 2025"
    )

    if use_cache:
        resolved_cache_folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(reports).to_csv(
            resolved_cache_folder / "normalization_report.csv",
            index=False,
        )

    return presupuesto


def build_presupuesto_monthly(
    presupuesto: pd.DataFrame,
    *,
    drop_empty: bool = True,
) -> pd.DataFrame:
    """
    Convierte un presupuesto canónico ancho a una tabla mensual analítica.

    Se recomienda filtrar una comuna antes de expandir, pues cada fila original
    produce doce filas mensuales.

    Examples
    --------
    >>> renca = data_loader(municipality="Renca")
    >>> renca_monthly = build_presupuesto_monthly(renca)
    >>> renca_monthly[["fecha", "Percibidopag"]].head()
    """
    missing = sorted(set(CANONICAL_COLUMNS) - set(presupuesto.columns))

    if missing:
        raise ValueError(
            "El dataframe no cumple el contrato 2025; faltan: "
            f"{', '.join(missing)}."
        )

    identifier_columns = [
        *IDENTIFIER_COLUMNS,
        CANONICAL_GROUP_COLUMN,
        "income_mapping_status",
        "income_mapping_method",
        "source_file",
        "source_row",
    ]
    monthly_frames = []

    for month_number, month_name in enumerate(MONTH_NAMES, start=1):
        month = presupuesto.loc[:, identifier_columns].copy()
        month["mes_numero"] = month_number
        month["mes"] = month_name
        month["fecha"] = pd.to_datetime(
            {
                "year": presupuesto["Ejercicio"].astype(int),
                "month": month_number,
                "day": 1,
            }
        )

        for measure in (
            "Ppto Modif",
            "Devengado",
            "Percibidopag",
            "Porpercibir",
        ):
            month[measure] = presupuesto[f"{measure} {month_name}"].array

        if drop_empty:
            measure_columns = [
                "Ppto Modif",
                "Devengado",
                "Percibidopag",
                "Porpercibir",
            ]
            month = month.loc[month[measure_columns].notna().any(axis=1)]

        monthly_frames.append(month)

    result = pd.concat(monthly_frames, ignore_index=True)
    result["mes_numero"] = result["mes_numero"].astype("Int64")
    result["mes"] = result["mes"].astype("string")

    return result
