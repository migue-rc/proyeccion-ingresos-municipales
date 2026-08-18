"""Homologación de cuentas de ingreso al agrupamiento analítico moderno."""

from csv import DictReader
from pathlib import Path
import unicodedata

import pandas as pd

from utils.data_schema import CANONICAL_GROUP_COLUMN, INCOME_GROUPS
from utils.ip_keys import FCM_KEYS, IPP_KEYS


MAPPING_STATUS_COLUMN = "income_mapping_status"
MAPPING_METHOD_COLUMN = "income_mapping_method"
CROSSWALK_PATH = (
    Path(__file__).resolve().parent
    / "crosswalks"
    / "legacy_income_groups.csv"
)

LEGACY_IPP_TEXT_PATTERN = (
    r"IMPUESTO TERRITORIAL|PERMISOS? DE CIRCULACION|"
    r"PATENTES? MUNICIPALES|DERECHOS? DE ASEO|"
    r"DERECHOS? (?:VARIOS|MUNICIPALES)|LICENCIAS? DE CONDUCIR|"
    r"RENTA DE INVERSIONES|PATENTES? MINERAS|PATENTES? ACUICOLAS|"
    r"CASINOS? DE JUEGOS?|CONCESIONES?|MULTAS?(?: E| Y)? INTERESES|"
    r"JUZGADO(?: DE)? POLICIA LOCAL|MULTAS? LEY(?: DE)? ALCOHOLES"
)
LEGACY_CAPITAL_TRANSFER_PATTERN = (
    r"GASTOS? DE CAPITAL|INVERSION|INFRAESTRUCT|CONSTRUCC|OBRAS?|"
    r"MEJORAMIENTO (?:DE )?BARRIOS|MEJORAMIENTO URBANO|"
    r"EQUIPAMIENTO|PAVIMENT|PMU|PMB"
)


def _normalized_text(values: pd.Series) -> pd.Series:
    def normalize(value: object) -> str:
        text = unicodedata.normalize(
            "NFKD",
            str(value) if pd.notna(value) else "",
        )

        return "".join(
            character
            for character in text
            if not unicodedata.combining(character)
        ).upper()

    return values.map(normalize).astype("string")


def _account_text(dataframe: pd.DataFrame, mask: pd.Series) -> pd.Series:
    columns = [
        "source_nombre_subtitulo",
        "source_nombre_item",
        "source_nombre_asignacion",
        "source_nombre_subasignacion",
    ]
    text = pd.Series("", index=dataframe.index[mask], dtype="string")

    for column in columns:
        if column in dataframe.columns:
            text = text.str.cat(
                _normalized_text(dataframe.loc[mask, column]),
                sep=" | ",
            )

    return text


def _key_mask(
    dataframe: pd.DataFrame,
    keys: set[tuple[int, int]],
) -> pd.Series:
    subtitle = dataframe["Cod Subtítulo"].fillna(-1).astype("Int64")
    subassignment = dataframe["Cod Subasignación"].fillna(-1).astype("Int64")
    encoded = subtitle.astype("string").str.cat(
        subassignment.astype("string"),
        sep=":",
    )
    expected = {f"{subtitle}:{subassignment}" for subtitle, subassignment in keys}

    return encoded.isin(expected)


def _source_integer(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """Lee un código de auditoría incluso si Excel lo representó como 7.0."""
    return pd.to_numeric(dataframe[column], errors="coerce").astype("Int64")


def _legacy_crosswalk_rules() -> list[dict[str, object]]:
    with CROSSWALK_PATH.open(encoding="utf-8", newline="") as crosswalk_file:
        rows = list(DictReader(crosswalk_file))

    return [
        {
            **row,
            "valid_from": int(row["valid_from"]),
            "valid_to": int(row["valid_to"]),
            "source_cod_subtitulo": int(row["source_cod_subtitulo"]),
            "source_cod_item": int(row["source_cod_item"]),
        }
        for row in rows
    ]


def add_income_mapping(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Agrega grupo, estado y método de homologación a cada fila."""
    result = dataframe.copy()
    result[CANONICAL_GROUP_COLUMN] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )
    result[MAPPING_STATUS_COLUMN] = "not_applicable"
    result[MAPPING_METHOD_COLUMN] = "not_applicable"

    service_name = _normalized_text(result["Nombre Serv Incorporado"])
    scope = (
        result["Cod Tipo Cuenta"].eq(1)
        & (
            result["Cod Serv Incorporado"].eq(1)
            | service_name.eq("GESTION MUNICIPAL")
        )
    )
    modern = scope & result["Ejercicio"].ge(2008)
    legacy = scope & result["Ejercicio"].lt(2008)

    modern_ipp = modern & _key_mask(result, IPP_KEYS)
    modern_fcm = modern & _key_mask(result, FCM_KEYS)
    modern_current = modern & result["Cod Subtítulo"].eq(5)
    modern_capital = modern & result["Cod Subtítulo"].eq(13)

    result.loc[modern, CANONICAL_GROUP_COLUMN] = "Otros ingresos"
    result.loc[modern_current, CANONICAL_GROUP_COLUMN] = (
        "Transferencias corrientes"
    )
    result.loc[modern_capital, CANONICAL_GROUP_COLUMN] = (
        "Transferencias de capital"
    )
    result.loc[modern_ipp, CANONICAL_GROUP_COLUMN] = "IPP"
    result.loc[modern_fcm, CANONICAL_GROUP_COLUMN] = "FCM"
    result.loc[modern, MAPPING_STATUS_COLUMN] = "mapped"
    result.loc[modern, MAPPING_METHOD_COLUMN] = "modern_code"

    if legacy.any():
        legacy_text = _account_text(result, legacy)
        result.loc[legacy, CANONICAL_GROUP_COLUMN] = "Otros ingresos"
        result.loc[legacy, MAPPING_STATUS_COLUMN] = "mapped"
        result.loc[legacy, MAPPING_METHOD_COLUMN] = "legacy_other"

        for rule in _legacy_crosswalk_rules():
            rule_mask = (
                legacy
                & result["Ejercicio"].between(
                    rule["valid_from"],
                    rule["valid_to"],
                )
                & _source_integer(result, "source_cod_subtitulo").eq(
                    rule["source_cod_subtitulo"]
                )
                & _source_integer(result, "source_cod_item").eq(
                    rule["source_cod_item"]
                )
            )
            result.loc[rule_mask, CANONICAL_GROUP_COLUMN] = rule[
                "canonical_group"
            ]
            result.loc[rule_mask, MAPPING_METHOD_COLUMN] = "legacy_crosswalk"

        legacy_fcm_text = legacy_text.str.contains(
            r"FONDO COMUN MUNICIPAL|\bF\.?C\.?M\.\b|COMPENSACION LEY.*19[.]?850",
            regex=True,
            na=False,
        )
        legacy_ipp_text = legacy_text.str.contains(
            LEGACY_IPP_TEXT_PATTERN,
            regex=True,
            na=False,
        )
        legacy_indices = result.index[legacy]
        fcm_indices = legacy_indices[legacy_fcm_text]
        ipp_indices = legacy_indices[legacy_ipp_text & ~legacy_fcm_text]
        exact_rule = result[MAPPING_METHOD_COLUMN].eq("legacy_crosswalk")
        fcm_indices = fcm_indices[~exact_rule.loc[fcm_indices]]
        ipp_indices = ipp_indices[~exact_rule.loc[ipp_indices]]
        result.loc[fcm_indices, CANONICAL_GROUP_COLUMN] = "FCM"
        result.loc[fcm_indices, MAPPING_METHOD_COLUMN] = "legacy_text"
        result.loc[ipp_indices, CANONICAL_GROUP_COLUMN] = "IPP"
        result.loc[ipp_indices, MAPPING_METHOD_COLUMN] = "legacy_text"

        legacy_transfer = legacy & _source_integer(
            result,
            "source_cod_subtitulo",
        ).eq(6)
        transfer_indices = result.index[legacy_transfer]
        transfer_text = _account_text(result, legacy_transfer)
        capital_transfer = transfer_text.str.contains(
            LEGACY_CAPITAL_TRANSFER_PATTERN,
            regex=True,
            na=False,
        )
        capital_indices = transfer_indices[capital_transfer]
        current_indices = transfer_indices[~capital_transfer]
        protected = result[CANONICAL_GROUP_COLUMN].isin(["IPP", "FCM"])
        capital_indices = capital_indices[~protected.loc[capital_indices]]
        current_indices = current_indices[~protected.loc[current_indices]]
        result.loc[capital_indices, CANONICAL_GROUP_COLUMN] = (
            "Transferencias de capital"
        )
        result.loc[capital_indices, MAPPING_STATUS_COLUMN] = "review"
        result.loc[capital_indices, MAPPING_METHOD_COLUMN] = (
            "legacy_transfer_capital_keyword"
        )
        result.loc[current_indices, CANONICAL_GROUP_COLUMN] = (
            "Transferencias corrientes"
        )
        result.loc[current_indices, MAPPING_STATUS_COLUMN] = "review"
        result.loc[current_indices, MAPPING_METHOD_COLUMN] = (
            "legacy_transfer_default_current"
        )

    for column in (MAPPING_STATUS_COLUMN, MAPPING_METHOD_COLUMN):
        result[column] = result[column].astype("string")

    return result
