"""Carga y unión de límites comunales con resultados municipales."""

from dataclasses import dataclass
from pathlib import Path
import unicodedata

import pandas as pd
import shapefile

from utils.municipal_profiles import MunicipalityIncomeProfile


@dataclass(frozen=True)
class MunicipalityGeography:
    """GeoJSON y catálogo comunal de una región."""

    geojson: dict[str, object]
    municipalities: pd.DataFrame
    region_code: str


def normalize_municipality_name(value: object) -> str:
    """Normaliza tildes, mayúsculas y espacios para unir fuentes distintas."""
    normalized = unicodedata.normalize("NFKD", str(value).strip().casefold())
    without_accents = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )

    return " ".join(without_accents.split())


def load_municipality_geography(
    shapefile_path: str | Path,
    *,
    region_code: str | int,
    expected_count: int | None = None,
) -> MunicipalityGeography:
    """Lee solamente los polígonos comunales de la región solicitada."""
    path = Path(shapefile_path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el shapefile comunal: {path}")

    normalized_region_code = str(region_code).strip().zfill(2)
    features = []
    records = []

    with shapefile.Reader(str(path), encoding="utf-8") as reader:
        for shape_record in reader.iterShapeRecords():
            properties = shape_record.record.as_dict()

            if str(properties["CUT_REG"]).zfill(2) != normalized_region_code:
                continue

            cut_code = str(properties["CUT_COM"])
            municipality_name = str(properties["COMUNA"]).strip()
            features.append(
                {
                    "type": "Feature",
                    "id": cut_code,
                    "properties": properties,
                    "geometry": shape_record.shape.__geo_interface__,
                }
            )
            records.append(
                {
                    "CUT_COM": cut_code,
                    "COMUNA_DPA": municipality_name,
                    "comuna_key": normalize_municipality_name(
                        municipality_name
                    ),
                }
            )

    municipalities = pd.DataFrame.from_records(records)

    if municipalities.empty:
        raise ValueError(
            f"No se encontraron comunas para la región {normalized_region_code}."
        )
    if municipalities["CUT_COM"].duplicated().any():
        raise ValueError("La cartografía contiene códigos comunales duplicados.")
    if municipalities["comuna_key"].duplicated().any():
        raise ValueError("La cartografía contiene nombres comunales duplicados.")
    if expected_count is not None and len(municipalities) != expected_count:
        raise ValueError(
            f"Se esperaban {expected_count} comunas y se encontraron "
            f"{len(municipalities)}."
        )

    return MunicipalityGeography(
        geojson={"type": "FeatureCollection", "features": features},
        municipalities=municipalities,
        region_code=normalized_region_code,
    )


def join_municipality_geography(
    data: pd.DataFrame,
    geography: MunicipalityGeography,
    *,
    municipality_column: str = "Nombre Municipio",
    require_complete: bool = True,
) -> pd.DataFrame:
    """Une resultados únicos por comuna con códigos CUT y valida cobertura."""
    if municipality_column not in data.columns:
        raise KeyError(f"Falta la columna comunal: {municipality_column}")

    result_data = data.copy()

    if result_data[municipality_column].isna().any():
        raise ValueError("Existen resultados sin nombre de comuna.")

    result_data["comuna_key"] = result_data[municipality_column].map(
        normalize_municipality_name
    )
    if result_data["comuna_key"].duplicated().any():
        raise ValueError("Los resultados contienen comunas duplicadas.")

    data_keys = set(result_data["comuna_key"])
    geography_keys = set(geography.municipalities["comuna_key"])

    if require_complete and data_keys != geography_keys:
        raise ValueError(
            "La unión entre resultados y cartografía no es completa. "
            f"Solo en resultados: {sorted(data_keys - geography_keys)}; "
            f"solo en cartografía: {sorted(geography_keys - data_keys)}."
        )
    if not data_keys.issubset(geography_keys):
        raise ValueError(
            "Hay comunas sin cartografía: "
            f"{sorted(data_keys - geography_keys)}."
        )

    return result_data.merge(
        geography.municipalities,
        on="comuna_key",
        how="inner",
        validate="one_to_one",
    )


def build_municipality_income_map_data(
    profile: MunicipalityIncomeProfile,
    geography: MunicipalityGeography,
) -> pd.DataFrame:
    """Prepara métricas monetarias y porcentuales para mapas comunales."""
    data = profile.amounts.copy()
    data.insert(0, "Nombre Municipio", data.index.astype(str))
    data["ingreso_total_clp"] = profile.totals
    data["ingreso_no_ip_clp"] = (
        profile.totals - data["IPP"] - data["FCM"]
    )
    data["ingreso_total_mil_mm_clp"] = (
        data["ingreso_total_clp"] / 1_000_000_000
    )
    data["ingreso_no_ip_mil_mm_clp"] = (
        data["ingreso_no_ip_clp"] / 1_000_000_000
    )
    data["porcentaje_no_ip"] = (
        data["ingreso_no_ip_clp"] / data["ingreso_total_clp"] * 100
    )
    data = data.reset_index(drop=True)

    return join_municipality_geography(data, geography)
