import pandas as pd

from utils.data_schema import CANONICAL_GROUP_COLUMN, INCOME_GROUPS
from utils.ip_keys import FCM_KEYS, IPP_KEYS


def _is_municipal_income(row: pd.Series) -> bool:
    """Indica si una fila corresponde a ingreso de Gestión Municipal."""
    if row.get("Cod Tipo Cuenta") != 1:
        return False

    service_code = row.get("Cod Serv Incorporado")
    service_name = str(row.get("Nombre Serv Incorporado", "")).strip().casefold()

    return service_code == 1 or service_name in {
        "gestión municipal",
        "gestion municipal",
    }


def get_income_key(
    row: pd.Series,
) -> tuple[int, int] | None:
    """
    Retorna la clave presupuestaria necesaria para clasificar
    IPP/FCM, solamente para ingresos de Gestión Municipal.
    """

    if not _is_municipal_income(row):
        return None

    if (
        pd.isna(row["Cod Subtítulo"])
        or pd.isna(row["Cod Subasignación"])
    ):
        return None

    return (
        int(row["Cod Subtítulo"]),
        int(row["Cod Subasignación"]),
    )


def es_ipp(row: pd.Series) -> bool:
    key = get_income_key(row)

    return key in IPP_KEYS if key is not None else False


def es_fcm(row: pd.Series) -> bool:
    key = get_income_key(row)

    return key in FCM_KEYS if key is not None else False


def get_income_group(row: pd.Series) -> str | None:
    """
    Clasifica un ingreso municipal en un único grupo.

    IPP y FCM tienen prioridad sobre los subtítulos de transferencias
    para evitar que una misma fila pertenezca a más de un grupo.
    """
    canonical_group = row.get(CANONICAL_GROUP_COLUMN)

    if pd.notna(canonical_group) and canonical_group in INCOME_GROUPS:
        return str(canonical_group)

    key = get_income_key(row)

    if key is None:
        return None

    if key in IPP_KEYS:
        return "IPP"

    if key in FCM_KEYS:
        return "FCM"

    cod_subtitulo = key[0]

    if cod_subtitulo == 5:
        return "Transferencias corrientes"

    if cod_subtitulo == 13:
        return "Transferencias de capital"

    return "Otros ingresos"
