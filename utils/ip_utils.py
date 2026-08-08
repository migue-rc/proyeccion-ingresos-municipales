import pandas as pd

from utils.ip_keys import FCM_KEYS, IPP_KEYS

def get_income_key(
    row: pd.Series,
) -> tuple[int, int] | None:
    """
    Retorna la clave presupuestaria necesaria para clasificar
    IPP/FCM, solamente para ingresos de Gestión Municipal.
    """

    if row["Cod Tipo Cuenta"] != 1:
        return None

    if row["Nombre Serv Incorporado"] != "GESTIÓN MUNICIPAL":
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
