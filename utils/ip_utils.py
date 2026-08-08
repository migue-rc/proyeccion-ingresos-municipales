import pandas as pd

from utils.ip_keys import FCM_KEYS, IPP_KEYS


def get_income_key(row: pd.Series) -> tuple[int, int] | None:
    """
    Obtiene la clave (Cod Subtítulo, Cod Subasignación)
    para ingresos correspondientes a Gestión Municipal.

    Compatible con las bases presupuestarias analizadas
    entre 2013 y 2025.
    """

    # Cod Tipo Cuenta = 1 corresponde a ingresos.
    # Utilizar el código evita diferencias históricas como
    # "Ingreso" vs "INGRESOS".
    try:
        if int(row["Cod Tipo Cuenta"]) != 1:
            return None
    except KeyError, TypeError, ValueError:
        return None

    # El nombre de la columna de servicio cambia según el año.
    if "Nombre Serv Incorporado" in row.index:
        servicio = row["Nombre Serv Incorporado"]
    else:
        servicio = row.get("Nombre Servicio")

    # Normalizamos para tolerar espacios y diferencias de mayúsculas.
    servicio = str(servicio).strip().upper()

    if servicio not in {
        "GESTION MUNICIPAL",
        "GESTIÓN MUNICIPAL",
    }:
        return None

    # Las columnas de códigos también cambiaron de nombre.
    if "Cod Subtítulo" in row.index:
        cod_subtitulo = row["Cod Subtítulo"]
    else:
        cod_subtitulo = row.get("Cod Subtitulo")

    if "Cod Subasignación" in row.index:
        cod_subasignacion = row["Cod Subasignación"]
    else:
        cod_subasignacion = row.get("Cod Subasignacion")

    # Una fila sin ambos códigos no puede clasificarse
    # mediante nuestra clave presupuestaria.
    if pd.isna(cod_subtitulo) or pd.isna(cod_subasignacion):
        return None

    return (
        int(cod_subtitulo),
        int(cod_subasignacion),
    )


def es_ipp(row: pd.Series) -> bool:
    """Indica si una fila corresponde a un IPP."""
    key = get_income_key(row)

    return key in IPP_KEYS if key is not None else False


def es_fcm(row: pd.Series) -> bool:
    """Indica si una fila corresponde a un ingreso del FCM."""
    key = get_income_key(row)

    return key in FCM_KEYS if key is not None else False
