import os
from pathlib import Path

import pandas as pd

DATA_ENV = "DATA_FOLDER"


# Nombres canónicos basados en los archivos más recientes.
#
# La clave es el nombre antiguo.
# El valor es el nombre normalizado que queremos utilizar.
COLUMN_ALIASES = {
    # Región
    "Region": "Región",
    # Servicio incorporado
    "Cod Servicio": "Cod Serv Incorporado",
    "Nombre Servicio": "Nombre Serv Incorporado",
    # Subárea
    "Cod Subarea": "Cod Subárea",
    "Nombre Subarea": "Nombre Subárea",
    # Jerarquía presupuestaria
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
}


def load_presupuesto_normalized(path_file: str | Path) -> pd.DataFrame:
    """
    Carga un archivo presupuestario Excel y normaliza su estructura
    al formato utilizado por los archivos más recientes.

    La normalización incluye:
    - nombres históricos de columnas;
    - espacios accidentales en nombres de columnas;
    - nombres históricos de Tipo Cuenta;
    - nombres históricos de Gestión Municipal;
    - incorporación del archivo de origen.

    Parameters
    ----------
    path_file:
        Ruta al archivo .xlsx.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas normalizadas.
    """
    path_file = Path(path_file)

    df = pd.read_excel(path_file)

    # Eliminamos espacios accidentales alrededor de los nombres.
    df.columns = df.columns.str.strip()

    # Renombramos las columnas históricas a la convención más reciente.
    df = df.rename(columns=COLUMN_ALIASES)

    # ------------------------------------------------------------
    # Normalización de valores
    # ------------------------------------------------------------

    if "Nombre Tipo Cuenta" in df.columns:
        # En archivos antiguos encontramos:
        #   Ingreso / Gasto
        #
        # Mientras que en los nuevos:
        #   INGRESOS / GASTOS
        #
        # Dejamos todo usando la convención reciente.
        df["Nombre Tipo Cuenta"] = (
            df["Nombre Tipo Cuenta"]
            .astype("string")
            .str.strip()
            .str.upper()
            .replace(
                {
                    "INGRESO": "INGRESOS",
                    "GASTO": "GASTOS",
                }
            )
        )

    if "Nombre Serv Incorporado" in df.columns:
        # Normalizamos también diferencias históricas de acentos,
        # espacios y capitalización.
        servicio = (
            df["Nombre Serv Incorporado"].astype("string").str.strip().str.upper()
        )

        servicio = servicio.replace(
            {
                "GESTION MUNICIPAL": "GESTIÓN MUNICIPAL",
            }
        )

        df["Nombre Serv Incorporado"] = servicio

    # ------------------------------------------------------------
    # Normalización de códigos
    # ------------------------------------------------------------

    # Usamos Int64 de pandas porque permite almacenar enteros y NA.
    # Esto evita que códigos como 3 se conviertan en 3.0.
    code_columns = [
        "Ejercicio",
        "Cod Municipio",
        "Cod Serv Incorporado",
        "Cod Subárea",
        "Cod Tipo Cuenta",
        "Cod Subtítulo",
        "Cod Ítem",
        "Cod Asignación",
        "Cod Subasignación",
        "Cod Subsubasignación",
    ]

    for column in code_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            ).astype("Int64")

    # Saber de qué archivo vino cada registro es muy útil después
    # para revisar anomalías.
    df["source_file"] = path_file.name

    return df

def data_loader() -> pd.DataFrame:
    """
    Carga y normaliza todos los archivos .xlsx encontrados
    en DATA_FOLDER.

    Si DATA_FOLDER no está definido, utiliza ./data/.

    Returns
    -------
    pd.DataFrame
        Todos los presupuestos concatenados y normalizados.
    """
    data_folder = Path(
        os.getenv(DATA_ENV, "data/")
    )

    if not data_folder.exists():
        raise FileNotFoundError(
            f"La carpeta de datos no existe: {data_folder}"
        )

    # Solo buscamos Excel.
    files = sorted(data_folder.glob("*.xlsx"))

    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos .xlsx en {data_folder}"
        )

    print(
        f"Cargando {len(files)} archivos desde "
        f"{data_folder.resolve()}"
    )

    dataframes = []

    for file in files:
        print(f"  - {file.name}")

        df = load_presupuesto_normalized(file)

        dataframes.append(df)

    # Pandas automáticamente crea NA en las columnas que no existían
    # en determinados años.
    presupuesto = pd.concat(
        dataframes,
        ignore_index=True,
        sort=False,
    )

    return presupuesto
