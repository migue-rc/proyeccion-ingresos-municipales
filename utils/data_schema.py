"""Contrato de datos presupuestarios normalizados al formato 2025."""

CANONICAL_SCHEMA_VERSION = "2025"
NORMALIZATION_VERSION = "2025.2"
CANONICAL_GROUP_COLUMN = "grupo_ingreso_canonico"
INCOME_GROUPS = (
    "Transferencias corrientes",
    "Transferencias de capital",
    "Otros ingresos",
    "IPP",
    "FCM",
)

MONTH_NAMES = (
    "Enero",
    "Febrero",
    "Marzo",
    "Abril",
    "Mayo",
    "Junio",
    "Julio",
    "Agosto",
    "Septiembre",
    "Octubre",
    "Noviembre",
    "Diciembre",
)

IDENTIFIER_COLUMNS = [
    "Ejercicio",
    "Moneda",
    "Región",
    "Cod Municipio",
    "Nombre Municipio",
    "Cod Serv Incorporado",
    "Nombre Serv Incorporado",
    "Cod Subárea",
    "Nombre Subárea",
    "Cod Tipo Cuenta",
    "Nombre Tipo Cuenta",
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

BUDGET_COLUMNS = [
    "Ppto Inicial",
    *[f"Ppto Modif {month}" for month in MONTH_NAMES],
    "Ppto Actualizado",
]
DEVENGADO_COLUMNS = [
    *[f"Devengado {month}" for month in MONTH_NAMES],
    "Devengado Total",
]
PERCIBIDOPAG_COLUMNS = [
    *[f"Percibidopag {month}" for month in MONTH_NAMES],
    "Percibidopag Total",
]
PORPERCIBIR_COLUMNS = [
    *[f"Porpercibir {month}" for month in MONTH_NAMES],
    "Porpercibir Total",
]
VALUE_COLUMNS = [
    *BUDGET_COLUMNS,
    *DEVENGADO_COLUMNS,
    *PERCIBIDOPAG_COLUMNS,
    *PORPERCIBIR_COLUMNS,
]

CANONICAL_COLUMNS = [*IDENTIFIER_COLUMNS, *VALUE_COLUMNS]

CODE_COLUMNS = [
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

TEXT_COLUMNS = [
    column
    for column in IDENTIFIER_COLUMNS
    if column not in CODE_COLUMNS
]

SOURCE_FIELD_MAP = {
    "Ejercicio": "source_ejercicio",
    "Moneda": "source_moneda",
    "Región": "source_region",
    "Cod Municipio": "source_cod_municipio",
    "Nombre Municipio": "source_nombre_municipio",
    "Cod Serv Incorporado": "source_cod_serv_incorporado",
    "Nombre Serv Incorporado": "source_nombre_serv_incorporado",
    "Cod Subárea": "source_cod_subarea",
    "Nombre Subárea": "source_nombre_subarea",
    "Cod Tipo Cuenta": "source_cod_tipo_cuenta",
    "Nombre Tipo Cuenta": "source_nombre_tipo_cuenta",
    "Cod Subtítulo": "source_cod_subtitulo",
    "Nombre Subtítulo": "source_nombre_subtitulo",
    "Cod Ítem": "source_cod_item",
    "Nombre Ítem": "source_nombre_item",
    "Cod Asignación": "source_cod_asignacion",
    "Nombre Asignación": "source_nombre_asignacion",
    "Cod Subasignación": "source_cod_subasignacion",
    "Nombre Subasignación": "source_nombre_subasignacion",
    "Cod Subsubasignación": "source_cod_subsubasignacion",
    "Nombre Subsubasignación": "source_nombre_subsubasignacion",
}

AUDIT_COLUMNS = [
    "canonical_schema_version",
    "normalization_version",
    "source_schema_version",
    "source_file",
    "source_row",
    *SOURCE_FIELD_MAP.values(),
    "municipality_mapping_status",
    "service_mapping_status",
    CANONICAL_GROUP_COLUMN,
    "income_mapping_status",
    "income_mapping_method",
]
