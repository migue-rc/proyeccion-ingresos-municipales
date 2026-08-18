"""Perfiles comparables y clustering de ingresos municipales."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, leaves_list, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from utils.data_schema import INCOME_GROUPS


@dataclass(frozen=True)
class MunicipalityIncomeProfile:
    """Matrices alineadas de montos, porcentajes y totales por comuna."""

    amounts: pd.DataFrame
    shares: pd.DataFrame
    totals: pd.Series


@dataclass(frozen=True)
class MunicipalityClusterAnalysis:
    """Resultado completo y reutilizable del clustering municipal."""

    profile: MunicipalityIncomeProfile
    clr_features: pd.DataFrame
    linkage_matrix: np.ndarray
    silhouette_scores: pd.DataFrame
    selected_cluster_count: int
    assignments: pd.DataFrame
    summary: pd.DataFrame
    ordered_municipalities: tuple[str, ...]
    pca_projection: pd.DataFrame
    pca_explained_variance_ratio: tuple[float, float]

    @property
    def cluster_validation_scores(self) -> pd.DataFrame:
        """Expone todos los criterios conservando la API silhouette histórica."""
        return self.silhouette_scores


def _require_columns(
    dataframe: pd.DataFrame,
    columns: Collection[str],
) -> None:
    missing = sorted(set(columns).difference(dataframe.columns))

    if missing:
        raise KeyError(
            "Los ingresos no contienen las columnas requeridas: "
            f"{', '.join(missing)}"
        )


def validate_income_group_sequence(
    values: Sequence[str],
    *,
    parameter_name: str,
    require_all: bool,
) -> tuple[str, ...]:
    """Valida listas o tuplas ordenadas de grupos de ingreso."""
    if not isinstance(values, (list, tuple)):
        raise TypeError(
            f"{parameter_name} debe ser una lista o tupla de grupos."
        )

    normalized = tuple(values)

    if not all(isinstance(group, str) for group in normalized):
        raise TypeError(
            f"{parameter_name} solamente puede contener nombres de grupos."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{parameter_name} contiene grupos duplicados.")

    known_groups = set(INCOME_GROUPS)
    unknown = sorted(set(normalized).difference(known_groups))

    if unknown:
        raise ValueError(
            f"{parameter_name} contiene grupos desconocidos: "
            f"{', '.join(unknown)}."
        )

    if require_all:
        missing = [group for group in INCOME_GROUPS if group not in normalized]

        if missing:
            raise ValueError(
                f"{parameter_name} debe incluir todos los grupos. "
                f"Faltan: {', '.join(missing)}."
            )

    return normalized


def build_municipality_income_profile(
    income: pd.DataFrame,
) -> MunicipalityIncomeProfile:
    """
    Agrega ingresos clasificados y calcula su composición por comuna.

    El DataFrame puede contener múltiples filas por comuna y grupo. Los grupos
    ausentes se completan con cero, pero cada comuna debe conservar un total
    estrictamente positivo.
    """
    required_columns = {
        "Nombre Municipio",
        "grupo_ingreso",
        "Percibidopag Total",
    }
    _require_columns(income, required_columns)
    plot_data = income.loc[:, sorted(required_columns)].copy()

    if plot_data["Nombre Municipio"].isna().any():
        raise ValueError("Existen filas sin nombre de municipio.")
    if plot_data["grupo_ingreso"].isna().any():
        raise ValueError("Existen filas sin grupo de ingreso.")

    plot_data["Nombre Municipio"] = (
        plot_data["Nombre Municipio"].astype(str).str.strip()
    )
    if plot_data["Nombre Municipio"].eq("").any():
        raise ValueError("Existen filas sin nombre de municipio.")

    if not plot_data["grupo_ingreso"].map(
        lambda value: isinstance(value, str)
    ).all():
        raise TypeError("Los grupos de ingreso deben estar expresados como texto.")

    unknown_groups = sorted(
        set(plot_data["grupo_ingreso"].unique()).difference(INCOME_GROUPS)
    )
    if unknown_groups:
        raise ValueError(
            "Existen grupos de ingreso desconocidos: "
            f"{', '.join(unknown_groups)}."
        )

    numeric_amounts = pd.to_numeric(
        plot_data["Percibidopag Total"],
        errors="coerce",
    )
    if numeric_amounts.isna().any():
        raise ValueError("Existen ingresos nulos o no numéricos.")
    plot_data["Percibidopag Total"] = numeric_amounts

    amounts = (
        plot_data.groupby(
            ["Nombre Municipio", "grupo_ingreso"],
            observed=True,
        )["Percibidopag Total"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=INCOME_GROUPS, fill_value=0)
        .sort_index()
    )

    if (amounts < 0).any().any():
        raise ValueError(
            "Los perfiles no admiten grupos con ingresos agregados negativos."
        )

    totals = amounts.sum(axis=1).rename("ingreso_total_clp")
    invalid_totals = totals.loc[totals <= 0]
    if not invalid_totals.empty:
        municipalities = ", ".join(invalid_totals.index.astype(str))
        raise ValueError(
            "Todas las comunas deben tener un ingreso total positivo. "
            f"Revise: {municipalities}."
        )

    shares = amounts.div(totals, axis=0).mul(100)

    return MunicipalityIncomeProfile(
        amounts=amounts,
        shares=shares,
        totals=totals,
    )


def ensure_municipality_income_profile(
    income: pd.DataFrame | MunicipalityIncomeProfile,
) -> MunicipalityIncomeProfile:
    """Acepta ingresos crudos o un perfil ya agregado."""
    if isinstance(income, MunicipalityIncomeProfile):
        return income
    if not isinstance(income, pd.DataFrame):
        raise TypeError(
            "income debe ser un DataFrame o MunicipalityIncomeProfile."
        )

    return build_municipality_income_profile(income)


def sort_municipalities_by_share(
    profile: MunicipalityIncomeProfile,
    *,
    sort_priority: Sequence[str] | None = None,
    ascending: bool = False,
) -> list[str]:
    """Ordena comunas por porcentajes y usa el nombre como desempate."""
    normalized_priority = (
        ()
        if sort_priority is None
        else validate_income_group_sequence(
            sort_priority,
            parameter_name="sort_priority",
            require_all=False,
        )
    )

    if not normalized_priority:
        return sorted(profile.shares.index.astype(str))

    sort_data = profile.shares.reset_index()

    return (
        sort_data.sort_values(
            [*normalized_priority, "Nombre Municipio"],
            ascending=[ascending] * len(normalized_priority) + [True],
            kind="stable",
        )["Nombre Municipio"]
        .astype(str)
        .tolist()
    )


def _clr_transform(
    shares: pd.DataFrame,
    *,
    zero_replacement: float,
) -> pd.DataFrame:
    if not 0 < zero_replacement < 1:
        raise ValueError("zero_replacement debe estar entre 0 y 1.")

    compositions = shares.astype(float).div(100)
    adjusted = compositions.mask(compositions <= 0, zero_replacement)
    adjusted = adjusted.div(adjusted.sum(axis=1), axis=0)
    log_values = np.log(adjusted)
    clr_values = log_values.sub(log_values.mean(axis=1), axis=0)

    return pd.DataFrame(
        clr_values,
        index=shares.index.copy(),
        columns=shares.columns.copy(),
    )


def _normalize_cluster_counts(
    cluster_counts: Collection[int],
    *,
    sample_count: int,
) -> tuple[int, ...]:
    normalized = tuple(sorted({int(value) for value in cluster_counts}))

    if not normalized:
        raise ValueError("cluster_counts no puede estar vacío.")
    if normalized[0] < 2 or normalized[-1] >= sample_count:
        raise ValueError(
            "Cada cantidad de clústeres debe estar entre 2 y el número de "
            "comunas menos 1."
        )

    return normalized


def _relabel_by_dendrogram_order(
    labels: np.ndarray,
    leaf_order: np.ndarray,
) -> np.ndarray:
    mapping: dict[int, int] = {}

    for sample_position in leaf_order:
        original_label = int(labels[sample_position])
        if original_label not in mapping:
            mapping[original_label] = len(mapping) + 1

    return np.array([mapping[int(label)] for label in labels], dtype=int)


def cluster_municipalities_by_income_profile(
    income: pd.DataFrame | MunicipalityIncomeProfile,
    *,
    cluster_counts: Collection[int] = range(2, 7),
    cluster_count: int | None = None,
    zero_replacement: float = 0.001,
) -> MunicipalityClusterAnalysis:
    """
    Agrupa comunas por composición usando CLR y Ward jerárquico.

    Si ``cluster_count`` se omite, selecciona automáticamente el candidato con
    mayor silhouette promedio; los empates favorecen la solución más simple.
    """
    profile = ensure_municipality_income_profile(income)
    sample_count = len(profile.shares)

    if sample_count < 3:
        raise ValueError("Se requieren al menos tres comunas para agrupar.")

    candidate_counts = _normalize_cluster_counts(
        cluster_counts,
        sample_count=sample_count,
    )
    if cluster_count is not None:
        requested_count = int(cluster_count)
        if not 2 <= requested_count < sample_count:
            raise ValueError(
                "cluster_count debe estar entre 2 y el número de comunas "
                "menos 1."
            )
        candidate_counts = tuple(sorted({*candidate_counts, requested_count}))

    clr_features = _clr_transform(
        profile.shares,
        zero_replacement=zero_replacement,
    )
    feature_values = clr_features.to_numpy(dtype=float)
    linkage_matrix = linkage(
        feature_values,
        method="ward",
        metric="euclidean",
        optimal_ordering=True,
    )
    score_records = []
    labels_by_count: dict[int, np.ndarray] = {}

    for candidate_count in candidate_counts:
        labels = cut_tree(
            linkage_matrix,
            n_clusters=[candidate_count],
        ).reshape(-1)
        labels_by_count[candidate_count] = labels
        score_records.append(
            {
                "Número de clústeres": candidate_count,
                "Silhouette": silhouette_score(
                    feature_values,
                    labels,
                    metric="euclidean",
                ),
                "Calinski-Harabasz": calinski_harabasz_score(
                    feature_values,
                    labels,
                ),
                "Davies-Bouldin": davies_bouldin_score(
                    feature_values,
                    labels,
                ),
            }
        )

    silhouette_scores = pd.DataFrame(score_records).sort_values(
        "Número de clústeres",
        ignore_index=True,
    )
    if cluster_count is None:
        selected_cluster_count = int(
            silhouette_scores.sort_values(
                ["Silhouette", "Número de clústeres"],
                ascending=[False, True],
            ).iloc[0]["Número de clústeres"]
        )
    else:
        selected_cluster_count = int(cluster_count)

    leaf_order = leaves_list(linkage_matrix)
    selected_labels = _relabel_by_dendrogram_order(
        labels_by_count[selected_cluster_count],
        leaf_order,
    )
    selected_silhouettes = silhouette_samples(
        feature_values,
        selected_labels,
        metric="euclidean",
    )
    municipality_names = profile.shares.index.astype(str).tolist()
    pca = PCA(n_components=2)
    pca_values = pca.fit_transform(feature_values)
    pca_projection = pd.DataFrame(
        pca_values,
        index=profile.shares.index.copy(),
        columns=["CP1", "CP2"],
    )
    assignments = pd.DataFrame(
        {
            "Nombre Municipio": municipality_names,
            "cluster_id": selected_labels,
            "Clúster": [
                f"Clúster {cluster_id}" for cluster_id in selected_labels
            ],
            "Ingreso total (mil MM CLP)": (
                profile.totals.to_numpy(dtype=float) / 1_000_000_000
            ),
            "Silhouette": selected_silhouettes,
            "CP1": pca_projection["CP1"].to_numpy(dtype=float),
            "CP2": pca_projection["CP2"].to_numpy(dtype=float),
        }
    )
    shares_with_clusters = profile.shares.copy()
    shares_with_clusters["cluster_id"] = selected_labels
    totals_with_clusters = profile.totals.to_frame()
    totals_with_clusters["cluster_id"] = selected_labels
    summary_shares = shares_with_clusters.groupby("cluster_id")[
        list(INCOME_GROUPS)
    ].mean()
    cluster_sizes = assignments.groupby("cluster_id").size()
    median_totals = (
        totals_with_clusters.groupby("cluster_id")["ingreso_total_clp"]
        .median()
        .div(1_000_000_000)
    )
    summary = summary_shares.reset_index()
    summary.insert(
        1,
        "Clúster",
        summary["cluster_id"].map(lambda value: f"Clúster {int(value)}"),
    )
    summary.insert(
        2,
        "Número de comunas",
        summary["cluster_id"].map(cluster_sizes).astype(int),
    )
    summary.insert(
        3,
        "Ingreso total mediano (mil MM CLP)",
        summary["cluster_id"].map(median_totals),
    )
    ordered_municipalities = tuple(
        municipality_names[position] for position in leaf_order
    )

    return MunicipalityClusterAnalysis(
        profile=profile,
        clr_features=clr_features,
        linkage_matrix=linkage_matrix,
        silhouette_scores=silhouette_scores,
        selected_cluster_count=selected_cluster_count,
        assignments=assignments,
        summary=summary,
        ordered_municipalities=ordered_municipalities,
        pca_projection=pca_projection,
        pca_explained_variance_ratio=tuple(
            float(value) for value in pca.explained_variance_ratio_
        ),
    )


def export_cluster_membership_csv(
    analysis: MunicipalityClusterAnalysis,
    output_path: str | Path = "clusters.csv",
) -> pd.DataFrame:
    """Guarda una matriz booleana de pertenencia comunal a cada clúster."""
    assignments = analysis.assignments.loc[
        :, ["Nombre Municipio", "cluster_id"]
    ]
    cluster_ids = range(1, analysis.selected_cluster_count + 1)
    membership = pd.DataFrame(
        {
            "comuna": assignments["Nombre Municipio"].astype(str),
            **{
                f"cluster{cluster_id}": assignments["cluster_id"].eq(
                    cluster_id
                )
                for cluster_id in cluster_ids
            },
        }
    )
    membership.to_csv(output_path, index=False)

    return membership


def find_similar_municipalities(
    analysis: MunicipalityClusterAnalysis,
    municipality: str,
    *,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Busca las comunas más cercanas usando distancia euclidiana en CLR."""
    municipality_name = str(municipality).strip()
    names = analysis.clr_features.index.astype(str)
    normalized_names = pd.Series(
        names.str.strip().str.casefold(),
        index=names,
    )
    matches = normalized_names.loc[
        normalized_names.eq(municipality_name.casefold())
    ]

    if matches.empty:
        raise ValueError(f"No existe la comuna solicitada: {municipality}.")
    if len(matches) > 1:
        raise ValueError(
            f"El nombre de comuna es ambiguo: {municipality}."
        )

    sample_count = len(analysis.clr_features)
    requested_neighbors = int(n_neighbors)
    if not 1 <= requested_neighbors < sample_count:
        raise ValueError(
            "n_neighbors debe estar entre 1 y el número de comunas menos 1."
        )

    target_name = str(matches.index[0])
    target_position = names.tolist().index(target_name)
    model = NearestNeighbors(
        n_neighbors=requested_neighbors,
        metric="euclidean",
    ).fit(analysis.clr_features.to_numpy(dtype=float))
    distances, indices = model.kneighbors()
    neighbor_positions = indices[target_position]
    neighbor_distances = distances[target_position]
    assignment_details = analysis.assignments.set_index("Nombre Municipio")
    result = assignment_details.loc[
        names[neighbor_positions],
        ["cluster_id", "Clúster", "Silhouette"],
    ].reset_index()
    result.insert(
        1,
        "Ranking de similitud",
        np.arange(1, requested_neighbors + 1, dtype=int),
    )
    result.insert(2, "Distancia CLR", neighbor_distances)
    result.insert(0, "Comuna de referencia", target_name)

    return result
