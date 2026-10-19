"""Local adapters for explicitly supported mission data products."""

from app.services.data_sources.base import IngestionResult
from app.services.data_sources.alignment import align_observations
from app.services.data_sources.hel1os import read_hel1os_product
from app.services.data_sources.goes_labels import label_observations_from_goes, read_goes_flare_report_csv
from app.services.data_sources.solexs import read_solexs_l1_light_curve

__all__ = ["IngestionResult", "align_observations", "label_observations_from_goes",
           "read_goes_flare_report_csv", "read_hel1os_product", "read_solexs_l1_light_curve"]
