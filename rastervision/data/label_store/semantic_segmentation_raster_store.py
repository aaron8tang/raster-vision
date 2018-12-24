import numpy as np
import rasterio

import rastervision as rv
from rastervision.utils.files import (get_local_path, make_dir, upload_or_copy,
                                      file_exists)
from rastervision.data import ActivateMixin
from rastervision.data.label import SemanticSegmentationLabels
from rastervision.data.label_store import LabelStore
from rastervision.data.label_source import SegmentationClassTransformer


class SemanticSegmentationRasterStore(ActivateMixin, LabelStore):
    """A prediction label store for segmentation raster files.
    """

    def __init__(self, uri, extent, crs_transformer, tmp_dir, class_map=None):
        """Constructor.

        Args:
            uri: (str) URI of GeoTIFF file used for storing predictions as RGB values
            crs_transformer: (CRSTransformer)
            tmp_dir: (str) temp directory to use
            class_map: (ClassMap) with color values used to convert class ids to
                RGB values
        """
        self.uri = uri
        self.extent = extent
        self.crs_transformer = crs_transformer
        self.tmp_dir = tmp_dir
        # Note: can't name this class_transformer due to Python using that attribute
        if class_map:
            self.class_trans = SegmentationClassTransformer(class_map)
        else:
            self.class_trans = None

        self.source = None
        if file_exists(uri):
            self.source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                                               .with_uri(self.uri) \
                                               .build() \
                                               .create_source(self.tmp_dir)

    def _subcomponents_to_activate(self):
        if self.source is not None:
            return [self.source]
        return []

    def _activate(self):
        pass

    def _deactivate(self):
        pass

    def get_labels(self, chip_size=1000):
        """Get all labels.

        Returns:
            SemanticSegmentationLabels with windows of size chip_size covering the
                scene with no overlap.
        """

        def label_fn(window):
            raw_labels = self.source.get_raw_chip(window)
            if self.class_trans:
                labels = self.class_trans.rgb_to_class(raw_labels)
            else:
                labels = np.squeeze(raw_labels)
            return labels

        extent = self.source.get_extent()
        windows = extent.get_windows(chip_size, chip_size)
        return SemanticSegmentationLabels(windows, label_fn)

    def save(self, labels):
        """Save.

        Args:
            labels - (SemanticSegmentationLabels) labels to be saved
        """
        local_path = get_local_path(self.uri, self.tmp_dir)
        make_dir(local_path, use_dirname=True)

        # TODO: this only works if crs_transformer is RasterioCRSTransformer.
        # Need more general way of computing transform for the more general case.
        transform = self.crs_transformer.transform
        crs = self.crs_transformer.get_image_crs()

        band_count = 1
        dtype = np.uint8
        if self.class_trans:
            band_count = 3

        # https://github.com/mapbox/rasterio/blob/master/docs/quickstart.rst
        # https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
        with rasterio.open(
                local_path,
                'w',
                driver='GTiff',
                height=self.extent.ymax,
                width=self.extent.xmax,
                count=band_count,
                dtype=dtype,
                transform=transform,
                crs=crs) as dataset:
            for window in labels.get_windows():
                class_labels = labels.get_label_arr(
                    window, clip_extent=self.extent)
                clipped_window = ((window.ymin,
                                   window.ymin + class_labels.shape[0]),
                                  (window.xmin,
                                   window.xmin + class_labels.shape[1]))
                if self.class_trans:
                    rgb_labels = self.class_trans.class_to_rgb(class_labels)
                    for chan in range(3):
                        dataset.write_band(
                            chan + 1,
                            rgb_labels[:, :, chan],
                            window=clipped_window)
                else:
                    img = class_labels.astype(dtype)
                    dataset.write_band(1, img, window=clipped_window)

        upload_or_copy(local_path, self.uri)

    def empty_labels(self):
        """Returns an empty SemanticSegmentationLabels object."""
        return SemanticSegmentationLabels()
