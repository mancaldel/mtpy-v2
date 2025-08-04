# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:58:58 2022

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

from mtpy.imaging.mtplot_tools import PlotBaseMaps

try:
    import contextily as cx
    has_cx = True
except ModuleNotFoundError:
    has_cx = False

# =============================================================================


class PlotPenetrationDepthMap(PlotBaseMaps):
    """Plot the depth of penetration based on the Niblett-Bostick approximation."""

    def __init__(self, mt_data, **kwargs):

        self.mt_data = mt_data

        super().__init__(**kwargs)

        self.depth_units = "km"
        self.plot_period = 1
        self._old_plot_period = None
        self.plot_det = True
        self.plot_te = True
        self.plot_tm = True
        self.plot_stations = True
        self.depth_cmap = "magma"
        self.depth_alpha = 1.0
        self.marker_color = "k"
        self.marker_size = 10
        self.marker_edgecolor = "w"
        self.marker_linewidth = 0.5
        self.subplot_title_dict = {
            "det": "Determinant",
            "xy": "TE Mode",
            "yx": "TM Mode",
        }
        self.depth_range = [None, None]
        self.depth_tolerance = 1

        self.subplot_wspace = 0.2
        self.subplot_hspace = 0.1
        self.font_size = 8

        self.plot_cx = False
        self.cx_source = None
        self.cx_limits = None
        self.cx_zoom = "auto"

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.show_plot:
            self.plot()

    @property
    def depth_units(self):
        """Depth units."""
        return self._depth_units

    @depth_units.setter
    def depth_units(self, value):
        """Depth units."""
        self._depth_units = value
        if value in ["km"]:
            self.depth_scale = 1.0 / 1000
        if value in ["m"]:
            self.depth_scale = 1

    def _get_nb_estimation(self, z_object):
        """Get the depth of investigation estimation."""

        return z_object.estimate_depth_of_investigation()

    def _filter_depth_array(self, depth_array, comp):
        """Filter out some bad data points.
        :param comp:
        :param depth_array: DESCRIPTION.
        :type depth_array: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        depth_array = depth_array[np.nonzero(depth_array)]
        d_median = np.median(depth_array[comp])
        d_min = d_median - (depth_array[comp].std() * self.depth_tolerance)
        d_max = d_median + (depth_array[comp].std() * self.depth_tolerance)
        good_index = np.where(
            (depth_array[comp] >= d_min) & (depth_array[comp] <= d_max)
        )

        return depth_array[good_index]

    def _get_depth_array(self):
        """Get a depth array with xyz values.
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        depth_array = np.zeros(
            len(self.mt_data),
            dtype=[
                ("station", "U20"),
                ("latitude", float),
                ("longitude", float),
                ("elevation", float),
                ("det", float),
                ("xy", float),
                ("yx", float),
            ],
        )

        for ii, tf in enumerate(self.mt_data.values()):

            z_object = tf.Z.interpolate([self.plot_period])
            if (np.nan_to_num(z_object.z) == 0).all():
                continue
            d = self._get_nb_estimation(z_object)

            depth_array["station"][ii] = tf.station
            depth_array["latitude"][ii] = tf.latitude
            depth_array["longitude"][ii] = tf.longitude
            elev = 0
            if tf.elevation is not None:
                depth_array["elevation"][ii] = tf.elevation * self.depth_scale
                elev = tf.elevation * self.depth_scale
            depth_array["det"][ii] = (
                d["depth_det"][0] - elev
            ) * self.depth_scale
            depth_array["xy"][ii] = (
                d["depth_xy"][0] - elev
            ) * self.depth_scale
            depth_array["yx"][ii] = (
                d["depth_yx"][0] - elev
            ) * self.depth_scale

        return depth_array

    def _get_n_subplots(self):
        """Get the number of subplots.
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        n = 0
        if self.plot_det:
            n += 1
        if self.plot_te:
            n += 1
        if self.plot_tm:
            n += 1

        return n

    def _get_subplots(self):
        """Get subplots."""
        n = self._get_n_subplots()
        ax_det = None
        ax_xy = None
        ax_yx = None
        if self.plot_det:
            ax_det = self.fig.add_subplot(1, n, 1, aspect="equal")
            if self.plot_te:
                ax_xy = self.fig.add_subplot(1, n, 2, aspect="equal")
                if self.plot_tm:
                    ax_yx = self.fig.add_subplot(1, n, 3, aspect="equal")
            else:
                if self.plot_tm:
                    ax_yx = self.fig.add_subplot(1, n, 2, aspect="equal")
        else:
            if self.plot_te:
                ax_xy = self.fig.add_subplot(1, n, 1, aspect="equal")
                if self.plot_tm:
                    ax_yx = self.fig.add_subplot(1, n, 2, aspect="equal")
            else:
                if self.plot_tm:
                    ax_yx = self.fig.add_subplot(1, n, 1, aspect="equal")

        return ax_det, ax_xy, ax_yx

    def _get_plot_component_dict(self):
        """Get all the components to plot."""
        ax_det, ax_xy, ax_yx = self._get_subplots()

        components = {}
        if self.plot_det:
            components["det"] = ax_det
        if self.plot_te:
            components["xy"] = ax_xy
        if self.plot_tm:
            components["yx"] = ax_yx

        if len(components.keys()) == 0:
            raise ValueError(
                "must set at least one of the following to True  'plot_det', "
                "'plot_te', 'plot_tm"
            )

        return components

    def plot(self):
        """Plot the depth of investigation as a 1d plot with period on the y-axis
        and depth on the x axis
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        self._set_subplot_params()

        # get data array and make sure there is depth information, if not
        # break
        if self._old_plot_period != self.plot_period:
            depth_array = self._get_depth_array()
            if depth_array.size == 0:
                self.logger.warning(
                    f"No stations have data for period {self.plot_period} "
                )
                return

        self.fig = plt.figure(self.fig_num, self.fig_size, dpi=self.fig_dpi)
        plt.clf()

        plot_components = self._get_plot_component_dict()

        for comp, ax in plot_components.items():

            # Plot background image if specified
            if self.image_file is not None:
                im = plt.imread(self.image_file)
                self.ax.imshow(
                    im, origin="lower", extent=self.image_extent, aspect="auto"
                )
            # plot stations (just to obtain gax for base map)
            gdf = self.mt_data.to_geo_df(model_locations=False)
            gax = gdf.plot(
                ax=ax,
                marker=self.marker,
                color=self.marker_color,
                markersize=0,
            )
            # Plot base map if specified
            if has_cx and self.plot_cx:
                cx_kwargs = {
                    "crs": gdf.crs.to_string(),
                    "source": self.cx_source,
                    "zoom": self.cx_zoom
                }
                if self.cx_limits is not None:
                    ax.set_xlim(self.cx_limits[0], self.cx_limits[1])
                    ax.set_ylim(self.cx_limits[2], self.cx_limits[3])
                cx.add_basemap(
                    gax,
                    **cx_kwargs,
                )

            plot_depth_array = self._filter_depth_array(depth_array, comp)
            if self.interpolation_method in ["nearest", "linear", "cubic"]:
                plot_x, plot_y, image = self.interpolate_to_map(
                    plot_depth_array, comp
                )

                im = ax.pcolormesh(
                    plot_x,
                    plot_y,
                    image,
                    cmap=self.depth_cmap,
                    vmin=self.depth_range[0],
                    vmax=self.depth_range[1],
                    alpha=self.depth_alpha,
                )
            elif self.interpolation_method in [
                "fancy",
                "delaunay",
                "triangulate",
            ]:
                triangulation, image, indices = self.interpolate_to_map(
                    plot_depth_array, comp
                )
                if self.depth_range[0] != None and self.depth_range[1] != None:
                    levels = np.linspace(
                        self.depth_range[0],
                        self.depth_range[1],
                        50,
                    )
                    im = ax.tricontourf(
                        triangulation,
                        image,
                        # mask=indices,
                        levels=levels,
                        extend="both",
                        cmap=self.depth_cmap,
                        alpha=self.depth_alpha,
                    )
                else:
                    im = ax.tricontourf(
                        triangulation,
                        image,
                        levels=50,
                        # mask=indices,
                        extend="both",
                        cmap=self.depth_cmap,
                        alpha=self.depth_alpha,
                    )

            plt.colorbar(
                im,
                ax=ax,
                label=f"Penetration Depth ({self.depth_units})",
                shrink=0.35,
                extend="both",
                extendfrac="auto",
            )

            if self.plot_stations:
                ax.scatter(
                    plot_depth_array["longitude"],
                    plot_depth_array["latitude"],
                    marker=self.marker,
                    s=self.marker_size,
                    c=self.marker_color,
                    edgecolors=self.marker_edgecolor,
                    linewidths=self.marker_linewidth,
                )

            ax.set_xlabel("Longitude (deg)", fontdict=self.font_dict)
            ax.set_ylabel("Latitude (deg)", fontdict=self.font_dict)
            ax.set_title(
                self.subplot_title_dict[comp], fontdict=self.font_dict
            )

        self.fig.suptitle(
            f"Depth of investigation for period {self.plot_period:5g} (s)",
            fontdict=self.font_dict,
        )

        plt.tight_layout()
