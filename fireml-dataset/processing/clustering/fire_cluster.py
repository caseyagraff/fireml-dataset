"""
Converts a data frame of fire data to a cluser data frame.
"""
import pyximport

import os
import click
import cPickle as pickle
import pandas as pd
import logging
import numpy as np
from datetime import date

from base.converter import Converter
from helper import df_util as dfu
from helper import date_util as du
import clustering as clust

pyximport.install()


class FireDfToClusterConverter(Converter):
    """
    Converts a data frame of fire data to a cluser data frame.
    """

    def __init__(
        self,
        cluster_id_path=None,
        cluster_type=clust.CLUST_TYPE_SPATIAL_TEMPORAL,
        cluster_thresh_km=5.0,
        cluster_thresh_days=3.0,
        only_day=False,
        fixed_centroids=True,
        fire_season=((5, 14), (8, 31)),
    ):
        super(FireDfToClusterConverter, self).__init__()

        self.cluster_id_path = cluster_id_path
        self.cluster_thresh_km = cluster_thresh_km
        self.cluster_thresh_days = cluster_thresh_days
        self.cluster_type = cluster_type
        self.only_day = only_day
        self.fixed_centroids = fixed_centroids
        self.fire_season = fire_season

    def load(self, src_path):
        logging.info("Loading file from %s" % src_path)
        with open(src_path, "rb") as fin:
            data = pickle.load(fin)  # nosec

        return data

    def save(self, dest_path, data):
        logging.info("Saving data frame to %s" % dest_path)

        with open(dest_path, "wb") as fout:
            pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)

    def transform(self, data):
        logging.debug("Applying transforms to data frame")

        def is_day_det(x):
            local = du.utc_to_local_time(x[0], x[1], du.round_to_nearest_quarter_hour)
            hour = local.hour
            if hour > 9 and hour < 17:
                return 1
            else:
                return 0

        # Add local datetime col to determine the day the fire occured in
        df = data.assign(
            date_local=map(
                lambda x: du.utc_to_local_time(
                    x[0], x[1], du.round_to_nearest_quarter_hour
                ).date(),
                zip(data.datetime_utc, data.lon),
            )
        )
        if self.only_day:
            df = df.assign(
                day_det=map(lambda x: is_day_det, zip(data.datetime_utc, data.lon))
            )

        # Build cluster id data frame
        if not (self.cluster_id_path and os.path.isfile(self.cluster_id_path)):
            logging.debug("Building cluster id data frame")
            fire_season_df = self.filter_fire_season(df)
            cluster_id_df = self.append_cluster_id(fire_season_df)
            if self.cluster_id_path:
                self.save(self.cluster_id_path, cluster_id_df)
        else:
            logging.debug("Loading cluster id data frame")
            cluster_id_df = self.load(self.cluster_id_path)

        cluster_df = self.build_cluster_df(cluster_id_df)

        cluster_df.sort_values("date_local", inplace=True)
        cluster_df.reset_index(drop=True, inplace=True)

        return cluster_df

    def filter_fire_season(self, df):
        logging.debug(
            "Filtering to only include fire season %s" % str(self.fire_season)
        )

        year_range = dfu.get_year_range(df, "datetime_utc")
        year_dfs = []
        for year in range(year_range[0], year_range[1] + 1):
            begin_season = date(year, self.fire_season[0][0], self.fire_season[0][1])
            end_season = date(year, self.fire_season[1][0], self.fire_season[1][1])

            year_dfs.append(
                df[(df.date_local >= begin_season) & (df.date_local <= end_season)]
            )

        return pd.concat(year_dfs)

    def append_cluster_id(self, data):
        """
        Appends cluster id to each row of data frame.
        """
        logging.debug("Appending cluster ids")

        # TODO: Possibly replace with a dictionary for type lookup
        if self.cluster_type == clust.CLUST_TYPE_SPATIAL:
            data = clust.cluster_spatial(data, self.cluster_thresh_km)
        elif self.cluster_type == clust.CLUST_TYPE_SPATIAL_TEMPORAL:
            data = clust.cluster_spatial_temporal(
                data, self.cluster_thresh_km, self.cluster_thresh_days
            )
        elif self.cluster_type == clust.CLUST_TYPE_SPATIAL_TEMPORAL_FORWARDS:
            data = clust.cluster_spatial_temporal_forwards(
                data, self.cluster_thresh_km, self.cluster_thresh_days
            )
        else:
            raise ValueError(
                'Invalid selection for clustering type: "%s"' % self.cluster_type
            )

        return data

    def build_cluster_df(self, df):
        logging.debug("Starting building of final cluster data frame")

        cluster_df_data = []

        cluster_ids = range(int(np.max(df.cluster_id)) + 1)
        for cluster_id in cluster_ids:
            logging.debug(
                "Starting processing of cluster %d/%d"
                % (cluster_id + 1, len(cluster_ids))
            )
            c_df = df[df.cluster_id == cluster_id]

            if self.fixed_centroids:
                lat_centroid, lon_centroid = (
                    np.mean(c_df.lat),
                    np.mean(c_df.lon),
                )

            # dates = set(c_df.date_local)
            dates = du.daterange(
                np.min(c_df.date_local), np.max(c_df.date_local) + du.INC_ONE_DAY,
            )
            for cur_date in dates:
                if self.only_day:
                    date_df = c_df[(c_df.date_local == date) & (c_df.day_det == 1)]
                else:
                    date_df = c_df[(c_df.date_local == cur_date)]
                if not date_df.empty:
                    num_det = len(date_df)
                    avg_frp = np.mean(date_df.FRP)
                    avg_conf = np.mean(date_df.conf)

                    if not self.fixed_centroids:
                        lat_centroid, lon_centroid = (
                            np.mean(date_df.lat),
                            np.mean(date_df.lon),
                        )
                else:
                    num_det, avg_frp, avg_conf = 0, np.nan, np.nan

                cluster_df_data.append(
                    (
                        cur_date,
                        cluster_id,
                        num_det,
                        lat_centroid,
                        lon_centroid,
                        avg_frp,
                        avg_conf,
                    )
                )

        cluster_df = pd.DataFrame(
            cluster_df_data,
            columns=[
                "date_local",
                "cluster_id",
                "num_det",
                "lat_centroid",
                "lon_centroid",
                "avg_frp",
                "avg_conf",
            ],
        )

        return cluster_df


@click.command()
@click.argument("src_path", type=click.Path(exists=True))
@click.argument("dest_path")
@click.option("--cluster", default=None)
@click.option(
    "--cluster_type",
    default=clust.CLUST_TYPE_SPATIAL_TEMPORAL,
    type=click.Choice(
        [
            clust.CLUST_TYPE_SPATIAL,
            clust.CLUST_TYPE_SPATIAL_TEMPORAL,
            clust.CLUST_TYPE_SPATIAL_TEMPORAL_FORWARDS,
        ]
    ),
)
@click.option("--cluster_km", default=5.0, type=click.FLOAT)
@click.option("--cluster_days", default=10, type=click.INT)
@click.option("--daytime", default=False, type=click.BOOL)
@click.option("--fixed_cents", default=True, type=click.BOOL)
@click.option("--log", default="INFO")
def main(
    src_path,
    dest_path,
    cluster,
    cluster_type,
    cluster_km,
    cluster_days,
    daytime,
    fixed_cents,
    log,
):
    """
    Load fire data frame and create clusters.
    """
    log_fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=getattr(logging, log.upper()), format=log_fmt)

    logging.info("Starting fire data frame to cluster conversion")
    logging.info(
        "Cluster Type: %s, Cluster Thresh km: %s, Cluster Thresh days: %s"
        % (cluster_type, cluster_km, cluster_days)
    )
    FireDfToClusterConverter(
        cluster, cluster_type, cluster_km, cluster_days, daytime, fixed_cents
    ).convert(src_path, dest_path)
    logging.info("Finished fire data frame to cluster conversion")


if __name__ == "__main__":
    main()
