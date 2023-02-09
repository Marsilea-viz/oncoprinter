import warnings
from collections import Counter
from itertools import count
from copy import deepcopy

import numpy as np
import pandas as pd

from heatgraphy import ClusterBoard, Heatmap
from heatgraphy.layers import LayersMesh, FrameRect
from heatgraphy.plotter import Labels, StackBar
from heatgraphy.utils import get_canvas_size_by_data
from .preset import SHAPE_BANK, MATCH_POOL, Alteration


def guess_alteration(event: str):
    for alt, rule in MATCH_POOL.items():
        if rule.is_match(event):
            return alt
    return Alteration.OTHER


class GenomicData:
    """Handle class for genomics data

    Parameters
    ----------
    data : pd.DataFrame
        Each column is:
            1) Patient name
            2) Track/Gene name
            3) Alteration event
    """

    def __init__(self,
                 data,
                 patients_order=None,
                 tracks_order=None,
                 custom_pieces=None,
                 background_color="#BEBEBE",
                 ):
        self.data = data.copy()
        self.data.columns = ['patient', 'track', 'event']

        if patients_order is None:
            patients_order = self.data['patient'].unique()
        self.patients = patients_order
        self._patients_ix = dict(zip(patients_order, count(0, 1)))

        if tracks_order is None:
            tracks_order = self.data['track'].unique()
        self.tracks = tracks_order
        self._tracks_ix = dict(zip(tracks_order, count(0, 1)))

        if custom_pieces is None:
            custom_pieces = {}
        custom_events = list(custom_pieces.keys())
        self.custom_pieces = custom_pieces

        raw_events = self.data['event'].unique()
        self.events_alt = dict()

        unknown_alterations = []
        for e in raw_events:
            alt = guess_alteration(e)
            if alt == Alteration.OTHER:
                alt = e
                if e not in custom_events:
                    unknown_alterations.append(e)
            self.events_alt[e] = alt
        self.data['event'] = [self.events_alt[e] for e in self.data['event']]
        self.events = self.data['event'].unique()
        if len(unknown_alterations) > 0:
            msg = f"Found unknown alterations: {unknown_alterations}, "\
                  f"please specify a piece for this alteration."
            warnings.warn(msg)

        self._shape = (len(self.tracks), len(self.patients))
        self.background_color = background_color

    @property
    def shape(self):
        return self._shape

    def get_layers_pieces(self):
        layers = {}
        for e in self.events:
            layers[e] = np.zeros(self._shape, dtype=bool)

        for _, row in self.data.iterrows():
            patient, track, event = row
            row_ix = self._tracks_ix[track]
            col_ix = self._patients_ix[patient]
            layers[event][row_ix, col_ix] = True

        # explicitly make copy
        bg_pieces = deepcopy(SHAPE_BANK[Alteration.BACKGROUND])
        bg_pieces.background_color = self.background_color
        new_pieces = [bg_pieces]
        new_layers = [np.ones(self._shape)]
        colors = []
        for alt in self.events:
            new_layers.append(layers[alt])
            if not isinstance(alt, Alteration):
                piece = self.custom_pieces.get(alt)
                if piece is None:
                    # The default style for OTHER
                    piece = FrameRect(color="pink", label=alt)
                    piece.background_color = self.background_color
                    colors.append("pink")
                else:
                    piece = deepcopy(piece)
                    piece.background_color = self.background_color
                    colors.append(piece.color)
                new_pieces.append(piece)
            else:
                p = deepcopy(SHAPE_BANK[alt])
                p.background_color = self.background_color
                new_pieces.append(p)
                colors.append(p.color)

        return new_layers, new_pieces, colors

    def get_track_mutation_rate(self):
        gb = self.data.groupby('track', sort=False, observed=True)
        ts = {}
        for track, df in gb:
            ts[track] = len(df['patient'].unique())
        counts = np.array([ts[t] for t in self.tracks])
        return counts / len(self.patients)

    def get_track_mutation_types(self):
        gb = self.data.groupby('track', sort=False, observed=True)
        cs = {}
        for track, df in gb:
            cs[track] = Counter(df['event'])
        types = [cs[t] for t in self.tracks]
        return pd.DataFrame(types).loc[::-1, self.events].fillna(0.).T

    def get_patient_mutation_types(self):
        gb = self.data.groupby('patient', sort=False, observed=True)
        cs = {}
        for track, df in gb:
            cs[track] = Counter(df['event'])
        types = [cs[p] for p in self.patients]
        return pd.DataFrame(types).loc[:, self.events].fillna(0.).T


class OncoPrint:

    def __init__(self, genomic_data=None,
                 patients_order=None,
                 tracks_order=None,
                 pieces=None,
                 background_color="#BEBEBE",
                 shrink=(.8, .8),
                 width=None,
                 height=None,
                 aspect=2.5,
                 legend_kws=None,
                 name=None,
                 ):
        data = GenomicData(genomic_data,
                           patients_order=patients_order,
                           tracks_order=tracks_order,
                           custom_pieces=pieces,
                           background_color=background_color)
        self.data = data
        width, height = get_canvas_size_by_data(
            data.shape, width=width, height=height, scale=.2, aspect=aspect)
        self.canvas = ClusterBoard(
            name=name,
            cluster_data=np.zeros(data.shape),
            width=width, height=height)
        layers, pieces, bar_colors = data.get_layers_pieces()
        track_names = data.tracks

        legend_options = dict(handleheight=aspect, handlelength=1)
        legend_kws = {} if legend_kws is None else legend_kws
        legend_options.update(legend_kws)

        mesh = LayersMesh(layers=layers, pieces=pieces,
                          shrink=shrink, legend_kws=legend_options)
        self.canvas.add_layer(mesh)
        self.canvas.add_left(Labels(track_names, text_pad=.1))
        # Add other statistics
        track_mut_rate = data.get_track_mutation_rate()
        # Convert to percentage string
        rates = [f"{i}%" for i in (np.array(track_mut_rate) * 100).astype(int)]
        self.canvas.add_right(Labels(rates, text_pad=.1))

        track_counter = data.get_track_mutation_types()
        track_bar = StackBar(track_counter, colors=bar_colors,
                             show_value=False)
        self.canvas.add_right(track_bar, legend=False)

        patients_counter = data.get_patient_mutation_types()
        patients_bar = StackBar(patients_counter, colors=bar_colors,
                                show_value=False)
        self.canvas.add_top(patients_bar, legend=False, pad=.1)

    def render(self):
        self.canvas.add_legends()
        self.canvas.render()

    @property
    def patients_order(self):
        return self.data.patients

    @property
    def tracks_order(self):
        return self.data.tracks


