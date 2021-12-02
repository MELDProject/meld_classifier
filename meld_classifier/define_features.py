#!/bin/sh

import numpy as np
from itertools import chain


class Feature:
    def __init__(self, feature, smoother=None):
        """
        Class to define feature names adding appropriate suffixes and prefixes
        e.g. sm for smooth or _combat for combat features

        Args:
            feature (string): raw feature name
            smoother(int): value for smoothing in mm , if none, no smoothing
        """
        self.raw = feature
        self.smoother = smoother
        self.smooth = self.get_smooth_feature(self.smoother)
        self.combat = self.get_combat_feature()
        self.norm = self.get_norm_feature()
        self.asym = self.get_asym_feature()
        self.all = self.get_list_features()

    def get_smooth_feature(self, smoother=None):
        if smoother != None:
            smooth_part = "sm" + str(int(smoother))
            list_name = self.raw.split(".")
            new_name = list(chain.from_iterable([list_name[0:-1], [smooth_part, list_name[-1]]]))
            smooth_feat = ".".join(new_name)
            return smooth_feat
        else:
            return self.raw

    def get_combat_feature(self):
        combat_feat = "".join([".combat", self.smooth])
        return combat_feat

    def get_norm_feature(self):
        norm_feat = "".join([".inter_z.intra_z", self.combat])
        return norm_feat

    def get_asym_feature(self):
        asym_feat = "".join([".inter_z.asym.intra_z", self.combat])
        return asym_feat

    def get_list_features(self):
        list_feat = [self.smooth, self.combat, self.norm, self.asym]
        return list_feat
