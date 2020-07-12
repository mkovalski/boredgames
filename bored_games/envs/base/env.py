#!/usr/bin/env python

from abc import ABC, abstractmethod

class Env:
    @abstracmethod
    def n_players(self):
        raise NotImplemented

    @abstactmethod
    def step(self):
        raise NotImplemented

    @abstractmethod
    def reset(self, player):
        raise NotImplemented

