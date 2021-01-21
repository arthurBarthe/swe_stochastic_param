#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 22:05:57 2020

@author: arthur
"""
from torch.nn.modules import Module
import torch.tensor
import numpy as np


class Parameterization:
    def __init__(self, nn: Module, device, mult_factor: float = 1.,
                 every: int = 1):
        self.nn = nn.to(device=device)
        self.device = device
        self.means = dict(s_x=None, s_y=None)
        self.betas = dict(s_x=None, s_y=None)
        self.mult_factor = mult_factor
        self.every = every
        self.counter_0 = 0
        self.counter_1 = 0

    def __call__(self, u, v, eta):
        u *= 10
        v *= 10
        if self.counter_0 == 0:
            with torch.no_grad():
                u = torch.tensor(u, device=self.device).unsqueeze(dim=0).float()
                v = torch.tensor(v, device=self.device).unsqueeze(dim=0).float()
                input_tensor = torch.stack((u, v), dim=1)
                output_tensor = self.nn.forward(input_tensor)
                mean_sx, mean_sy, beta_sx, beta_sy = torch.split(output_tensor,
                                                                 1, dim=1)
                mean_sx = mean_sx.cpu().numpy().squeeze()
                mean_sy = mean_sy.cpu().numpy().squeeze()
                beta_sx = beta_sx.cpu().numpy().squeeze()
                beta_sy = beta_sy.cpu().numpy().squeeze()
                self.apply_mult_factor(mean_sx, mean_sy, beta_sx, beta_sy)
                self.means['s_x'] = mean_sx
                self.means['s_y'] = mean_sy
                self.betas['s_x'] = beta_sx
                self.betas['s_y'] = beta_sy
        else:
            mean_sx = self.means['s_x']
            mean_sy = self.means['s_y']
            beta_sx = self.betas['s_x']
            beta_sy = self.betas['s_y']
        if self.counter_1 == 0:
            self.s_x = mean_sx + np.random.randn(*mean_sx.shape) / beta_sx
            self.s_y = mean_sy + np.random.randn(*mean_sy.shape) / beta_sy
            self.s_x = self.force_zero_sum(self.s_x, mean_sx, 1 / beta_sx)
            self.s_y = self.force_zero_sum(self.s_y, mean_sy, 1 / beta_sy)
            self.s_x *= 1e-7
            self.s_y *= 1e-7
        self.counter_0 += 1
        self.counter_0 %= self.every * 1
        self.counter_1 += 1
        self.counter_1 %= 1
        return self.s_x, self.s_y

    @staticmethod
    def force_zero_sum(data, mean, std):
        return data
        sum_ = np.sum(data)
        sum_std = np.sum(std)
        data = data - sum_ * std / sum_std
        return data

    def apply_mult_factor(self, *args):
        for a in args:
            a *= self.mult_factor


class WaterModelWithDLParameterization:
    def __init__(self, model, parameterization):
        self.model = model
        self.parameterization = parameterization
        self.model.s_x = None
        self.model.s_y = None

        func = self.model.rhs

        def new_rhs(u, v, eta):
            du, dv, deta = func(u, v, eta)
            u = self.model.IuT.dot(u)
            v = self.model.IvT.dot(v)
            u = self.model.h2mat(u)
            v = self.model.h2mat(v)
            # Compute forcing
            s_x, s_y = self.parameterization(u, v, eta)
            # Interpolate
            s_x = s_x.flatten()
            s_x = self.model.ITu.dot(s_x)
            s_y = s_y.flatten()
            s_y = self.model.ITv.dot(s_y)
            self.model.s_x, self.model.s_y = s_x, s_y
            self.model.du, self.model.dv = du, dv
            return du + s_x, dv + s_y, deta
        self.model.rhs = new_rhs

    def __getattr__(self, attr_name: str):
        return getattr(self.model, attr_name)
