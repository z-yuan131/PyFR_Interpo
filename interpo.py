# -*- coding: utf-8 -*-
from base import BaseInterpo

class Interpo(BaseInterpo):
    def __init__(self, argv):
        super().__init__(argv)

        print(list(self.mesh.keys()))
        
