import numpy as np
import inspect
import scipy as scp
from scipy.integrate import solve_ivp

#x as dim 1, y as dim 2
class rectangular_domain():
    def __init__(self, size, resolution):

        self.width = size[0]

        self.height = size[1]

        self.X = np.linspace(-self.width/2, self.width/2, resolution)

        self.Y = np.linspace(-self.width/2, self.width/2, resolution)[np.newaxis,:]

        self.fields = {}

    def evaluate(self, func, field_names_tuple):
        ###evalutes a function of the form func(x:np.ndarray, y:np.ndarray) note X is a column vector, Y is a row vector ready for broadcasting
        #verify function signature
        func_sig = inspect.signature(func)

        expected_signature = ['X', 'Y']
        for field in field_names_tuple
            expected_signature.append(field)

        if list(func_sig.parameters.keys()) != expected_signature:
            raise ValueError("func doesnt take (X, Y, fields) as its signature")

        return func(self.X, self.Y, *field_names_tuple)

    def initialize_field_from_func(self, name, func, overwrite=False):

        if (name in self.fields) & ~overwrite:
            raise ValueError("field already initialized")

        self.fields['name'] = self.evaluate(func)

    def time_evolution(self, func, t_array):

        def func_formatted_for_ivp_solver():
