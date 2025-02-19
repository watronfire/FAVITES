#! /usr/bin/env python3
'''
Niema Moshiri 2016

"Driver" module
'''
import abc

class Driver(metaclass=abc.ABCMeta):
    '''
    Abstract class defining the ``Driver`` module

    Methods
    -------
    cite()
        Return citation string (or None)
    init()
        Initialize the module (if need be)
    run()
        Run the simulation
    '''

    @staticmethod
    @abc.abstractmethod
    def init():
        '''
        Initialize the module (if need be)
        '''
        pass

    @staticmethod
    @abc.abstractmethod
    def cite():
        '''
        Return citation string (or None)

        Returns
        -------
        citation : str
            The citation string (or None)
        '''
        pass

    @staticmethod
    @abc.abstractmethod
    def run(path, orig_config):
        '''
        Run the simulation. Will probably want to make use of FAVITES_Global
        variables.

        Parameters
        ----------
        path : str
            Path in which run_favites.py is located
        orig_config : str
            The original configuration file (just to recreate in execution)
        '''
        pass