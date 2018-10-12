from django.shortcuts import render
from django.shortcuts import render_to_response
import django.http # to raise 404's
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.template import Context, Template, loader
from django.views.decorators.cache import cache_control
from django.views.decorators.cache import cache_page
from django.contrib.sessions.backends.db import SessionStore
from django.db import close_old_connections
from django.core.mail import EmailMessage
import os, sys, time,  signal, copy, pickle
import sys, numpy, uuid,  pyeq3, json, re, scipy
from scipy import stats
from itertools import chain
from django.http import HttpResponse
from fitter import Fitter
from numpy import *
from matplotlib.pyplot import *
import scipy.stats

ver = sys.version_info[0]
if ver < 3:
    raise Exception('Python 3 is required to use pyeq3')


class Tags:
    web = []

class Report(object):

    def __init__(self, dataObject):
        self.dataObject = dataObject
        self.stringList = []
        self.name= ""
        self.uuid = uuid.uuid4()


    def PrepareForReportOutput(self):
        self.PrepareForCharacterizerOutput()

    def PrepareForCharacterizerOutput(self):
        pass

    def CreateReportOutput(self):
        self.CreateCharacterizerOutput()


class CurveFit():
    result = []
    array = []
    stats = []
    fittedparameters = []
    fittedparameters_data = []
    upperCoefficientBounds = []
    dataDep = []
    resultData = []
    coefficient_covariance_matrix = []
    results = []
    DataStats = []
    datal = 0
    Fit_Summary = []
    dataString = ""


    def __init__(self,eq):
        self.equation = eq

    def Getfittedparameters(self):
        for i in range(len(self.equation.solvedCoefficients)):
            self.fittedparameters.insert(i, [{
                self.equation.GetCoefficientDesignators()[i]:self.equation.solvedCoefficients[i]
            }
                ]
            )

            self.fittedparameters_data.insert(0,[
            {
                "fitted_parameters": self.fittedparameters
            }
        ])
            return self

    def GetCalculateModelErrors(self):
        datalen = len(self.equation.dataCache.allDataCacheDictionary['DependentData'])

        self.equation.CalculateModelErrors(self.equation.solvedCoefficients,
                                           self.equation.dataCache.allDataCacheDictionary)

        dopIndep1 = 10
        dopIndep2 = 10
        dopDep = 10


        breakLoop = False
        for i in reversed(list(range(dopIndep1))):
            if breakLoop == True:
                break
            for j in range(datalen):  # number of data points

                datapoint = self.equation.dataCache.allDataCacheDictionary['IndependentData'][0][j]

                self.dataDep.insert(0, [{
                    "IndependentDataX": str(self.equation.dataCache.allDataCacheDictionary['IndependentData'][0]),
                    "DependentDataY": str(self.equation.dataCache.allDataCacheDictionary['DependentData']),
                    "m": str(self.equation.modelPredictions),
                    "abserror": str(self.equation.modelAbsoluteError),
                }] )
                if float(('% .' + str(i + 1) + 'E') % (datapoint)) != float(('% .' + str(i) + 'E') % (datapoint)):
                    dopIndep1 = i + 1
                    breakLoop = True
                    break

        breakLoop = False

        return self

    def SetArrays(self):

        self.stats.insert(0, [{
            'stats': self.result
        }])

        return self

    def StartSolve(self):

        pyeq3.dataConvertorService().ConvertAndSortColumnarASCII(self.dataString, self.equation, False)
        self.equation.Solve()


        self.resultData.insert(0,[{
            "Equation":self.equation.GetDisplayHTML(),
            "dimension": str(self.equation.GetDimensionality()) + "D",
            "fittingTarget": self.equation.fittingTargetDictionary[self.equation.fittingTarget],
            "solvedCoefficients": self.equation.CalculateAllDataFittingTarget(self.equation.solvedCoefficients)
        }])

        self.equation.CalculateCoefficientAndFitStatistics()

        upperCoefficientBounds = {
        "Degress_of_freedom error":  self.equation.df_e,
        "Degress_of_freedom_regression":  self.equation.df_r
        }

        self.resultData.insert(0,[upperCoefficientBounds])

        if self.equation.rmse == None:
            self.resultData.insert(0,[{ "RMSE":"n/a" }])
        else:
            self.resultData.insert(0, [{"RMSE": self.equation.rmse}])

        if self.equation.r2 == None:
            self.resultData.insert(0, [{'R_squared': "n/a"}])
        else:
            self.resultData.insert(0,[ {'R_squared':  self.equation.r2}])

        if self.equation.r2adj == None:
            self.resultData.insert(0,[{'R_squared_adjusted': 'n/a'}])
        else:
            self.resultData.insert(0,[{'R_squared_adjusted':  self.equation.r2adj }])

            for i in range(len(self.equation.solvedCoefficients)):
                if type(self.equation.tstat_beta) == type(None):
                    self.resultData.insert(i,[
                       {
                           'coefficientA': { "t-stat" : 'n/a'}
                       }
                   ])
                else:
                    self.resultData.insert(i, [
                        {
                            'coefficientA': {"t-stat": '%-.5E' %  ( self.equation.tstat_beta[i]) }
                        }
                    ])

                if type(self.equation.pstat_beta) == type(None):

                    self.resultData.insert(i, [
                        {
                            'coefficientB': {"p-stat": 'n/a'}
                        }
                    ])
                else:
                    self.resultData.insert(1, [
                        {
                            'coefficientB': {"p-stat": '%-.5E' %  ( self.equation.pstat_beta[i]) }
                        }
                    ])

            self.coefficient_covariance_matrix.insert(0, [{
                "coefficient_covariance_matrix_A":self.equation.cov_beta[0][0]
            }])
            self.coefficient_covariance_matrix.insert(0, [{
                "coefficient_covariance_matrix_A": self.equation.cov_beta[0][1]
            }])
            self.coefficient_covariance_matrix.insert(1, [{
                "coefficient_covariance_matrix_B": self.equation.cov_beta[1][0]
            }])
            self.coefficient_covariance_matrix.insert(1, [{
                "coefficient_covariance_matrix_B": self.equation.cov_beta[1][1]
            }])


            textData = self.dataString
            equationBase = pyeq3.IModel.IModel()
            equationBase._dimensionality = 1
            pyeq3.dataConvertorService().ConvertAndSortColumnarASCII(textData, equationBase, False)

            rawData = equationBase.dataCache.allDataCacheDictionary['IndependentData'][0]

            self.DataStats.insert(0, [{
                "minimum": min(rawData),
                "maximum": max(rawData),
                "mean": scipy.mean(rawData),
                "standard_deviation": scipy.std(rawData),
                "variance": scipy.var(rawData),
                "median": scipy.median(rawData),
                "skew": scipy.stats.skew(rawData),
                "std_error_of_mean": scipy.stats.sem(rawData),
                "kurtosis": scipy.stats.kurtosis(rawData)
            }]
            )
            self.HighChart(rawData)

            self.result = pyeq3.Services.SolverService.SolverService().SolveStatisticalDistribution("beta", rawData,'AICc_BA')
            return self
    def HighChart(self,rawData):

        data = stats.gamma.rvs(2, loc=scipy.mean(rawData), scale=scipy.std(rawData), size=int(self.datal))
        f = Fitter(data,distributions=['norm', 't', 'triang', 'lognorm', 'uniform', 'expon', 'weibull_min',
                                'weibull_max','beta','gamma','logistic','pareto'])
        f.fit()
        dist = scipy.stats.gamma
        param = (f.fitted_param['gamma'])
        X = linspace(min(rawData), max(rawData), int(self.datal))
        pdf_fitted = dist.pdf(X, *param)

        self.Fit_Summary.insert(0, [{
                "graph": {
                    "results": '"' + str(f.summary()) + '"',
                    "Y": str(pdf_fitted),
                    "X":  str(X),
                }
            }]
        )

    @staticmethod
    def CleanArray():
        CurveFit.stats.clear()
        CurveFit.dataDep.clear()
        CurveFit.fittedparameters.clear()
        CurveFit.coefficient_covariance_matrix.clear()
        CurveFit.resultData.clear()
        CurveFit.Fit_Summary.clear()
        CurveFit.DataStats.clear()

    @staticmethod
    def startApp(post,d):

        CurveFit.CleanArray()

        CurveFit.dataString = post
        CurveFit.datal = d
        #equation = pyeq3.Models_2D.Power.Geometric_Modified('SSQABS','Default')
        #equation = pyeq3.Models_2D.Power.PowerLawExponentialCutoff('SSQABS','Default')

        # equation = pyeq3.Models_2D.Exponential.Hocket_Sherby('SSQABS')
        #equation = pyeq3.Models_2D.Polynomial.UserSelectablePolynomial('SSQABS', 'Default', 2)
        #Tempporary Equation running, final version need loop all formulas
        equation = pyeq3.Models_2D.Logarithmic.LinearLogarithmic('SSQABS')
        Fitting = CurveFit(equation)
        Fitting.StartSolve().Getfittedparameters().GetCalculateModelErrors().SetArrays()

        Finalresult = []

        Finalresult.insert(0,[
            {
                'dataDep': CurveFit.dataDep[0],
                'fittedparameters_data': CurveFit.fittedparameters_data[0],
                'coefficient_covariance_matrix': CurveFit.coefficient_covariance_matrix[0],
                'DataStats': CurveFit.DataStats,
                'resultData': CurveFit.resultData,
                'Fit_Summary': CurveFit.Fit_Summary,
                'stats': CurveFit.stats,
            }
        ])
        return json.dumps(Finalresult)
       