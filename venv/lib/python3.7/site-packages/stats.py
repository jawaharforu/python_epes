#!/usr/bin/env python3

##  Module stats.py
##
##  Copyright (c) 2010 Steven D'Aprano.
##
##  Permission is hereby granted, free of charge, to any person obtaining
##  a copy of this software and associated documentation files (the
##  "Software"), to deal in the Software without restriction, including
##  without limitation the rights to use, copy, modify, merge, publish,
##  distribute, sublicense, and/or sell copies of the Software, and to
##  permit persons to whom the Software is furnished to do so, subject to
##  the following conditions:
##
##  The above copyright notice and this permission notice shall be
##  included in all copies or substantial portions of the Software.
##
##  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
##  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
##  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
##  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
##  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
##  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
##  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
'Scientific calculator' statistics for Python 3.

Features:

(1) Standard calculator statistics such as mean and standard deviation:

    >>> mean([-1.0, 2.5, 3.25, 5.75])
    2.625
    >>> stdev([2.5, 3.25, 5.5, 11.25, 11.75])  #doctest: +ELLIPSIS
    4.38961843444...

(2) Single-pass variations on common statistics for use on large iterators
    with little or no loss of precision:

    >>> data = iter([2.5, 3.25, 5.5, 11.25, 11.75])
    >>> stdev1(data)  #doctest: +ELLIPSIS
    4.38961843444...

(3) Order statistics such as the median and quartiles:

    >>> median([6, 1, 5, 4, 2, 3])
    3.5
    >>> quartiles([2, 4, 5, 3, 1, 6])
    (2, 3.5, 5)

(4) Over forty statistics, including such functions as trimean and standard
    error of the mean:

    >>> trimean([15, 18, 20, 29, 35])
    21.75
    >>> sterrmean(3.25, 100, 1000)  #doctest: +ELLIPSIS
    0.30847634861...

"""


# Module metadata.
__version__ = "0.1.2a"
__date__ = "2010-12-31"
__author__ = "Steven D'Aprano"
__author_email__ = "steve+python@pearwood.info"



__all__ = [
    # Means and averages:
    'mean', 'harmonic_mean', 'geometric_mean', 'quadratic_mean',
    # Other measures of central tendancy:
    'median', 'mode', 'midrange', 'midhinge', 'trimean',
    # Moving averages:
    'running_average', 'weighted_running_average', 'simple_moving_average',
    # Order statistics:
    'quartiles', 'hinges', 'quantile', 'decile', 'percentile',
    # Measures of spread:
    'pvariance', 'variance', 'pstdev', 'stdev',
    'pvariance1', 'variance1', 'pstdev1', 'stdev1',
    'range', 'iqr', 'average_deviation', 'median_average_deviation',
    # Other moments:
    'quartile_skewness', 'pearson_mode_skewness', 'skewness', 'kurtosis',
    # Multivariate statistics:
    'qcorr', 'corr', 'corr1', 'pcov', 'cov', 'errsumsq', 'linr',
    # Sums and products:
    'sum', 'sumsq', 'product', 'cumulative_sum', 'running_sum',
    'Sxx', 'Syy', 'Sxy',
    # Assorted others:
    'StatsError', 'QUARTILE_DEFAULT', 'QUANTILE_DEFAULT',
    'sterrmean', 'stderrskewness', 'stderrkurtosis', 'minmax',
    'coroutine', 'feed',
    # Statistics of circular quantities:
    'circular_mean',
    ]


import math
import operator
import functools
import itertools
import collections


# === Global variables ===

# Default schemes to use for order statistics:
QUARTILE_DEFAULT = 1
QUANTILE_DEFAULT = 1


# === Exceptions ===

class StatsError(ValueError):
    pass


# === Utility functions and classes ===


def sorted_data(func):
    """Decorator to sort data passed to stats functions."""
    @functools.wraps(func)
    def inner(data, *args, **kwargs):
        data = sorted(data)
        return func(data, *args, **kwargs)
    return inner


def minmax(*values, **kw):
    """minmax(iterable [, key=func]) -> (minimum, maximum)
    minmax(a, b, c, ... [key=func]) -> (minimum, maximum)

    With a single iterable argument, return a two-tuple of its smallest
    item and largest item. With two or more arguments, return the smallest
    and largest arguments.
    """
    if len(values) == 0:
        raise TypeError('minmax expected at least one argument, but got none')
    elif len(values) == 1:
        values = values[0]
    if isinstance(values, collections.Sequence):
        # For speed, fall back on built-in min and max functions when
        # data is a sequence and can be safely iterated over twice.
        minimum = min(values, **kw)
        maximum = max(values, **kw)
    else:
        # Iterator argument, so fall back on a slow pure-Python solution.
        if list(kw.keys()) not in ([], ['key']):
            raise TypeError('minmax received an unexpected keyword argument')
        key = kw.get('key')
        if key is not None:
            it = ((key(value), value) for value in values)
        else:
            it = ((value, value) for value in values)
        try:
            keyed_min, minimum = next(it)
        except StopIteration:
            raise ValueError('minmax argument is empty')
        keyed_max, maximum = keyed_min, minimum
        try:
            while True:
                a = next(it)
                try:
                    b = next(it)
                except StopIteration:
                    b = a
                if a[0] > b[0]:
                    a, b = b, a
                if a[0] < keyed_min:
                    keyed_min, minimum = a
                if b[0] > keyed_max:
                    keyed_max, maximum = b
        except StopIteration:
            pass
    return (minimum, maximum)


# Modified from http://code.activestate.com/recipes/393090/
def add_partial(x, partials):
    """Helper function for full-precision summation of binary floats.

    Adds x in place to the list partials.
    """
    # Rounded x+y stored in hi with the round-off stored in lo.  Together
    # hi+lo are exactly equal to x+y.  The inner loop applies hi/lo summation
    # to each partial so that the list of partial sums remains exact.
    # Depends on IEEE-754 arithmetic guarantees.  See proof of correctness at:
    # www-2.cs.cmu.edu/afs/cs/project/quake/public/papers/robust-arithmetic.ps
    i = 0
    for y in partials:
        if abs(x) < abs(y):
            x, y = y, x
        hi = x + y
        lo = y - (hi - x)
        if lo:
            partials[i] = lo
            i += 1
        x = hi
    partials[i:] = [x]


def as_sequence(iterable):
    """Helper function to convert iterable arguments into sequences."""
    if isinstance(iterable, (list, tuple)): return iterable
    else: return list(iterable)


class _Multivariate:
    # Helpers for dealing with multivariate functions.

    def __new__(cls):
        raise RuntimeError('namespace, do not instantiate')

    def split(xdata, ydata=None):
        """Helper function which splits xydata into (xdata, ydata)."""
        # The two-argument case is easy -- just pass them unchanged.
        if ydata is not None:
            xdata = as_sequence(xdata)
            ydata = as_sequence(ydata)
            if len(xdata) < len(ydata):
                ydata = ydata[:len(xdata)]
            elif len(xdata) > len(ydata):
                xdata = xdata[:len(ydata)]
            assert len(xdata) == len(ydata)
            return (xdata, ydata)
        # The single argument case could be either [x0, x1, x2, ...] or
        # [(x0, y0), (x1, y1), (x2, y2), ...]. We decide which it is by
        # looking at the first item, and treating it as canonical.
        it = iter(xdata)
        try:
            first = next(it)
        except StopIteration:
            # If the iterable is empty, return two empty lists.
            return ([], [])
        # If we get here, we know we have a single iterable argument with at
        # least one item. Does it look like a sequence of (x,y) values, or
        # like a sequence of x values?
        try:
            n = len(first)
        except TypeError:
            # Looks like we're dealing with the case [x0, x1, x2, ...]
            # This isn't exactly *multivariate*, but we support it anyway.
            # We leave it up to the caller to decide what to do with the
            # fake y values.
            xdata = [first]
            xdata.extend(it)
            return (xdata, [None]*len(xdata))
        # Looks like [(x0, y0), (x1, y1), (x2, y2), ...]
        # Here we expect that each point has two items, and fail if not.
        if n != 2:
            raise TypeError('expecting 2-tuple (x, y) but got %d-tuple' % n)
        xlist = [first[0]]
        ylist = [first[1]]
        for x,y in it:
            xlist.append(x)
            ylist.append(y)
        assert len(xlist) == len(ylist)
        return (xlist, ylist)

    def merge(xdata, ydata=None):
        """Helper function which merges xdata, ydata into xydata."""
        if ydata is not None:
            # Two argument version is easy.
            return zip(xdata, ydata)
        # The single argument case could be either [x0, x1, x2, ...] or
        # [(x0, y0), (x1, y1), (x2, y2), ...]. We decide which it is by looking
        # at the first item, and treating it as canonical.
        it = iter(xdata)
        try:
            first = next(it)
        except StopIteration:
            # If the iterable is empty, return the original.
            return xdata
        # If we get here, we know we have a single iterable argument with at
        # least one item. Does it look like a sequence of (x,y) values, or
        # like a sequence of x values?
        try:
            len(first)
        except TypeError:
            # Looks like we're dealing with the case [x0, x1, x2, ...]
            first = (first, None)
            tail = ((x, None) for x in it)
            return itertools.chain([first], tail)
        # Looks like [(x0, y0), (x1, y1), (x2, y2), ...]
        # Notice that we DON'T care how many items are in the data points
        # here, we postpone dealing with any mismatches to later.
        return itertools.chain([first], it)

    def split_xydata(func):
        """Decorator to split a single (x,y) data iterable into separate x
        and y iterables.
        """
        @functools.wraps(func)
        def inner(xdata, ydata=None):
            xdata, ydata = _Multivariate.split(xdata, ydata)
            return func(xdata, ydata)
        return inner

    def merge_xydata(func):
        """Decorator to merge separate x, y data iterables into a single
        (x,y) iterator.
        """
        @functools.wraps(func)
        def inner(xdata, ydata=None):
            xydata = _Multivariate.merge(xdata, ydata)
            return func(xydata)
        return inner


def _validate_int(n):
    # This will raise TypeError, OverflowError (for infinities) or
    # ValueError (for NANs or non-integer numbers).
    if n != int(n):
        raise ValueError('requires integer value')


def _interpolate(data, x):
    i, f = math.floor(x), x%1
    if f:
        a, b = data[i], data[i+1]
        return a + f*(b-a)
    else:
        return data[i]


# Rounding modes.
_UP, _DOWN, _EVEN = 0, 1, 2

def _round(x, rounding_mode):
    """Round non-negative x, with ties rounding according to rounding_mode."""
    assert rounding_mode in (_UP, _DOWN, _EVEN)
    assert x >= 0.0
    n, f = int(x), x%1
    if rounding_mode == _UP:
        if f >= 0.5:
            return n+1
        else:
            return n
    elif rounding_mode == _DOWN:
        if f > 0.5:
            return n+1
        else:
            return n
    else:
        # Banker's rounding to EVEN.
        if f > 0.5:
            return n+1
        elif f < 0.5:
            return n
        else:
            if n%2:
                # n is odd, so round up to even.
                return n+1
            else:
                # n is even, so round down.
                return n


def coroutine(func):
    """Co-routine decorator"""
    @functools.wraps(func)
    def started(*args, **kwargs):
        cr = func(*args,**kwargs)
        cr.send(None)
        return cr
    return started


def feed(consumer, iterable):
    """feed(consumer, iterable) -> yield items

    Helper function to send elements from an iterable into a coroutine.

    >>> def counter():              # Count the items sent in.
    ...     c = 0
    ...     _ = (yield None)        # Start the coroutine.
    ...     while True:
    ...             c += 1
    ...             _ = (yield c)   # Accept a value sent into the coroutine.
    ... 
    >>> cr = counter()
    >>> cr.send(None)  # Prime the coroutine.
    >>> list(feed(cr, ["spam", "ham", "eggs"]))  # Send many values.
    [1, 2, 3]
    >>> cr.send("spam and eggs")  # Manually sending works too.
    4

    """
    for obj in iterable:
        yield consumer.send(obj)


# === Basic univariate statistics ===


# Measures of central tendency (means and averages)
# -------------------------------------------------


def mean(data):
    """Return the sample arithmetic mean of a sequence of numbers.

    >>> mean([1.0, 2.0, 3.0, 4.0])
    2.5

    The arithmetic mean is the sum of the data divided by the number of data.
    It is commonly called "the average". It is a measure of the central
    location of the data.
    """
    n, total = _generalised_sum(data, None)
    if n == 0:
        raise StatsError('mean of empty sequence is not defined')
    return total/n


def harmonic_mean(data):
    """Return the sample harmonic mean of a sequence of non-zero numbers.

    >>> harmonic_mean([0.25, 0.5, 1.0, 1.0])
    0.5

    The harmonic mean, or subcontrary mean, is the reciprocal of the
    arithmetic mean of the reciprocals of the data. It is best suited for
    averaging rates.
    """
    try:
        m = mean(1.0/x for x in data)
    except ZeroDivisionError:
        # FIXME need to preserve the sign of the zero?
        # FIXME is it safe to assume that if data contains 1 or more zeroes,
        # the harmonic mean must itself be zero?
        return 0.0
    if m == 0.0:
        return math.copysign(float('inf'), m)
    return 1/m


def geometric_mean(data):
    """Return the sample geometric mean of a sequence of positive numbers.

    >>> geometric_mean([1.0, 2.0, 6.125, 12.25])
    3.5

    The geometric mean is the Nth root of the product of the data. It is
    best suited for averaging exponential growth rates.
    """
    ap = add_partial
    log = math.log
    partials = []
    count = 0
    try:
        for x in data:
            count += 1
            ap(log(x), partials)
    except ValueError:
        if x < 0:
            raise StatsError('geometric mean of negative number')
        return 0.0
    if count == 0:
        raise StatsError('geometric mean of empty sequence is not defined')
    p = math.exp(math.fsum(partials))
    return pow(p, 1.0/count)


def quadratic_mean(data):
    """Return the sample quadratic mean of a sequence of numbers.

    >>> quadratic_mean([2, 2, 4, 5])
    3.5

    The quadratic mean, or root-mean-square (RMS), is the square root of the
    arithmetic mean of the squares of the data. It is best used when
    quantities vary from positive to negative.
    """
    return math.sqrt(mean(x*x for x in data))


@sorted_data
def median(data, sign=0):
    """Returns the median (middle) value of a sequence of numbers.

    >>> median([3.0, 5.0, 2.0])
    3.0

    The median is the middle data point in a sorted sequence of values. If
    the argument to median is a list, it will be sorted in place, otherwise
    the values will be collected into a sorted list.

    The median is commonly used as an average. It is more robust than the
    mean for data that contains outliers. The median is equivalent to the
    second quartile or the 50th percentile.

    Optional numeric argument sign specifies the behaviour of median in the
    case where there is an even number of elements:

    sign  value returned as median
    ----  ------------------------------------------------------
    0     The mean of the elements on either side of the middle
    < 0   The element just below the middle ("low median")
    > 0   The element just above the middle ("high median")

    The default is 0. Except for certain specialist applications, this is
    normally what is expected for the median.
    """
    n = len(data)
    if n == 0:
        raise StatsError('no median for empty iterable')
    m = n//2
    if n%2 == 1:
        # For an odd number of items, there is only one middle element, so
        # we always take that.
        return data[m]
    else:
        # If there are an even number of items, we decide what to do
        # according to sign:
        if sign == 0:
            # Take the mean of the two middle elements.
            return (data[m-1] + data[m])/2
        elif sign < 0:
            # Take the lower middle element.
            return data[m-1]
        elif sign > 0:
            # Take the higher middle element.
            return data[m]
        else:
            # Possibly a NAN? Something stupid, in any case.
            raise TypeError('sign is not ordered with respect to zero')


def mode(data):
    """Returns the single most common element of a sequence of numbers.

    >>> mode([5.0, 7.0, 2.0, 3.0, 2.0, 2.0, 1.0, 3.0])
    2.0

    Raises StatsError if there is no mode, or if it is not unique.

    The mode is commonly used as an average.
    """
    L = sorted(
        [(count, value) for (value, count) in count_elems(data).items()],
        reverse=True)
    if len(L) == 0:
        raise StatsError('no mode is defined for empty iterables')
    # Test if there are more than one modes.
    if len(L) > 1 and L[0][0] == L[1][0]:
        raise StatsError('no distinct mode')
    return L[0][1]


def midrange(data):
    """Returns the midrange of a sequence of numbers.

    >>> midrange([2.0, 4.5, 7.5])
    4.75

    The midrange is halfway between the smallest and largest element. It is
    a weak measure of central tendency.
    """
    try:
        L, H = minmax(data)
    except ValueError as e:
        e.args = ('no midrange defined for empty iterables',)
        raise
    return (L + H)/2


def midhinge(data):
    """Return the midhinge of a sequence of numbers.

    >>> midhinge([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
    4.5

    The midhinge is halfway between the first and second hinges. It is a
    better measure of central tendency than the midrange, and more robust
    than the sample mean (more resistant to outliers).
    """
    H1, _, H2 = hinges(data)
    return (H1 + H2)/2


@sorted_data
def trimean(data):
    """Return Tukey's trimean = (H1 + 2*M + H2)/4 of data


    >>> trimean([1, 1, 3, 5, 7, 9, 10, 14, 18])
    6.75
    >>> trimean([0, 1, 2, 3, 4, 5, 6, 7, 8])
    4.0

    The trimean is equivalent to the average of the median and the midhinge,
    and is considered a better measure of central tendancy than either alone.
    """
    H1, M, H2 = hinges(data)
    return (H1 + 2*M + H2)/4


# Moving averages
# ---------------

def running_average(data):
    """Iterate over data, yielding the running average.

    >>> list(running_average([40, 30, 50, 46, 39, 44]))
    [40, 35.0, 40.0, 41.5, 41.0, 41.5]

    The running average is also known as the cumulative moving average.
    Given data [a, b, c, d, ...] it yields the values:
        a, (a+b)/2, (a+b+c)/3, (a+b+c+d)/4, ...

    that is, the average of the first item, the first two items, the first
    three items, the first four items, ...
    """
    it = iter(data)
    ca = next(it)
    yield ca
    for i, x in enumerate(it, 2):
        ca = (x + (i-1)*ca)/i
        yield ca


def weighted_running_average(data):
    """Iterate over data, yielding a running average with exponentially
    decreasing weights.

    >>> list(weighted_running_average([40, 30, 50, 46, 39, 44]))
    [40, 35.0, 42.5, 44.25, 41.625, 42.8125]

    This running average yields the average between the previous running
    average and the current data point. Given data [a, b, c, d, ...] it
    yields the values:
        a, (a+b)/2, ((a+b)/2 + c)/2, (((a+b)/2 + c)/2 + d)/2, ...

    The values yielded are weighted means where the weight on older points
    decreases exponentially.
    """
    it = iter(data)
    ca = next(it)
    yield ca
    for x in it:
        ca = (ca + x)/2
        yield ca


def simple_moving_average(data, window=3):
    """Iterate over data, yielding the simple moving average with a fixed
    window size (defaulting to three).

    >>> list(simple_moving_average([40, 30, 50, 46, 39, 44]))
    [40.0, 42.0, 45.0, 43.0]

    """
    it = iter(data)
    d = collections.deque(itertools.islice(it, window))
    if len(d) != window:
        raise StatsError('too few data points for given window size')
    s = sum(d)
    yield s/window
    for x in it:
        s += x - d.popleft()
        d.append(x)
        yield s/window


# Order statistics: quartiles, quantiles (fractiles) and hinges
# -------------------------------------------------------------

# Grrr arggh!!! Nobody can agree on how to calculate order statistics.
# Langford (2006) finds no fewer than FIFTEEN methods for calculating
# quartiles (although some are mathematically equivalent to others):
#   http://www.amstat.org/publications/jse/v14n3/langford.html
# Mathword and Dr Math suggest five:
#   http://mathforum.org/library/drmath/view/60969.html
#   http://mathworld.wolfram.com/Quartile.html
#
# Quantiles (fractiles) and percentiles also have a plethora of methods.
# R (and presumably S) include nine different calculation methods for
# quantiles. Mathematica uses a parameterized quantile function capable
# of matching eight of those nine methods. Wikipedia lists a tenth method.
# There are probably others I don't know of.

class _Quartiles:
    """Private namespace for quartile calculation methods.

    ALL methods and attributes in this namespace class are private and
    subject to change without notice.
    """
    def __new__(cls):
        raise RuntimeError('namespace, do not initialise')

    def inclusive(data):
        """Return sample quartiles using Tukey's method.

        Q1 and Q3 are calculated as the medians of the two halves of the data,
        where the median Q2 is included in both halves. This is equivalent to
        Tukey's hinges H1, M, H2.
        """
        n = len(data)
        i = (n+1)//4
        m = n//2
        if n%4 in (0, 3):
            q1 = (data[i] + data[i-1])/2
            q3 = (data[-i-1] + data[-i])/2
        else:
            q1 = data[i]
            q3 = data[-i-1]
        if n%2 == 0:
            q2 = (data[m-1] + data[m])/2
        else:
            q2 = data[m]
        return (q1, q2, q3)

    def exclusive(data):
        """Return sample quartiles using Moore and McCabe's method.

        Q1 and Q3 are calculated as the medians of the two halves of the data,
        where the median Q2 is excluded from both halves.

        This is the method used by Texas Instruments model TI-85 calculator.
        """
        n = len(data)
        i = n//4
        m = n//2
        if n%4 in (0, 1):
            q1 = (data[i] + data[i-1])/2
            q3 = (data[-i-1] + data[-i])/2
        else:
            q1 = data[i]
            q3 = data[-i-1]
        if n%2 == 0:
            q2 = (data[m-1] + data[m])/2
        else:
            q2 = data[m]
        return (q1, q2, q3)

    def ms(data):
        """Return sample quartiles using Mendenhall and Sincich's method."""
        # Perform index calculations using 1-based counting, and adjust for
        # 0-based at the very end.
        n = len(data)
        M = _round((n+1)/2, _EVEN)
        L = _round((n+1)/4, _UP)
        U = n+1-L
        assert U == _round(3*(n+1)/4, _DOWN)
        return (data[L-1], data[M-1], data[U-1])

    def minitab(data):
        """Return sample quartiles using the method used by Minitab."""
        # Perform index calculations using 1-based counting, and adjust for
        # 0-based at the very end.
        n = len(data)
        M = (n+1)/2
        L = (n+1)/4
        U = n+1-L
        assert U == 3*(n+1)/4
        return (
                _interpolate(data, L-1),
                _interpolate(data, M-1),
                _interpolate(data, U-1)
                )

    def excel(data):
        """Return sample quartiles using Freund and Perles' method.

        This is also the method used by Excel and OpenOffice.
        """
        # Perform index calculations using 1-based counting, and adjust for
        # 0-based at the very end.
        n = len(data)
        M = (n+1)/2
        L = (n+3)/4
        U = (3*n+1)/4
        return (
                _interpolate(data, L-1),
                _interpolate(data, M-1),
                _interpolate(data, U-1)
                )

    def langford(data):
        """Langford's recommended method for calculating quartiles based on
        the cumulative distribution function (CDF).
        """
        n = len(data)
        m = n//2
        i, r = divmod(n, 4)
        if r == 0:
            q1 = (data[i] + data[i-1])/2
            q2 = (data[m-1] + data[m])/2
            q3 = (data[-i-1] + data[-i])/2
        elif r in (1, 3):
            q1 = data[i]
            q2 = data[m]
            q3 = data[-i-1]
        else:  # r == 2
            q1 = data[i]
            q2 = (data[m-1] + data[m])/2
            q3 = data[-i-1]
        return (q1, q2, q3)

    # Numeric method selectors for quartiles:
    QUARTILE_MAP = {
        1: inclusive,
        2: exclusive,
        3: ms,
        4: minitab,
        5: excel,
        6: langford,
        }
        # Note: if you modify this, you must also update the docstring for
        # the quartiles function.

    # Lowercase aliases for the numeric method selectors for quartiles:
    QUARTILE_ALIASES = {
        'cdf': 6,
        'excel': 5,
        'exclusive': 2,
        'f&p': 5,
        'hinges': 1,
        'inclusive': 1,
        'langford': 6,
        'm&m': 2,
        'm&s': 3,
        'minitab': 4,
        'openoffice': 5,
        'ti-85': 2,
        'tukey': 1,
        }
# End of private _Quartiles namespace.


class _Quantiles:
    """Private namespace for quantile calculation methods.

    ALL methods and attributes in this namespace class are private and
    subject to change without notice.
    """
    def __new__(cls):
        raise RuntimeError('namespace, do not instantiate')

    # The functions r1...r9 implement R's quartile types 1...9 respectively.
    # Except for r2, they are also equivalent to Mathematica's parametrized
    # quantile function: http://mathworld.wolfram.com/Quantile.html

    # Implementation notes
    # --------------------
    #
    # * The usual formulae for quartiles use 1-based indexes.
    # * Each of the functions r1...r9 assume that data is a sorted sequence,
    #   and that p is a fraction 0 <= p <= 1.

    def r1(data, p):
        h = len(data)*p + 0.5
        i = max(1, math.ceil(h - 0.5))
        assert 1 <= i <= len(data)
        return data[i-1]

    def r2(data, p):
        """Langford's Method #4 for calculating general quantiles using the
        cumulative distribution function (CDF); this is also R's method 2 and
        SAS' method 5.
        """
        n = len(data)
        h = n*p + 0.5
        i = max(1, math.ceil(h - 0.5))
        j = min(n, math.floor(h + 0.5))
        assert 1 <= i <= j <= n
        return (data[i-1] + data[j-1])/2

    def r3(data, p):
        h = len(data)*p
        i = max(1, round(h))
        assert 1 <= i <= len(data)
        return data[i-1]

    def r4(data, p):
        n = len(data)
        if p < 1/n: return data[0]
        elif p == 1.0: return data[-1]
        else: return _interpolate(data, n*p - 1)

    def r5(data, p):
        n = len(data)
        if p < 1/(2*n): return data[0]
        elif p >= (n-0.5)/n: return data[-1]
        h = n*p + 0.5
        return _interpolate(data, h-1)

    def r6(data, p):
        n = len(data)
        if p < 1/(n+1): return data[0]
        elif p >= n/(n+1): return data[-1]
        h = (n+1)*p
        return _interpolate(data, h-1)

    def r7(data, p):
        n = len(data)
        if p == 1: return data[-1]
        h = (n-1)*p + 1
        return _interpolate(data, h-1)

    def r8(data, p):
        n = len(data)
        h = (n + 1/3)*p + 1/3
        h = max(1, min(h, n))
        return _interpolate(data, h-1)

    def r9(data, p):
        n = len(data)
        h = (n + 0.25)*p + 3/8
        h = max(1, min(h, n))
        return _interpolate(data, h-1)

    def lqd(data, p):
        n = len(data)
        h = (n + 2)*p - 0.5
        h = max(1, min(h, n))
        return _interpolate(data, h-1)

    # Numeric method selectors for quartiles. Numbers 1-9 MUST match the R
    # calculation methods with the same number.
    QUANTILE_MAP = {
        1: r1,
        2: r2,
        3: r3,
        4: r4,
        5: r5,
        6: r6,
        7: r7,
        8: r8,
        9: r9,
        10: lqd,
        }
        # Note: if you add any additional methods to this, you must also
        # update the docstring for the quantiles function.

    # Lowercase aliases for quantile schemes:
    QUANTILE_ALIASES = {
        'cdf': 2,
        'excel': 7,
        'h&f': 8,
        'hyndman': 8,
        'matlab': 5,
        'minitab': 6,
        'sas-1': 4,
        'sas-2': 3,
        'sas-3': 1,
        'sas-4': 6,
        'sas-5': 2,
        }
# End of private _Quantiles namespace.


@sorted_data
def quartiles(data, scheme=None):
    """quartiles(data [, scheme]) -> (Q1, Q2, Q3)

    Return the sample quartiles (Q1, Q2, Q3) for data, where one quarter of
    the data is below Q1, two quarters below Q2, and three quarters below Q3.
    data must be an iterator of numeric values, with at least three items.

    >>> quartiles([0.5, 2.0, 3.0, 4.0, 5.0, 6.0])
    (2.0, 3.5, 5.0)

    In general, data sets don't divide evenly into four equal sets, and so
    calculating quartiles requires a method for splitting data points. The
    optional argument scheme specifies the calculation method used. The
    exact values returned as Q1, Q2 and Q3 will depend on the method.

    scheme  Description
    ======  =================================================================
    1       Tukey's method; median is included in the two halves
    2       Moore and McCabe's method; median is excluded from the two halves
    3       Method recommended by Mendenhall and Sincich
    4       Method used by Minitab software
    5       Method recommended by Freund and Perles
    6       Langford's CDF method

    Notes:

        (1) If scheme is missing or None, the default is taken from the
            global variable QUARTILE_DEFAULT (set to 1 by default).
        (2) Scheme 1 is equivalent to Tukey's hinges (H1, M, H2).
        (3) Scheme 2 is used by Texas Instruments calculators starting with
            model TI-85.
        (4) Scheme 3 ensures that the values returned are always data points.
        (5) Schemes 4 and 5 use linear interpolation between items.
        (6) For compatibility with Microsoft Excel and OpenOffice, use
            scheme 5.

    Case-insensitive named aliases are also supported: you can examine
    quartiles.aliases for a mapping of names to schemes.
    """
    n = len(data)
    if n < 3:
        raise StatsError('need at least 3 items to split data into quartiles')
    # Select a method.
    if scheme is None: scheme = QUARTILE_DEFAULT
    if isinstance(scheme, str):
        key = quartiles.aliases.get(scheme.lower())
    else:
        key = scheme
    func = _Quartiles.QUARTILE_MAP.get(key)
    if func is None:
        raise StatsError('unrecognised scheme `%s`' % scheme)
    return func(data)

quartiles.aliases = _Quartiles.QUARTILE_ALIASES  # TO DO make this read-only?


def hinges(data):
    """Return Tukey's hinges H1, M, H2 from data.

    >>> hinges([2, 4, 6, 8, 10, 12, 14, 16, 18])
    (6, 10, 14)

    If the data has length N of the form 4n+5 (e.g. 5, 9, 13, 17...) then
    the hinges can be visualised by writing out the ordered data in the
    shape of a W, where each limb of the W is equal is length. For example,
    the data (A,B,C,...) with N=9 would be written out like this:

        A       E       I
          B   D   F   H
            C       G

    and the hinges would be C, E and G.

    This is equivalent to quartiles() called with scheme=1.
    """
    return quartiles(data, scheme=1)


@sorted_data
def quantile(data, p, scheme=None):
    """quantile(data, p [, scheme]) -> value

    Return the value which is some fraction p of the way into data after
    sorting. data must be an iterator of numeric values, with at least two
    items. p must be a number between 0 and 1 inclusive. The result returned
    by quantile is the data point, or the interpolated data point, such that
    a fraction p of the data is less than that value.

    >>> data = [2.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    >>> quantile(data, 0.75)
    5.0


    Interpolation
    =============

    In general the quantile will not fall exactly on a data point. When that
    happens, the value returned is interpolated from the data points nearest
    the calculated position. There are a wide variety of interpolation methods
    used in the statistics literature, and quantile() allows you to choose
    between them using the optional argument scheme.

    >>> quantile(data, 0.75, scheme=4)
    4.5
    >>> quantile(data, 0.75, scheme=7)
    4.75

    scheme can be either an integer scheme number (see table below), a tuple
    of four numeric parameters, or a case-insensitive string alias for either
    of these. You can examine quantiles.aliases for a mapping of names to
    scheme numbers or parameters.

        WARNING:
        The use of arbitrary values as a four-parameter scheme is not
        recommended! Although quantile will calculate a result using them,
        the result is unlikely to be meaningful or statistically useful.

    Integer schemes 1-9 are equivalent to R's quantile types with the same
    number. These are also equivalent to Mathematica's parameterized quartile
    function with parameters shown:

    scheme  parameters   Description
    ======  ===========  ====================================================
    1       0,0,1,0      inverse of the empirical CDF
    2       n/a          inverse of empirical CDF with averaging
    3       1/2,0,0,0    closest actual observation
    4       0,0,0,1      linear interpolation of the empirical CDF
    5       1/2,0,0,1    Hazen's model (like Matlab's PRCTILE function)
    6       0,1,0,1      Weibull quantile
    7       1,-1,0,1     interpolation over range divided into n-1 intervals
    8       1/3,1/3,0,1  interpolation of the approximate medians
    9       3/8,1/4,0,1  approx. unbiased estimate for a normal distribution
    10      n/a          least expected square deviation relative to p

    Notes:

        (1) If scheme is missing or None, the default is taken from the
            global variable QUANTILE_DEFAULT (set to 1 by default).
        (2) Scheme 1 ensures that the values returned are always data points,
            and is the default used by Mathematica.
        (3) Scheme 5 is equivalent to Matlab's PRCTILE function.
        (4) Scheme 6 is equivalent to the method used by Minitab.
        (5) Scheme 7 is the default used by programming languages R and S,
            and is the method used by Microsoft Excel and OpenOffice.
        (6) Scheme 8 is recommended by Hyndman and Fan (1996).

    Example of using a scheme written in the parameterized form used by
    Mathematica:

    >>> data = [1, 2, 3, 3, 4, 5, 7, 9, 12, 12]
    >>> quantile(data, 0.2, scheme=(1, -1, 0, 1))  # First quintile.
    2.8

    This can also be written using an alias:

    >>> quantile(data, 0.2, scheme='excel')
    2.8

    """
    # More details here:
    # http://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html
    # http://en.wikipedia.org/wiki/Quantile
    if not 0.0 <= p <= 1.0:
        raise StatsError('quantile argument must be between 0.0 and 1.0')
    if len(data) < 2:
        raise StatsError('need at least 2 items to split data into quantiles')
    # Select a scheme.
    if scheme is None: scheme = QUANTILE_DEFAULT
    if isinstance(scheme, str):
        key = quantile.aliases.get(scheme.lower())
    else:
        key = scheme
    if isinstance(key, tuple) and len(key) == 4:
        return _parametrized_quantile(key, data, p)
    else:
        func = _Quantiles.QUANTILE_MAP.get(key)
        if func is None:
            raise StatsError('unrecognised scheme `%s`' % scheme)
        return func(data, p)

quantile.aliases = _Quantiles.QUANTILE_ALIASES  # TO DO make this read-only?


def _parametrized_quantile(parameters, data, p):
    """_parameterized_quantile(parameters, data, p) -> value

    Private function calculating a parameterized version of quantile,
    equivalent to the Mathematica Quantile() function.

    data is assumed to be sorted and with at least two items; p is assumed
    to be between 0 and 1 inclusive. If either of these assumptions are
    violated, the behaviour of this function is undefined.

    >>> from builtins import range; data = range(1, 21)
    >>> _parametrized_quantile((0, 0, 1, 0), data, 0.3)
    6.0
    >>> _parametrized_quantile((1/2, 0, 0, 1), data, 0.3)
    6.5

    WARNING: While this function will accept arbitrary numberic values for
    the parameters, not all such combinations are meaningful:

    >>> _parametrized_quantile((1, 1, 1, 1), [1, 2], 0.3)
    2.9

    """
    # More details here:
    # http://reference.wolfram.com/mathematica/ref/Quantile.html
    # http://mathworld.wolfram.com/Quantile.html
    a, b, c, d = parameters
    n = len(data)
    h = a + (n+b)*p
    f = h % 1
    i = max(1, min(math.floor(h), n))
    j = max(1, min(math.ceil(h), n))
    x = data[i-1]
    y = data[j-1]
    return x + (y - x)*(c + d*f)


def decile(data, d, scheme=None):
    """Return the dth decile of data, for integer d between 0 and 10.

    >>> data = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    >>> decile(data, 7, scheme=1)
    14

    See function quantile for details about the optional argument scheme.
    """
    _validate_int(d)
    if not 0 <= d <= 10:
        raise ValueError('decile argument d must be between 0 and 10')
    from fractions import Fraction
    return quantile(data, Fraction(d, 10), scheme)


def percentile(data, p, scheme=None):
    """Return the pth percentile of data, for integer p between 0 and 100.

    >>> import builtins; data = builtins.range(1, 201)
    >>> percentile(data, 7, scheme=1)
    14

    See function quantile for details about the optional argument scheme.
    """
    _validate_int(p)
    if not 0 <= p <= 100:
        raise ValueError('percentile argument p must be between 0 and 100')
    from fractions import Fraction
    return quantile(data, Fraction(p, 100), scheme)


# Measures of spread (dispersion or variability)
# ----------------------------------------------

def pvariance(data, m=None):
    """pvariance(data [, m]) -> population variance of data.

    >>> pvariance([0.25, 0.5, 1.25, 1.25,
    ...           1.75, 2.75, 3.5])  #doctest: +ELLIPSIS
    1.17602040816...

    If you know the population mean, or an estimate of it, then you can pass
    the mean as the optional argument m. See also pstdev.

    The variance is a measure of the variability (spread or dispersion) of
    data. The population variance applies when data represents the entire
    relevant population. If it represents a statistical sample rather than
    the entire population, you should use variance instead.
    """
    n, ss = _SS(data, m)
    if n < 1:
        raise StatsError('population variance or standard deviation'
        ' requires at least one data point')
    return ss/n


def variance(data, m=None):
    """variance(data [, m]) -> sample variance of data.

    >>> variance([0.25, 0.5, 1.25, 1.25,
    ...           1.75, 2.75, 3.5])  #doctest: +ELLIPSIS
    1.37202380952...

    If you know the population mean, or an estimate of it, then you can pass
    the mean as the optional argument m. See also stdev.

    The variance is a measure of the variability (spread or dispersion) of
    data. The sample variance applies when data represents a sample taken
    from the relevant population. If it represents the entire population, you
    should use pvariance instead.
    """
    n, ss = _SS(data, m)
    if n < 2:
        raise StatsError('sample variance or standard deviation'
        ' requires at least two data points')
    return ss/(n-1)


def _SS(data, m):
    """SS = sum of square deviations.
    Helper function for calculating variance directly.
    """
    if m is None:
        # Two pass algorithm.
        data = as_sequence(data)
        m = mean(data)
    return _generalised_sum(data, lambda x: (x-m)**2)


def pstdev(data, m=None):
    """pstdev(data [, m]) -> population standard deviation of data.

    >>> pstdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])  #doctest: +ELLIPSIS
    0.986893273527...

    If you know the true population mean by some other means, then you can
    pass that as the optional argument m:

    >>> pstdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75], 2.875)  #doctest: +ELLIPSIS
    0.986893273527...

    The reliablity of the result as an estimate for the true standard
    deviation depends on the estimate for the mean given. If m is not given,
    or is None, the sample mean of the data will be used.

    If data represents a statistical sample rather than the entire
    population, you should use stdev instead.
    """
    return math.sqrt(pvariance(data, m))


def stdev(data, m=None):
    """stdev(data [, m]) -> sample standard deviation of data.

    >>> stdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])  #doctest: +ELLIPSIS
    1.08108741552...

    If you know the population mean, or an estimate of it, then you can pass
    the mean as the optional argument m:

    >>> stdev([1.5, 2.5, 2.75, 2.75, 3.25, 4.25], 3)  #doctest: +ELLIPSIS
    0.921954445729...

    The reliablity of the result as an estimate for the true standard
    deviation depends on the estimate for the mean given. If m is not given,
    or is None, the sample mean of the data will be used.

    If data represents the entire population, and not just a sample, then
    you should use pstdev instead.
    """
    return math.sqrt(variance(data, m))


def pvariance1(data):
    """pvariance1(data) -> population variance.

    Return an estimate of the population variance for data using one pass
    through the data. Use this when you can only afford a single path over
    the data -- if you can afford multiple passes, pvariance is likely to be
    more accurate.

    >>> pvariance1([0.25, 0.5, 1.25, 1.25,
    ...           1.75, 2.75, 3.5])  #doctest: +ELLIPSIS
    1.17602040816...

    If data represents a statistical sample rather than the entire
    population, then you should use variance1 instead.
    """
    n, s = _welford(data)
    if n < 1:
        raise StatsError('pvariance requires at least one data point')
    return s/n


def variance1(data):
    """variance1(data) -> sample variance.

    Return an estimate of the sample variance for data using a single pass.
    Use this when you can only afford a single path over the data -- if you
    can afford multiple passes, variance is likely to be more accurate.

    >>> variance1([0.25, 0.5, 1.25, 1.25,
    ...           1.75, 2.75, 3.5])  #doctest: +ELLIPSIS
    1.37202380952...

    If data represents the entire population rather than a statistical
    sample, then you should use pvariance1 instead.
    """
    n, s = _welford(data)
    if n < 2:
        raise StatsError('sample variance or standard deviation'
        ' requires at least two data points')
    return s/(n-1)


def _welford(data):
    """Welford's method of calculating the running variance.

    This calculates the second moment M2 = sum( (x-m)**2 ) where m=mean of x.
    Returns (n, M2) where n = number of items.
    """
    # Note: for better results, use this on the residues (x - m) instead of x,
    # where m equals the mean of the data... except that would require two
    # passes, which we're trying to avoid.
    data = iter(data)
    n = 0
    M2 = 0.0  # Current sum of powers of differences from the mean.
    try:
        m = next(data)  # Current estimate of the mean.
        n = 1
    except StopIteration:
        pass
    else:
        for n, x in enumerate(data, 2):
            delta = x - m
            m += delta/n
            M2 += delta*(x - m)  # m here is the new, updated mean.
    assert M2 >= 0.0
    return (n, M2)


def pstdev1(data):
    """pstdev1(data) -> population standard deviation.

    Return an estimate of the population standard deviation for data using
    a single pass. Use this when you can only afford a single path over the
    data -- if you can afford multiple passes, pstdev is likely to be more
    accurate.

    >>> pstdev1([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])  #doctest: +ELLIPSIS
    0.986893273527...

    If data is a statistical sample rather than the entire population, you
    should use stdev1 instead.
    """
    return math.sqrt(pvariance1(data))


def stdev1(data):
    """stdev1(data) -> sample standard deviation.

    Return an estimate of the sample standard deviation for data using
    a single pass. Use this when you can only afford a single path over the
    data -- if you can afford multiple passes, stdev is likely to be more
    accurate.

    >>> stdev1([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])  #doctest: +ELLIPSIS
    1.08108741552...

    If data represents the entire population rather than a statistical
    sample, then use pstdev1 instead.
    """
    return math.sqrt(variance1(data))


def range(data):
    """Return the statistical range of data.

    >>> range([1.0, 3.5, 7.5, 2.0, 0.25])
    7.25

    The range is the difference between the smallest and largest element. It
    is a weak measure of statistical variability.
    """
    try:
        a, b = minmax(data)
    except ValueError as e:
        e.args = ('no range defined for empty iterables',)
        raise
    return b - a


def iqr(data, scheme=None):
    """Returns the Inter-Quartile Range of a sequence of numbers.

    >>> iqr([0.5, 2.25, 3.0, 4.5, 5.5, 6.5])
    3.25

    The IQR is the difference between the first and third quartile. The
    optional argument scheme is used to select the algorithm for calculating
    the quartiles. The default scheme is taken from the global variable
    QUARTILE_DEFAULT. See the quartile function for further details.

    The IQR with scheme 1 is equivalent to Tukey's H-spread.
    """
    q1, _, q3 = quartiles(data, scheme)
    return q3 - q1


def average_deviation(data, m=None):
    """average_deviation(data [, m]) -> average absolute deviation of data.

    data = iterable of data values
    m (optional) = measure of central tendency for data.

    m is usually chosen to be the mean or median, but any measure of central
    tendency is suitable. If m is not given, or is None, the sample mean is
    calculated from the data and used.

    >>> data = [2.0, 2.25, 2.5, 2.5, 3.25]
    >>> average_deviation(data)  # Use the sample mean.
    0.3
    >>> average_deviation(data, 2.75)  # Use the true mean known somehow.
    0.45

    """
    if m is None:
        data = as_sequence(data)
        m = mean(data)
    n, total = _generalised_sum(data, lambda x: abs(x-m))
    if n < 1:
        raise StatsError('average deviation requires at least 1 data point')
    return total/n


def median_average_deviation(data, m=None, sign=0, scale=1):
    """Return the median absolute deviation (MAD) of data.

    The MAD is the median of the absolute deviations from the median, and
    is approximately equivalent to half the IQR.

    >>> median_average_deviation([1, 1, 2, 2, 4, 6, 9])
    1

    Arguments are:

    data    Iterable of data values.
    m       Optional centre location, nominally the median. If m is not
            given, or is None, the median is calculated from data.
    sign    If sign = 0 (the default), the ordinary median is used, otherwise
            either the low-median or high-median are used. See the median()
            function for further details.
    scale   Optional scale factor, by default no scale factor is applied.

    The MAD can be used as a robust estimate for the standard deviation by
    multipying it by a scale factor. The scale factor can be passed directly
    as a numeric value, which is assumed to be positive but no check is
    applied. Other values accepted are:

    'normal'    Apply a scale factor of 1.4826, applicable to data from a
                normally distributed population.
    'uniform'   Apply a scale factor of approximately 1.1547, applicable
                to data from a uniform distribution.
    None, 'none' or not supplied:
                No scale factor is applied (the default).

    The MAD is a more robust measurement of spread than either the IQR or
    standard deviation, and is less affected by outliers. The MAD is also
    defined for distributions such as the Cauchy distribution which don't
    have a mean or standard deviation.
    """
    # Check for an appropriate scale factor.
    if isinstance(scale, str):
        f = median_average_deviation.scaling.get(scale.lower())
        if f is None:
            raise StatsError('unrecognised scale factor `%s`' % scale)
        scale = f
    elif scale is None:
        scale = 1
    if m is None:
        data = as_sequence(data)
        m = median(data, sign)
    med = median((abs(x - m) for x in data), sign)
    return scale*med

median_average_deviation.scaling = {
    # R defaults to the normal scale factor:
    # http://stat.ethz.ch/R-manual/R-devel/library/stats/html/mad.html
    'normal': 1.4826,
    # Wikpedia has a derivation of that constant:
    # http://en.wikipedia.org/wiki/Median_absolute_deviation
    'uniform': math.sqrt(4/3),
    'none': 1,
    }


# Other moments of the data
# -------------------------

def quartile_skewness(q1, q2, q3):
    """Return the quartile skewness coefficient, or Bowley skewness, from
    the three quartiles q1, q2, q3.

    >>> quartile_skewness(1, 2, 5)
    0.5
    >>> quartile_skewness(1, 4, 5)
    -0.5

    """
    if not q1 <= q2 <= q3:
        raise StatsError('quartiles must be ordered q1 <= q2 <= q3')
    if q1 == q2 == q3:
        return float('nan')
    skew = (q3 + q1 - 2*q2)/(q3 - q1)
    assert -1.0 <= skew <= 1.0
    return skew


def pearson_mode_skewness(mean, mode, stdev):
    """Return the Pearson Mode Skewness from the mean, mode and standard
    deviation of a data set.

    >>> pearson_mode_skewness(2.5, 2.25, 2.5)
    0.1

    """
    if stdev > 0:
        return (mean-mode)/stdev
    elif stdev == 0:
        return float('nan') if mode == mean else float('inf')
    else:
        raise StatsError("standard deviation cannot be negative")


def skewness(data, m=None, s=None):
    """skewness(data [,m [,s]]) -> sample skewness of data.

    Returns a biased estimate of the degree to which the data is skewed to
    the left or the right of the mean.

    >>> skewness([1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5])
    ... #doctest: +ELLIPSIS
    1.12521290135...

    If you know one or both of the population mean and standard deviation,
    or estimates of them, then you can pass the mean as optional argument m
    and the standard deviation as s.

    >>> skewness([1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5], m=2.25)
    ... #doctest: +ELLIPSIS
    0.965559535600599...

    The reliablity of the result as an estimate for the true skewness depends
    on the estimated mean and standard deviation. If m or s are not given, or
    are None, they are estimated from the data.

    A negative skewness indicates that the distribution's left-hand tail is
    longer than the tail on the right-hand side, and that the majority of
    the values (including the median) are to the right of the mean. A
    positive skew indicates that the right-hand tail is longer, and that the
    majority of values are to the left of the mean. A zero skew indicates
    that the values are evenly distributed around the mean, often but not
    necessarily implying the distribution is symmetric.

        :: CAUTION ::
        As a rule of thumb, a non-zero value for skewness should only be
        treated as meaningful if its absolute value is larger than
        approximately twice its standard error. See stderrskewness.

    """
    if m is None or s is None:
        data = as_sequence(data)
        if m is None: m = mean(data)
        if s is None: s = stdev(data, m)
    n, total = _generalised_sum(data, lambda x: ((x-m)/s)**3)
    return total/n


def kurtosis(data, m=None, s=None):
    """kurtosis(data [,m [,s]]) -> sample excess kurtosis of data.

    Returns a biased estimate of the excess kurtosis of the data, relative
    to the kurtosis of the normal distribution. To convert to kurtosis proper,
    add 3 to the result.

    >>> kurtosis([1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5])
    ... #doctest: +ELLIPSIS
    -0.1063790369...

    If you know one or both of the population mean and standard deviation,
    or estimates of them, then you can pass the mean as optional argument m
    and the standard deviation as s.

    >>> kurtosis([1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5], m=2.25)
    ... #doctest: +ELLIPSIS
    -0.37265014648437...

    The reliablity of the result as an estimate for the kurtosis depends on
    the estimated mean and standard deviation given. If m or s are not given,
    or are None, they are estimated from the data.

    The kurtosis of a population is a measure of the peakedness and weight
    of the tails. The normal distribution has kurtosis of zero; positive
    kurtosis has heavier tails and a sharper peak than normal; negative
    kurtosis has ligher tails and a flatter peak.

    There is no upper limit for kurtosis, and a lower limit of -2. Higher
    kurtosis means more of the variance is the result of infrequent extreme
    deviations, as opposed to frequent modestly sized deviations.

        :: CAUTION ::
        As a rule of thumb, a non-zero value for kurtosis should only
        be treated as meaningful if its absolute value is larger than
        approximately twice its standard error. See stderrkurtosis.

    """
    if m is None or s is None:
        data = as_sequence(data)
        if m is None: m = mean(data)
        if s is None: s = stdev(data, m)
    n, total = _generalised_sum(data, lambda x: ((x-m)/s)**4)
    k = total/n - 3
    assert k >= -2
    return k

def _generalised_sum(data, func):
    """_generalised_sum(data, func) -> len(data), sum(func(items of data))

    Return a two-tuple of the length of data and the sum of func() of the
    items of data. If func is None, use just the sum of items of data.
    """
    # Try fast path.
    try:
        count = len(data)
    except TypeError:
        # Slow path for iterables without len.
        # We want to support BIG data streams, so avoid converting to a
        # list. Since we need both a count and a sum, we iterate over the
        # items and emulate math.fsum ourselves.
        ap = add_partial
        partials = []
        count = 0
        if func is None:
            # Note: we could check for func is None inside the loop. That
            # is much slower. We could also say func = lambda x: x, which
            # isn't as bad but still costs somewhat.
            for count, x in enumerate(data, 1):
                ap(x, partials)
        else:
            for count, x in enumerate(data, 1):
                ap(func(x), partials)
        total = math.fsum(partials)
    else:
        if func is None:
            # See comment above.
            total = math.fsum(data)
        else:
            total = math.fsum(func(x) for x in data)
    return count, total
    # FIXME this may not be accurate enough for 2nd moments (x-m)**2
    # A more accurate algorithm may be the compensated version:
    #   sum2 = sum(x-m)**2) as above
    #   sumc = sum(x-m)  # Should be zero, but may not be.
    #   total = sum2 - sumc**2/n


def _terriberry(data):
    """Terriberry's algorithm for a single pass estimate of skew and kurtosis.

    This calculates the second, third and fourth moments
        M2 = sum( (x-m)**2 )
        M3 = sum( (x-m)**3 )
        M4 = sum( (x-m)**4 )
    where m = mean of x.

    Returns (n, M2, M3, M4) where n = number of items.
    """
    n = m = M2 = M3 = M4 = 0
    for n, x in enumerate(data, 1):
        delta = x - m
        delta_n = delta/n
        delta_n2 = delta_n*delta_n
        term = delta*delta_n*(n-1)
        m += delta_n
        M4 += term*delta_n2*(n*n - 3*n + 3) + 6*delta_n2*M2 - 4*delta_n*M3
        M3 += term*delta_n*(n-2) - 3*delta_n*M2
        M2 += term
    return (n, M2, M3, M4)
    # skewness = sqrt(n)*M3 / sqrt(M2**3)
    # kurtosis = (n*M4) / (M2*M2) - 3


# === Simple multivariate statistics ===

@_Multivariate.split_xydata
def qcorr(xdata, ydata):
    """Return the Q correlation coefficient of (x, y) data.

    If ydata is None or not given, then xdata must be an iterable of (x, y)
    pairs. Otherwise, both xdata and ydata must be iterables of values, which
    will be truncated to the shorter of the two.

    qcorr(xydata) -> float
    qcorr(xdata, ydata) -> float

    The Q correlation can be found by drawing a scatter graph of the points,
    diving the graph into four quadrants by marking the medians of the X
    and Y values, and then counting the points in each quadrant. Points on
    the median lines are skipped.

    The Q correlation coefficient is +1 in the case of a perfect positive
    correlation (i.e. an increasing linear relationship):

    >>> qcorr([1, 2, 3, 4, 5], [3, 5, 7, 9, 11])
    1.0

    -1 in the case of a perfect anti-correlation (i.e. a decreasing linear
    relationship), and some value between -1 and +1 in nearly all other cases,
    indicating the degree of linear dependence between the variables:

    >>> qcorr([(1, 1), (2, 3), (2, 1), (3, 5), (4, 2), (5, 3), (6, 4)])
    0.5

    In the case where all points are on the median lines, returns a float NAN.
    """
    n = len(xdata)
    assert n == len(ydata)
    if n == 0:
        raise StatsError('Q correlation requires non-empty data')
    xmed = median(xdata)
    ymed = median(ydata)
    # Traditionally, we count the values in each quadrant, but in fact we
    # really only need to count the diagonals: quadrants 1 and 3 together,
    # and quadrants 2 and 4 together.
    quad13 = quad24 = skipped = 0
    for x,y in zip(xdata, ydata):
        if x > xmed:
            if y > ymed:  quad13 += 1
            elif y < ymed:  quad24 += 1
            else:  skipped += 1
        elif x < xmed:
            if y > ymed:  quad24 += 1
            elif y < ymed:  quad13 += 1
            else:  skipped += 1
        else:  skipped += 1
    assert quad13 + quad24 + skipped == n
    if skipped == n:
        return float('nan')
    q = (quad13 - quad24)/(n - skipped)
    assert -1.0 <= q <= 1.0
    return q


@_Multivariate.split_xydata
def corr(xdata, ydata):
    """corr(xydata) -> float
    corr(xdata, ydata) -> float

    Return the sample Pearson's Correlation Coefficient of (x,y) data.

    If ydata is None or not given, then xdata must be an iterable of (x, y)
    pairs. Otherwise, both xdata and ydata must be iterables of values, which
    will be truncated to the shorter of the two.

    >>> corr([(0.1, 2.3), (0.5, 2.7), (1.2, 3.1), (1.7, 2.9)])
    ... #doctest: +ELLIPSIS
    0.827429009335...

    The Pearson correlation is +1 in the case of a perfect positive
    correlation (i.e. an increasing linear relationship), -1 in the case of
    a perfect anti-correlation (i.e. a decreasing linear relationship), and
    some value between -1 and 1 in all other cases, indicating the degree
    of linear dependence between the variables.

    >>> xdata = [1, 2, 3, 4, 5, 6]
    >>> ydata = [2*x for x in xdata]  # Perfect correlation.
    >>> corr(xdata, ydata)
    1.0
    >>> corr(xdata, [5-y for y in ydata])  # Perfect anti-correlation.
    -1.0

    If there are not at least two data points, or if either all the x values
    or all the y values are equal, StatsError is raised.
    """
    n = len(xdata)
    assert n == len(ydata)
    if n < 2:
        raise StatsError(
            'correlation requires at least two data points, got %d' % n)
    # First pass is to determine the means.
    mx = mean(xdata)
    my = mean(ydata)
    # Second pass to determine the standard deviations.
    sx = stdev(xdata, mx)
    sy = stdev(ydata, my)
    if sx == 0:
        raise StatsError('all x values are equal')
    if sy == 0:
        raise StatsError('all y values are equal')
    # Third pass to calculate the correlation coefficient.
    ap = add_partial
    total = []
    for x, y in zip(xdata, ydata):
        term = ((x-mx)/sx) * ((y-my)/sy)
        ap(term, total)
    r = math.fsum(total)/(n-1)
    assert -1 <= r <= r
    return r


def corr1(xydata):
    """corr1(xydata) -> float

    Calculate an estimate of the Pearson's correlation coefficient with
    a single pass over iterable xydata. See also the function corr which may
    be more accurate but requires multiple passes over the data.

    >>> data = zip([0, 5, 4, 9, 8, 4], [1, 2, 4, 8, 6, 3])
    >>> corr1(data)  #doctest: +ELLIPSIS
    0.903737838893...

    xydata must be an iterable of (x, y) points. Raises StatsError if there
    are fewer than two points, or if either of the estimated x and y variances
    are zero.
    """
    xydata = iter(xydata)
    sum_sq_x = 0
    sum_sq_y = 0
    sum_coproduct = 0
    try:
        mean_x, mean_y = next(xydata)
    except StopIteration:
        i = 0
    else:
        i = 1
        for i,(x,y) in zip(itertools.count(2), xydata):
            sweep = (i-1)/i
            delta_x = x - mean_x
            delta_y = y - mean_y
            sum_sq_x += sweep*delta_x**2
            sum_sq_y += sweep*(delta_y**2)
            sum_coproduct += sweep*(delta_x*delta_y)
            mean_x += delta_x/i
            mean_y += delta_y/i
    if i < 2:
        raise StatsError('correlation coefficient requires two or more items')
    pop_sd_x = math.sqrt(sum_sq_x)
    pop_sd_y = math.sqrt(sum_sq_y)
    if pop_sd_x == 0.0:
        raise StatsError('calculated x variance is zero')
    if pop_sd_y == 0.0:
        raise StatsError('calculated y variance is zero')
    r = sum_coproduct/(pop_sd_x*pop_sd_y)
    # r can sometimes exceed the limits -1, 1 by up to 2**-51. We accept
    # that without comment.
    excess = max(abs(r) - 1.0, 0.0)
    if 0 < excess <= 2**-51:
        r = math.copysign(1, r)
    assert -1.0 <= r <= 1.0, "expected -1.0 <= r <= 1.0 but got r = %r" % r
    return r


# Alternate implementation.
def _corr2(xdata, ydata=None):
    raise NotImplementedError('do not use this')
    #t = xysums(xdata, ydata)
    #r = t.Sxy/math.sqrt(t.Sxx*t.Syy)


@_Multivariate.split_xydata
def pcov(xdata, ydata=None):
    """Return the population covariance between (x, y) data.

    >>> pcov([0.75, 1.5, 2.5, 2.75, 2.75], [0.25, 1.1, 2.8, 2.95, 3.25])
    ... #doctest: +ELLIPSIS
    0.93399999999...
    >>> pcov([(0.1, 2.3), (0.5, 2.7), (1.2, 3.1), (1.7, 2.9)])
    0.15125

    """
    n, s = _SP(xdata, None, ydata, None)
    if n > 0:
        return s/n
    else:
        raise StatsError('population covariance requires at least one point')
    #t = xysums(xdata, ydata)
    #return t.Sxy/(t.n**2)


@_Multivariate.split_xydata
def cov(xdata, ydata):
    """Return the sample covariance between (x, y) data.

    >>> cov([(0.1, 2.3), (0.5, 2.7), (1.2, 3.1), (1.7, 2.9)])
    ... #doctest: +ELLIPSIS
    0.201666666666...

    >>> cov([0.75, 1.5, 2.5, 2.75, 2.75], [0.25, 1.1, 2.8, 2.95, 3.25])
    ... #doctest: +ELLIPSIS
    1.1675
    >>> cov([(0.1, 2.3), (0.5, 2.7), (1.2, 3.1), (1.7, 2.9)])
    ... #doctest: +ELLIPSIS
    0.201666666666...

    Covariance reduces down to standard variance when applied to the same
    data as both the x and y values:

    >>> data = [1.2, 0.75, 1.5, 2.45, 1.75]
    >>> cov(data, data)  #doctest: +ELLIPSIS
    0.40325000000...
    >>> variance(data)  #doctest: +ELLIPSIS
    0.40325000000...

    """
    n, s = _SP(xdata, None, ydata, None)
    if n > 1:
        return s/(n-1)
    else:
        raise StatsError('sample covariance requires at least two points')
    # t = xysums(xdata, ydata)
    # return t.Sxy/(t.n*(t.n-1))


def _SP(xdata, mx, ydata, my):
    """SP = sum of product of deviations.
    Helper function for calculating covariance directly.
    """
    if mx is None:
        # Two pass algorithm.
        xdata = as_sequence(xdata)
        mx = mean(xdata)
    if my is None:
        # Two pass algorithm.
        ydata = as_sequence(ydata)
        my = mean(ydata)
    return _generalised_sum(zip(xdata, ydata), lambda t: (t[0]-mx)*(t[1]-my))


@_Multivariate.split_xydata
def errsumsq(xdata, ydata):
    """Return the error sum of squares of (x,y) data.

    The error sum of squares, or residual sum of squares, is the estimated
    variance of the least-squares linear regression line of (x,y) data.

    >>> errsumsq([1, 2, 3, 4], [1.5, 1.5, 3.5, 3.5])
    0.4

    """
    t = xysums(xdata, ydata)
    return (t.Sxx*t.Syy - (t.Sxy**2))/(t.n*(t.n-2)*t.Sxx)


@_Multivariate.split_xydata
def linr(xdata, ydata):
    """Return the linear regression coefficients a and b for (x,y) data.

    Returns the y-intercept and slope of the straight line of the least-
    squared regression line, that is, the line which minimises the sum of
    the squares of the errors between the actual and calculated y values.

    >>> xdata = [0.0, 0.25, 1.25, 1.75, 2.5, 2.75]
    >>> ydata = [1.5*x + 0.25 for x in xdata]
    >>> linr(xdata, ydata)
    (0.25, 1.5)

    """
    t = xysums(xdata, ydata)
    if t.n < 2:
        raise StatsError('regression line requires at least two points')
    b = t.Sxy/t.Sxx
    a = t.sumy/t.n - b*t.sumx/t.n
    return (a, b)


# === Sums and products ===


def sum(data, start=0):
    """Return a full-precision sum of a sequence of numbers.

    >>> sum([2.25, 4.5, -0.5, 1.0])
    7.25

    Due to round-off error, the builtin sum can suffer from catastrophic
    cancellation, e.g. sum([1, 1e100, 1, -1e100] * 10000) returns zero
    instead of the correct value of 20000. This function avoids that error:

    >>> sum([1, 1e100, 1, -1e100] * 10000)
    20000.0

    If optional argument start is given, it is added to the sequence. If the
    sequence is empty, start (defaults to 0) is returned.
    """
    n, total = _generalised_sum(data, None)
    return total + start


def product(data, start=1):
    """Return the product of a sequence of numbers.

    >>> product([1, 2, -3, 2, -1])
    12

    If optional argument start is given, it is multiplied to the sequence.
    If the sequence is empty, start (defaults to 1) is returned.
    """
    # FIXME this doesn't seem to be numerically stable enough.
    return functools.reduce(operator.mul, data, start)
        # Note: do *not* be tempted to do something clever with logarithms:
        # return math.exp(sum([math.log(x) for x in data], start))
        # This is FAR less accurate than the naive multiplication above.


def sumsq(data, start=0):
    """Return the sum of the squares of a sequence of numbers.

    >>> sumsq([2.25, 4.5, -0.5, 1.0])
    26.5625

    If optional argument start is given, it is added to the sequence. If the
    sequence is empty, start (defaults to 0) is returned.
    """
    return sum((x*x for x in data), start)


def cumulative_sum(data, start=None):
    """Iterate over data, yielding the cumulative sums.

    >>> list(cumulative_sum([40, 30, 50, 46, 39, 44]))
    [40.0, 70.0, 120.0, 166.0, 205.0, 249.0]

    Given data [a, b, c, d, ...] the cumulative sum yields the values:
        a, a+b, a+b+c, a+b+c+d, ...

    If optional argument start is given, it must be a number, and it will
    be added to each of the running sums:
        start+a, start+a+b, start+a+b+c, start+a+b+c+d, ...

    """
    it = iter(data)
    ap = add_partial
    if start is not None:
        cs = [start]
    else:
        cs = []
    try:
        x = next(it)
    except StopIteration:
        # Empty data.
        if start is not None:
            yield start
    else:
        ap(x, cs)
        yield math.fsum(cs)
        for x in it:
            ap(x, cs)
            yield math.fsum(cs)


@coroutine
def running_sum(start=None):
    """Running sum co-routine.

    >>> rsum = running_sum()
    >>> rsum.send(1)
    1.0
    >>> rsum.send(2)
    3.0
    >>> rsum.send(3)
    6.0

    Given data [a, b, c, d, ...] the running sum yields the values:
        a, a+b, a+b+c, a+b+c+d, ...

    If optional argument start is given, it must be a number, and it will
    be added to each of the running sums:
        start+a, start+a+b, start+a+b+c, start+a+b+c+d, ...

    >>> rsum = running_sum(9)
    >>> rsum.send(1)
    10.0
    >>> rsum.send(2)
    12.0
    >>> rsum.send(3)
    15.0

    """
    ap = add_partial
    if start is not None:
        total = [start]
    else:
        total = []
    x = (yield None)
    while True:
        ap(x, total)
        x = (yield math.fsum(total))


@_Multivariate.merge_xydata
def Sxx(xydata):
    """Return Sxx = n*sum(x**2) - sum(x)**2 from (x,y) data or x data alone.

    Returns Sxx from either a single iterable or a pair of iterables.

    If given a single iterable argument, it must be either the (x,y) values,
    in which case the y values are ignored, or the x values alone:

    >>> Sxx([(1, 2), (3, 4), (5, 8)])
    24.0
    >>> Sxx([1, 3, 5])
    24.0

    In the two argument form, Sxx(xdata, ydata), the second argument ydata
    is ignored except that the data is truncated at the shorter of the
    two arguments:

    >>> Sxx([1, 3, 5, 7, 9], [2, 4, 8])
    24.0

    """
    n, ss = _SS((x for (x, y) in xydata), None)
    return ss*n


@_Multivariate.merge_xydata
def Syy(xydata):
    """Return Syy = n*sum(y**2) - sum(y)**2 from (x,y) data or y data alone.

    Returns Syy from either a single iterable or a pair of iterables.

    If given a single iterable argument, it must be either the (x,y) values,
    in which case the x values are ignored, or the y values alone:

    >>> Syy([(1, 2), (3, 4), (5, 8)])
    56.0
    >>> Syy([2, 4, 8])
    56.0

    In the two argument form, Syy(xdata, ydata), the first argument xdata
    is ignored except that the data is truncated at the shorter of the
    two arguments:

    >>> Syy([1, 3, 5], [2, 4, 8, 16, 32])
    56.0

    """
    # We expect (x,y) points, but if the caller passed a single iterable
    # ydata as argument, it gets mistaken as xdata with the y values all
    # set to None. (See the merge_xydata function.) We have to detect
    # that and swap the values around.
    try:
        first = next(xydata)
    except StopIteration:
        pass  # Postpone dealing with this.
    else:
        if len(first) == 2 and first[1] is None:
            # Swap the elements around.
            first = (first[1], first[0])
            xydata = ((x, y) for (y, x) in xydata)
        # Re-insert the first element back into the data stream.
        xydata = itertools.chain([first], xydata)
    n, ss = _SS((y for (x, y) in xydata), None)
    return ss*n


@_Multivariate.merge_xydata
def Sxy(xydata):
    """Return Sxy = n*sum(x*y) - sum(x)*sum(y) from (x,y) data.

    Returns Sxy from either a single iterable or a pair of iterables.

    If given a single iterable argument, it must be the (x,y) values:

    >>> Sxy([(1, 2), (3, 4), (5, 8)])
    36.0

    In the two argument form, Sxx(xdata, ydata), data is truncated at the
    shorter of the two arguments:

    >>> Sxy([1, 3, 5, 7, 9], [2, 4, 8])
    36.0

    """
    n = 0
    sumx, sumy, sumxy = [], [], []
    ap = add_partial
    fsum = math.fsum
    for x, y in xydata:
        n += 1
        ap(x, sumx)
        ap(y, sumy)
        ap(x*y, sumxy)
    return n*fsum(sumxy) - fsum(sumx)*fsum(sumy)


def xsums(xdata):
    """Return statistical sums from x data.

    xsums(xdata) -> tuple of sums with named fields

    Returns a named tuple with four fields:

        Name    Description
        ======  ==========================
        n:      number of data items
        sumx:   sum of x values
        sumx2:  sum of x-squared values
        Sxx:    n*(sumx2) - (sumx)**2

    Note that the last field is named with an initial uppercase S, to match
    the standard statistical term.

    >>> tuple(xsums([2.0, 1.5, 4.75]))
    (3, 8.25, 28.8125, 18.375)

    This function calculates all the sums with one pass over the data, and so
    is more efficient than calculating the individual fields one at a time.
    """
    ap = add_partial
    n = 0
    sumx, sumx2 = [], []
    for x in xdata:
        n += 1
        ap(x, sumx)
        ap(x*x, sumx2)
    sumx = math.fsum(sumx)
    sumx2 = math.fsum(sumx2)
    Sxx = n*sumx2 - sumx*sumx
    statsums = collections.namedtuple('statsums', 'n sumx sumx2 Sxx')
    return statsums(*(n, sumx, sumx2, Sxx))


def xysums(xdata, ydata=None):
    """Return statistical sums from x,y data pairs.

    xysums(xdata, ydata) -> tuple of sums with named fields
    xysums(xydata) -> tuple of sums with named fields

    Returns a named tuple with nine fields:

        Name    Description
        ======  ==========================
        n:      number of data items
        sumx:   sum of x values
        sumy:   sum of y values
        sumxy:  sum of x*y values
        sumx2:  sum of x-squared values
        sumy2:  sum of y-squared values
        Sxx:    n*(sumx2) - (sumx)**2
        Syy:    n*(sumy2) - (sumy)**2
        Sxy:    n*(sumxy) - (sumx)*(sumy)

    Note that the last three fields are named with an initial uppercase S,
    to match the standard statistical term.

    This function calculates all the sums with one pass over the data, and so
    is more efficient than calculating the individual fields one at a time.

    If ydata is missing or None, xdata must be an iterable of pairs of numbers
    (x,y). Alternately, both xdata and ydata can be iterables of numbers, which
    will be truncated to the shorter of the two.
    """
    if ydata is None:
        data = xdata
    else:
        data = zip(xdata, ydata)
    ap = add_partial
    n = 0
    sumx, sumy, sumxy, sumx2, sumy2 = [], [], [], [], []
    for x, y in data:
        n += 1
        ap(x, sumx)
        ap(y, sumy)
        ap(x*y, sumxy)
        ap(x*x, sumx2)
        ap(y*y, sumy2)
    sumx = math.fsum(sumx)
    sumy = math.fsum(sumy)
    sumxy = math.fsum(sumxy)
    sumx2 = math.fsum(sumx2)
    sumy2 = math.fsum(sumy2)
    Sxx = n*sumx2 - sumx*sumx
    Syy = n*sumy2 - sumy*sumy
    Sxy = n*sumxy - sumx*sumy
    statsums = collections.namedtuple(
        'statsums', 'n sumx sumy sumxy sumx2 sumy2 Sxx Syy Sxy')
    return statsums(*(n, sumx, sumy, sumxy, sumx2, sumy2, Sxx, Syy, Sxy))


# === Partitioning, sorting and binning ===

def count_elems(data):
    """Count the elements of data, returning a Counter.

    >>> d = count_elems([1.5, 2.5, 1.5, 0.5])
    >>> sorted(d.items())
    [(0.5, 1), (1.5, 2), (2.5, 1)]

    """
    D = {}
    for element in data:
        D[element] = D.get(element, 0) + 1
    return D  #collections.Counter(data)


# === Trimming of data ===

"this section intentionally left blank"

# === Other statistical formulae ===

def sterrmean(s, n, N=None):
    """sterrmean(s, n [, N]) -> standard error of the mean.

    Return the standard error of the mean, optionally with a correction for
    finite population. Arguments given are:

    s: the standard deviation of the sample
    n: the size of the sample
    N (optional): the size of the population, or None

    If the sample size n is larger than (approximately) 5% of the population,
    it is necessary to make a finite population correction. To do so, give
    the argument N, which must be larger than or equal to n.

    >>> sterrmean(2, 16)
    0.5
    >>> sterrmean(2, 16, 21)
    0.25

    """
    if N is not None and N < n:
        raise StatsError('population size must be at least sample size')
    if n < 0:
        raise StatsError('cannot have negative sample size')
    if s < 0.0:
        raise StatsError('cannot have negative standard deviation')
    if n == 0:
        if N == 0: return float('nan')
        else: return float('inf')
    sem = s/math.sqrt(n)
    if N is not None:
        # Finite population correction.
        f = (N - n)/(N - 1)  # FPC squared.
        assert 0 <= f <= 1
        sem *= math.sqrt(f)
    return sem


# Tabachnick and Fidell (1996) appear to be the most commonly quoted
# source for standard error of skewness and kurtosis; see also "Numerical
# Recipes in Pascal", by William H. Press et al (Cambridge University Press).
# Presumably "Numerical Recipes in C" and "... Fortran" by the same authors
# say the same thing.

def stderrskewness(n):
    """stderrskewness(n) -> float

    Return the approximate standard error of skewness for a sample of size
    n taken from an approximately normal distribution.

    >>> stderrskewness(15)  #doctest: +ELLIPSIS
    0.63245553203...

    """
    if n == 0:
        return float('inf')
    return math.sqrt(6/n)


def stderrkurtosis(n):
    """stderrkurtosis(n) -> float

    Return the approximate standard error of kurtosis for a sample of size
    n taken from an approximately normal distribution.

    >>> stderrkurtosis(15)  #doctest: +ELLIPSIS
    1.2649110640...

    """
    if n == 0:
        return float('inf')
    return math.sqrt(24/n)


# === Statistics of circular quantities ===

def circular_mean(data, deg=True):
    """Return the mean of circular quantities such as angles.

    Taking the mean of angles requires some care. Consider the mean of 15
    degrees and 355 degrees. The conventional mean of the two would be 185
    degrees, but a better result would be 5 degrees. This matches the result
    of averaging 15 and -5 degrees, -5 being equivalent to 355.

    >>> circular_mean([15, 355])  #doctest: +ELLIPSIS
    4.9999999999...

    If optional argument deg is a true value (the default), the angles are
    interpreted as degrees, otherwise they are interpreted as radians:

    >>> pi = math.pi
    >>> circular_mean([pi/4, -pi/4], False)
    0.0
    >>> theta = circular_mean([pi/3, 2*pi-pi/6], False)
    >>> theta  # Exact value is pi/12  #doctest: +ELLIPSIS
    0.261799387799...

    """
    ap = add_partial
    if deg:
        data = (math.radians(theta) for theta in data)
    n, cosines, sines = 0, [], []
    for n, theta in enumerate(data, 1):
        ap(math.cos(theta), cosines)
        ap(math.sin(theta), sines)
    if n == 0:
        raise StatsError('circular mean of empty sequence is not defined')
    x = math.fsum(cosines)/n
    y = math.fsum(sines)/n
    theta = math.atan2(y, x)  # Note the order is swapped.
    if deg:
        theta = math.degrees(theta)
    return theta



if __name__ == '__main__':
    import doctest
    doctest.testmod()

