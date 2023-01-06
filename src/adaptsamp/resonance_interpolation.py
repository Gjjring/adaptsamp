import numpy as np
from numpy import ma
from run_adaptive_wavelength_scan import run_parameter_scan, adaptive_parameter_scan
import scipy.optimize
import scipy.interpolate
import scipy.signal
import shapely.geometry
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from adaptive_sampling import AdaptiveSampler, UniformSampler
from statisticum.statistics import InformationCriteria, information_criterion

def lorentzian(x, x0, gamma, a):
    return a*(0.5*gamma)/( (x-x0)**2 + (0.5*gamma)**2)

def lorentzian_peak_val(gamma, a):
    return a*0.5*gamma/(0.5*gamma)**2

def lorentzian_a_factor(gamma, peak_val):
    return (2*peak_val/gamma)*(0.5*gamma)**2

def linear(x, m, c):
    return x*m + c

def lorentz_plus_linear(x, x0, gamma, a, m, c):
    lorentz = lorentzian(x, x0, gamma, a)
    line = linear(x, m, c)
    return lorentz + line

def lorentz_plus_linear_peak_val(x0, gamma, a, m, c):
    peak1 = lorentzian_peak_val(gamma, a)
    background = x0*m + c
    return peak1 + background

def lorentz_plus_const(x, x0, gamma, a, c):
    lorentz = lorentzian(x, x0, gamma, a)
    return lorentz + c

def lorentz_plus_const_peak_val(gamma, a, c):
    peak1 = lorentzian_peak_val(gamma, a)
    background = c
    return peak1 + background

def multi_lorentz(x, *args):
    #print("multi_lorentz_args: {}".format(int(len(args)/3)))
    y = np.zeros(x.shape)
    for ipeak in range(int(len(args)/3)):
        x0 = args[ipeak*3]
        gamma = args[ipeak*3+1]
        a = args[ipeak*3+2]
        y += lorentzian(x, x0, gamma, a)
    return y

def multi_lorentz_plus_const(x, *args):
    y = multi_lorentz(x, *args[1:])
    y[y<args[0]] = args[0]
    return y


def multi_lorentz_fixed_center_plus_const(x, centers, *args):
    #print(x.size, centers, args)
    combined_args = []
    for ic, center in enumerate(centers):
        combined_args.append(center)
        combined_args.append(args[1+ic*2])
        combined_args.append(args[2+ic*2])
    #print("combined_args: {}".format(combined_args))
    y = multi_lorentz(x, *combined_args)
    #print(y + args[0])
    return y + args[0]


def fit_lorentzian(x, y, x0=None):
    if x0 is None:
        x0 = x[np.where(np.isclose(y, np.max(y)))][0]
    a = np.max(y)
    gamma = 5*((np.max(x)-np.min(x))/x.size)
    #print(x0, a, gamma)
    parameters = scipy.optimize.curve_fit(lorentzian, x, y,
                                          p0=[x0, gamma, a])
    return parameters

def relax_lorentzian(x, y, x0):
    a = np.max(y)
    gamma = 5*((np.max(x)-np.min(x))/x.size)
    #print(x0, a, gamma)
    fit_func = lambda x, gamma, a : lorentzian(x, x0, gamma, a)
    parameters = scipy.optimize.curve_fit(fit_func, x, y,
                                          p0=[gamma, a])
    return parameters


def fit_lorentzian_plus_linear(x, y):
    x0 = x[np.where(np.isclose(y, np.max(y)))][0]
    a = np.max(y)
    gamma = 5*((np.max(x)-np.min(x))/x.size)
    constant = 0.
    gradient = 0.
    m = 0.
    c = 0.
    #print(x0, a, gamma)
    parameters = scipy.optimize.curve_fit(lorentz_plus_linear, x, y,
                                          p0=[x0, gamma, a, m, c])
    return parameters

def fit_lorentzian_plus_const(x, y):
    x0 = x[np.where(np.isclose(y, np.max(y)))][0]
    a = np.max(y)
    gamma = 5*((np.max(x)-np.min(x))/x.size)
    constant = 0.
    gradient = 0.
    c = 0.
    #print(x0, a, gamma)
    parameters = scipy.optimize.curve_fit(lorentz_plus_const, x, y,
                                          p0=[x0, gamma, a, c])
    return parameters


#def fit_multi_lorentz(x, y, n_peaks=1, x0s=None, gammas=None,
#                      a0s=None):
#
#    c = 0.
#    init_params = []
#    if x0s is not None:
#        assert(len(x0s)==n_peaks)
#
#    x0 = x[np.where(np.isclose(y, np.max(y)))][0]
#    gamma0 = 1*((np.max(x)-np.min(x))/x.size)
#    a0 = np.max(y)
#    c = 0.
#    init_params = [c, x0, gamma0, a0]
#    print(init_params)
#    x_mask = ma.make_mask_none(x.shape)
#    current_x = x0
#    current_gamma = gamma0
#    current_a = a0
#    fit_func = lambda x, args: multi_lorentz_fixed_center_plus_const(x, x0s, *args)
#    parameters = scipy.optimize.curve_fit(fit_func, x, y,
#                                          p0=init_params)
#    return parameters[0]


def relax_multi_lorentz(x, y, n_peaks, x0s, gammas=None,
                        a0s=None):

    c = 0.
    init_params = [c]
    bounds = [[0.], [np.inf]]
    assert(len(x0s)==n_peaks)
    for i_peak in range(n_peaks):
        x0 = x0s[i_peak]
        init_params.append(x0)
        bounds[0].append(x0-5)
        bounds[1].append(x0+5)
        if gammas is not None:
            gamma = gammas[i_peak]
        else:
            gamma = (np.max(x)-np.min(x))/x.size
        init_params.append(gamma)
        bounds[0].append(gamma*0.5)
        bounds[1].append(gamma*5)
        if a0s is not None:
            a0 = a0s[i_peak]
        else:
            a0 = np.max(y)
        init_params.append(a0)
        bounds[0].append(a0*0.5)
        bounds[1].append(a0*5)

    print("init_params: {}".format(init_params))    
    #fit_func = lambda x, *args: multi_lorentz_fixed_center_plus_const(x, x0s, *args)
    fit_func = lambda x, *args: multi_lorentz_plus_const(x, *args)
    parameters = scipy.optimize.curve_fit(fit_func,
                                          x, y,
                                          p0=init_params,
                                          bounds=bounds,
                                          maxfev=(500*(x.size+1)))
    return parameters[0]



class ResonanceFinder():

    def __init__(self, function, keys, wvl_range, width_range):
        self.function = function
        self.keys = keys
        self.wvl_range = wvl_range
        self.width_range = width_range
        self.mode = 'adaptive'
        self.param_step_manager = ParameterStepManager(width_range)
        self.create_boundary()
        self.max_peaks = 1
        self.peak_shift_tolerance = 30.


    def create_boundary(self):
        p0 = (self.wvl_range[1], self.width_range[0])
        p1 = (self.wvl_range[1], self.width_range[1])
        p2 = (self.wvl_range[0], self.width_range[1])
        p3 = (self.wvl_range[0], self.width_range[0])
        self.boundary = shapely.geometry.LinearRing([p0, p1, p2, p3])
        #poly = shapely.geometry.Polygon(self.boundary)
        #patch = PolygonPatch(poly, fill=False, ec='k')
        #plt.gca().add_patch(patch)

    def create_diagonal_space(self):
        top_left = np.array([self.wvl_range[0], self.width_range[1]])
        bottom_right = np.array([self.wvl_range[1], self.width_range[0]])
        m = (bottom_right[1]-top_left[1])/(bottom_right[0]-top_left[0])
        c = top_left[1] + (0-top_left[0])*m
        print(m, c)

    def sample_diagonal_space(self, x):
        top_left = np.array([self.wvl_range[0], self.width_range[1]])
        bottom_right = np.array([self.wvl_range[1], self.width_range[0]])
        m = (bottom_right[1]-top_left[1])/(bottom_right[0]-top_left[0])
        c = top_left[1] + (0-top_left[0])*m
        wvl = self.wvl_range[0] + (self.wvl_range[1]-self.wvl_range[0])*x
        width = c + m*wvl
        return wvl, width

    def create_const_model(self, x, value):
        self.models['constant'] = PolynomialModel([x], [value], max_order=1)
        self.models['constant'].fit_params()

    def create_lorentz_model(self, x, params, number):
        name = "lorentz_{}".format(number)
        self.models[name] = {}
        self.models[name]['peak_pos'] = PolynomialModel([x], [params[0]], max_order=2)
        self.models[name]['gamma'] = PolynomialModel([x], [params[1]], max_order=1)
        self.models[name]['a'] = PolynomialModel([x], [params[2]], max_order=2)
        self.models[name]['peak_pos'].fit_params()
        self.models[name]['gamma'].fit_params()
        self.models[name]['a'].fit_params()

    def create_models(self, x, parameters):
        self.models = {}
        parameters = parameters.tolist()
        const = parameters.pop(0)
        self.create_const_model(x, const)
        n_peaks = int(len(parameters)/3)
        for i_peak in range(n_peaks):
            params = parameters[i_peak*3:i_peak*3+3]
            self.create_lorentz_model(x, params, i_peak)



    def append_models(self, x, parameters):
        assert(len(parameters)==3)
        n_lorentz = len(self.models)-1
        self.create_lorentz_model(x, parameters, n_lorentz+1)


    def find_intial_resonances(self):
        width = self.param_step_manager.get_suggestion()
        sliced_function = lambda x: self.function(x, width)
        if self.mode == 'adaptive':
            sampler = AdaptiveSampler(sliced_function, self.wvl_range[0],
                                       self.wvl_range[1],
                                       tol=self.keys['function_error_tolerance'],
                                       max_func=self.keys['max_function_calls'],
                                       n_parallel=self.keys['Multiplicity'])
            sampler.init_intervals()
            sampler.refine_intervals()
        elif self.mode=='uniform':
            sampler = UniformSampler(sliced_function, self.wvl_range[0],
                                     self.wvl_range[1],
                                     max_func=self.keys['max_function_calls'])
            sampler.evaluate()
        x, f = sampler.get_grid()
        y = width*np.ones(x.size)
        n_peaks, params = self.fit_model(x, f)

        self.create_models(width, params)
        self.param_step_manager.add_evaluation(width)
        return x, y, f, n_peaks, params

    def update_models(self, x, parameters):
        n_peak = 0
        parameters = parameters.tolist()
        self.models['constant'].add_data(x, parameters.pop(0))
        n_peaks = int((len(parameters)/3))
        for param_set in range(n_peaks):
            peak_pos = parameters[param_set*3]
            gamma = parameters[param_set*3+1]
            a = parameters[param_set*3+2]
            model_number = None
            for model_key in self.models.keys():
                if model_key == 'constant':
                    continue
                model = self.models[model_key]
                param_name, number = model_key.split("_")
                if param_name == "lorentz":
                    if np.abs(model['peak_pos'].evaluate(x)-peak_pos) < self.peak_shift_tolerance:
                        model_number = number
            if model_number is not None:
                model = self.models['lorentz_{}'.format(model_number)]
                model["peak_pos"].add_data(x, peak_pos)
                model["gamma"].add_data(x, gamma)
                model["a"].add_data(x, a)
            else:
                self.append_models(x, [peak_pos, gamma, a])
                self.param_step_manager.reset_step_size()



    def fit_resonances(self):
        range_covered = False
        x_vals = []
        y_vals = []
        f_vals = []
        step = 0
        while not range_covered:
            width = self.param_step_manager.get_suggestion()
            #print(step, width)
            #if step == 3:
            #    break
            sliced_function = lambda x: self.function(x, width)
            #predict_peak_pos = self.models['peakpos_0'].evaluate(width)
            #predict_peak_width = np.max([self.models['gamma_0'].evaluate(width), 5.])
            #for model in self.models:
            #print(step, width, predict_peak_pos, predict_peak_width)
            #wvl_lower = predict_peak_pos - 1.5*predict_peak_width
            #wvl_upper = predict_peak_pos + 1.5*predict_peak_width
            #print(wvl_lower, wvl_upper)
            wvl_lower =self.wvl_range[0]
            wvl_upper =self.wvl_range[1]
            if self.mode == 'adaptive':
                sampler = AdaptiveSampler(sliced_function, wvl_lower,
                                           wvl_upper,
                                           tol=self.keys['function_error_tolerance'],
                                           max_func=self.keys['max_function_calls'],
                                           n_parallel=self.keys['Multiplicity'])
                seeds = []
                for model_key in self.models.keys():
                    if model_key == 'constant':
                        continue

                    estimate = self.models[model_key]['peak_pos'].evaluate(width)
                    if estimate >= wvl_lower and estimate <= wvl_upper:
                        seeds.append(estimate)
                #print("seed positions: {}".format(seeds))
                seeds = np.sort(np.array(seeds)).tolist()
                sampler.init_intervals(seed_positions=seeds)
                sampler.refine_intervals()
            elif self.mode == 'uniform':
                sampler = UniformSampler(sliced_function, self.wvl_range[0],
                                         self.wvl_range[1],
                                         max_func=self.keys['max_function_calls'])
                sampler.evaluate()
            x, f = sampler.get_grid()
            #print(x, f)
            y = width*np.ones(x.size)
            x_vals.append(x)
            y_vals.append(y)
            f_vals.append(f)
            n_peaks, params = self.fit_model(x, f)
            #print(step, width, params)
            self.update_models(width, params)
            self.param_step_manager.add_evaluation(width)
            range_covered = self.param_step_manager.is_range_covered()
            step += 1
        return x_vals, y_vals, f_vals


    def fit_model(self, x, f):
        #x, y = self.sample_diagonal_space(t)
        #average_step_size = np.mean(np.diff(x))
        peaks = scipy.signal.find_peaks(f, prominence=self.keys['peak_prominence'],
                                        height=(None, None))
        n_peaks = len(peaks[0])
        prominences = scipy.signal.peak_prominences(f, peaks[0])
        widths = scipy.signal.peak_widths(f, peaks[0],
                                          prominence_data=prominences)
        peak_x_vals = []
        gamma_vals = []
        a_vals = []
        heights = peaks[1]['peak_heights']
        #print("average_step_size: {}".format(average_step_size))
        #print("peaks: {}".format(peaks[0]))
        #print("heights: {}".format(peaks[1]['peak_heights']))
        #print("widths: {}".format(widths[0]*average_step_size))
        for i_peak, peak in enumerate(peaks[0]):
            #print("peak: {}, peak_start: {}".format(peak, x[peak]))
            #params = relax_lorentzian(x, f, x[peak])
            peak_x_vals.append(x[peak])
            local_step_size = np.diff(x)[peak]
            gamma = widths[0][i_peak]*local_step_size
            gamma_vals.append(gamma)
            peak = heights[i_peak]
            a_vals.append(lorentzian_a_factor(gamma, peak))
        #print("peak_x_vals: {}".format(peak_x_vals))
        #print("gammas: {}".format(gamma_vals))
        #print("a_vals: {}".format(a_vals))
        params = relax_multi_lorentz(x, f, n_peaks, peak_x_vals,
                                     gammas=gamma_vals, a0s=a_vals)
        #print("resulting params: {}".format(params))
        full_params = params
        #full_params = [params[0]]
        #for i_peak in range(n_peaks):
        #    full_params.append(peak_x_vals[i_peak])
        #    full_params.append(params[1+i_peak*2])
        #    full_params.append(params[2+i_peak*2])
        return n_peaks, full_params


class PolynomialModel():

    def __init__(self, x, y, order=0, max_order=100):
        self.order= order
        self.x = x
        self.y = y
        self.max_order = max_order

    def fit_params(self):
        self.params = np.polyfit(self.x, self.y, self.order)

    def update_data(self, x, y):
        sort_indices = np.argsort(np.array(x))
        self.x = np.array(x)[sort_indices].tolist()
        self.y = np.array(y)[sort_indices].tolist()
        self.order = np.min([len(self.x)-1, self.max_order])
        self.fit_params()

    def add_data(self, x, y):
        self.x.append(x)
        self.y.append(y)
        self.update_data(self.x, self.y)

    def evaluate(self, x):
        fit = np.zeros(x.shape)
        for ip, param in enumerate(self.params[::-1]):
            fit += param*np.power(x, ip)
        return fit





class ParameterStepManager():

    def __init__(self, parameter_range, min_steps=3):
        self.upper = parameter_range[1]
        self.lower = parameter_range[0]
        self.step_size = self.min_step_size()
        self.evaluations = []
        self.direction = 1.
        self.step_size_growth_factor = 3
        self.min_steps = min_steps
        self.step_size_max = 60.
        self.mode = 'adaptive'

    def min_step_size(self):
        return (self.upper-self.lower)/20.

    def scale_step_size(self, factor):
        if self.mode == 'adaptive':
            self.step_size = np.clip(self.step_size*factor,   self.min_step_size(), self.step_size_max)

    def reset_step_size(self):
        if self.mode == 'adaptive':
            self.step_size = self.min_step_size()

    def add_evaluation(self, evaluation):
        self.evaluations.append(evaluation)
        if np.isclose(evaluation, self.upper):
            self.direction = -1.
            self.last_evaluation = self.evaluations[0]
            #self.scale_step_size(1./4.)
        elif np.isclose(evaluation, self.lower):
            self.direction = 1.
            self.last_evaluation = self.evaluations[0]
        else:
            self.last_evaluation = evaluation
        #self.evaluations = np.sort(np.array(self.evaluations)).tolist()

    def get_suggestion(self):
        if self.mode == 'adaptive':
            return self.get_suggestion_adaptive()
        elif self.mode == 'uniform':
            return self.get_suggestion_uniform()
        else:
            raise ValueError("unknown mode: {}".format(self.mode))

    def get_suggestion_adaptive(self):
        if len(self.evaluations) == 0:
            new_pos = 0.5*(self.lower+self.upper)
        else:
            new_pos = np.clip(self.last_evaluation +self.direction*self.step_size,
                              self.lower, self.upper)
            self.scale_step_size(2.)
        return new_pos


    def get_suggestion_uniform(self):
        if len(self.evaluations) == 0:
            new_pos = self.lower
        else:
            new_pos = np.clip(self.last_evaluation +self.step_size,
                              self.lower, self.upper)
        return new_pos

    def is_range_covered(self):
        min_steps_reached = len(self.evaluations) >= self.min_steps
        upper_in_range = np.any(np.isclose(np.array(self.evaluations),self.upper))
        lower_in_range = np.any(np.isclose(np.array(self.evaluations),self.lower))
        return min_steps_reached and upper_in_range and lower_in_range





def interpolate_resonance_parameters(function, width_range, wvl_range, keys):
    adaptive, params, initial_width = find_initial_resonance(function, keys, width_range, wvl_range)

    evaluated_widths = [initial_width]
    evaluated_wvls = [params[0]]
    evaluated_fwhms = [params[1]]
    evaluated_peaks = [lorentzian_peak_val(params[1], params[2])]

    peak_pos_model = PolynomialModel(evaluated_widths, evaluated_wvls, order=0)
    peak_fwhm_model = PolynomialModel(evaluated_widths, evaluated_fwhms, order=0)
    peak_value_model = PolynomialModel(evaluated_widths, evaluated_peaks, order=0)

    parameter_space_covered = False
    parameter_step_man = ParameterStepManager(width_range)
    parameter_step_man.add_evaluation(initial_width)
    while not parameter_space_covered:
        new_width = parameter_step_man.get_suggestion()

        peak_wvl = peak_pos_model.evaluate(new_width)
        peak_width = peak_fwhm_model.evaluate(new_width)
        current_wvl_range = [peak_wvl-peak_width, peak_wvl+peak_width]
        peak_present, params = find_resonance(function, keys, new_width, current_wvl_range)
        if peak_present:
            parameter_step_max.add_evaluation(new_width)
            evaluated_widths.append(new_width)
            evaluated_wvls.append(params[0])
            peak_pos_model.update_data(evaluated_widths, evaluated_wvls)
            evaluated_fwhms.append(params[1])
            peak_fwhm_model.update_data(evaluated_widths, evaluated_fwhms)
            evaluated_peaks.append(params[2])
            peak_value_model.update_data(evaluated_widths, evaluated_peaks)

            crossing_points = crossing(wdith_range, wvl_range, peak_pos_model)


            parameter_space_covered = parameter_step_max.is_range_covered()
        else:
            parameter_step_man.step_size /= 2.

    return peak_pos_model, peak_fwhm_model, peak_value_model












def interpolate_resonance_parameters(function, width_range, wvl_range, height):
    colors = matplotlib.cm.tab10(range(10))
    initial_width = (width_range[1]+width_range[0])*0.5
    wvl_step = 1.
    wvls = np.arange(wvl_range[0], wvl_range[1]+wvl_step, wvl_step)
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.5, hspace=0.5, width_ratios=[1.1, 1., 1.])

    ax0 = fig.add_subplot(gs[:2, :2])
    ax1 = fig.add_subplot(gs[2, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])
    ax4 = fig.add_subplot(gs[2, 2])

    plt.sca(ax0)
    width_step = 1.
    widths = np.arange(width_range[0], width_range[1]+width_step, width_step)
    X, Y = np.meshgrid(wavelengths, widths)
    Z = function(X, Y, height)
    plt.pcolormesh(X,Y,Z, shading='gouraud')
    plt.colorbar()
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Width (nm)")
    plt.xlim(wvl_range)
    plt.ylim([width_range[0]-5, width_range[1]+5])
    adaptive, params = find_resonance_wavelength(initial_width, 600, 780, height)

    plt.sca(ax1)
    x, y = adaptive.get_grid()
    total_grid_size = x.size
    fit = lorentzian(wvls, params[0], params[1], params[2])
    plt.plot(wvls, fit, color=colors[0])

    evaluated_widths = [initial_width]
    evaluated_wvls = [params[0]]
    evaluated_fwhms = [params[1]]
    evaluated_peaks = [lorentzian_peak_val(params[1], params[2])]

    plt.scatter(x, y, color=colors[0])
    plt.xlim(wvl_range)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")

    plt.sca(ax0)
    plt.scatter(res_wvl, initial_width, label='Resonance 0', color=colors[0])
    plt.axhline(initial_width, alpha=0.6, color=colors[0])

    plt.sca(ax2)
    plt.scatter(initial_width, res_wvl, color=colors[0])
    plt.xlim([width_range[0]-5, width_range[1]+5])
    #plt.xlim(width_range)
    plt.ylim(wvl_range)
    plt.xlabel("Width (nm)")
    plt.ylabel("Peak\nWavelength (nm)")

    plt.sca(ax3)
    plt.scatter(initial_width, params[1], color=colors[0])
    #plt.xlim(width_range)
    plt.xlim([width_range[0]-5, width_range[1]+5])
    #plt.ylim(wvl_range)
    plt.xlabel("Width (nm)")
    plt.ylabel("Peak\nWidth (nm)")
    plt.ylim([0, 15.])

    plt.sca(ax4)
    plt.scatter(initial_width, lorentzian_peak_val(params[1], params[2]), color=colors[0])
    #plt.xlim(width_range)
    plt.xlim([width_range[0]-5, width_range[1]+5])
    #plt.ylim(wvl_range)
    plt.xlabel("Width (nm)")
    plt.ylabel("Peak\nIntensity")
    plt.ylim([0, 2.])

    wvl_zoom_width = 20
    width_step_distance = 5
    current_res_wavelength = res_wvl
    direction = 1.
    current_width = initial_width + direction*width_step_distance
    peak_pos_model = PolynomialModel(evaluated_widths, evaluated_wvls)
    peak_fwhm_model = PolynomialModel(evaluated_widths, evaluated_fwhms)
    peak_intensity_model = PolynomialModel(evaluated_widths, evaluated_peaks)

    #plt.savefig(os.path.join("figures", "2D_sampling2_{}.png".format(0)),
    #               dpi=300, bbox_inches='tight')

    wvl_peak_guess = res_wvl
    stop_after_next = False
    for step in range(8):
        #print(step)
        wvl_min = wvl_peak_guess-wvl_zoom_width*0.5
        wvl_max = wvl_peak_guess+wvl_zoom_width*0.5
        adaptive, params = find_resonance_wavelength(current_width, wvl_min, wvl_max, height)

        plt.sca(ax1)
        x, y = adaptive.get_grid()
        total_grid_size += x.size
        reduced_wvls = np.arange(wvl_min, wvl_max+wvl_step, wvl_step)
        fit = lorentzian(reduced_wvls, params[0], params[1], params[2])
        plt.plot(reduced_wvls, fit, color=colors[step+1])
        plt.scatter(x, y, color=colors[step+1])
        plt.sca(ax0)
        plt.scatter(params[0], current_width, color=colors[step+1])
        #print(current_width, wvl_min, wvl_max)
        plt.hlines(current_width, wvl_min, wvl_max, color=colors[step+1])

        evaluated_widths.append(current_width)
        evaluated_wvls.append(params[0])
        evaluated_fwhms.append(params[1])
        evaluated_peaks.append(lorentzian_peak_val(params[1], params[2]))

        peak_pos_model.update_data(evaluated_widths, evaluated_wvls)
        peak_fwhm_model.update_data(evaluated_widths, evaluated_fwhms)
        peak_intensity_model.update_data(evaluated_widths, evaluated_peaks)
        if step > 0:
            peak_pos_model.order = 2
            peak_fwhm_model.order = 2
            peak_intensity_model.order =2

        #print(peak_pos_model.x, peak_pos_model.y)
        peak_pos_model.fit_params()
        #print(peak_fwhm_model.x, peak_fwhm_model.y)
        peak_fwhm_model.fit_params()
        peak_intensity_model.fit_params()

        #peak_pos_model.x = evaluated_widths
        #peak_pos_model.y = evaluated_wvls

        plt.sca(ax2)
        plt.scatter(current_width, params[0], color=colors[step+1])

        wvl_fit = peak_pos_model.evaluate(widths)
        if step == 0:
            model_plot_handle1 = plt.plot(widths, wvl_fit, color='k', zorder=-1)[0]
        else:
            model_plot_handle1.set_ydata(wvl_fit)
        plt.sca(ax0)
        if step == 0:
            model_plot_handle2 = plt.plot(wvl_fit, widths, color='r')[0]
        else:
            model_plot_handle2.set_xdata(wvl_fit)

        plt.sca(ax3)
        plt.scatter(current_width, params[1], color=colors[step+1])

        fwhm_fit = peak_fwhm_model.evaluate(widths)
        if step == 0:
            model_plot_handle3 = plt.plot(widths, fwhm_fit, color='k', zorder=-1)[0]
        else:
            model_plot_handle3.set_ydata(fwhm_fit)


        plt.sca(ax4)
        plt.scatter(current_width, lorentzian_peak_val(params[1], params[2]), color=colors[step+1])

        peak_fit = peak_intensity_model.evaluate(widths)
        if step == 0:
            model_plot_handle4 = plt.plot(widths, peak_fit, color='k', zorder=-1)[0]
        else:
            model_plot_handle4.set_ydata(peak_fit)


        width_step_distance *= 2
        if direction >0:
            current_width = np.min([current_width+direction*width_step_distance, width_range[1]])
        else:
            current_width = np.max([current_width+direction*width_step_distance, width_range[0]])
        wvl_peak_guess = peak_pos_model.evaluate(np.array([current_width]))[0]

        #plt.savefig(os.path.join("figures", "2D_sampling2_{}.png".format(step+1)),
        #           dpi=300, bbox_inches='tight')

        if stop_after_next:
            print("total grid size: {}".format(total_grid_size))
            return peak_pos_model, peak_fwhm_model, peak_intensity_model
        if np.isclose(current_width, width_range[1]):
            width_step_dsitance = 10
            direction = -1
        elif np.isclose(current_width, width_range[0]):
            stop_after_next = True

if __name__ == "__main__":
    width_range = [120, 230]
    wvl_range = [600, 780]
    height = 250.
    models = interpolate_resonance_parameters(test_function, width_range, wvl_range, height)
