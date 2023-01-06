import numpy as np
import scipy.spatial
class Interval():

    def __init__(self, x, y):
        """
        evaluate the trapz and simpsons error on an equally spaced three
        point interval.

        x, np.array(3,) float:
          the x positions
        y, np.array(3,) float:
          the evaluated function        
        """
        self.x = x
        self.y = y

    def estimate_trapz(self):
        a = self.x[0]
        b = self.x[2]
        fa = self.y[0]
        fb = self.y[2]
        f_mid = self.y[1]
        return (b-a)/2. *(0.5*(fa+fb) + f_mid)

    def estimate_simps(self):
        a = self.x[0]
        b = self.x[2]
        fa = self.y[0]
        fb = self.y[2]
        f_mid = self.y[1]
        return (b-a)/6. * (fa+4.*f_mid+fb)

    def estimate_error(self):
        return  np.abs(self.estimate_trapz()-self.estimate_simps())

    def split(self):
        mid1 = 0.5*(self.x[0]+self.x[1])
        new_x1 = np.array([self.x[0], mid1, self.x[1]])
        new_y1 = np.array([self.y[0], np.nan, self.y[1]])
                          
        mid2 = 0.5*(self.x[1]+self.x[2])
        new_x2 = np.array([self.x[1], mid2, self.x[2]])
        new_y2 = np.array([self.y[1], np.nan, self.y[2]])
        return [Interval(new_x1, new_y1), Interval(new_x2, new_y2)]

    def contains_nan(self):
        return np.any(np.isnan(self.y))
    
    def __repr__(self):
        return "(({},{}), ({},{}), ({},{}))".format(self.x[0], self.y[0], self.x[1], 
                                                    self.y[1], self.x[2], self.y[2])

class UniformSampler():

    def __init__(self, function, x_min, x_max, max_func=101):
        self.function = function
        self.x_min = x_min
        self.x_max = x_max
        self.x_range = x_max-x_min
        self.max_func = max_func

    def evaluate(self):
        self.x = np.linspace(self.x_min, self.x_max, self.max_func)
        self.y = self.function(self.x)

    def get_grid(self):
        return self.x, self.y


class AdaptiveSampler():

    def __init__(self, function, x_min, x_max, tol=1e-3, max_func=100,
                 n_parallel=1):
        self.function = function
        self.x_min = x_min
        self.x_max = x_max
        self.x_range = x_max-x_min
        self.tol = tol
        if max_func < 3:
            raise ValueError("Sampler requires at least 3 function evaluations")
        self.max_func = max_func
        
        self.n_parallel = n_parallel
        #self.init_intervals()
        #self.refine_intervals()

    def _init_seeds(self, seed_positions):
        if seed_positions is not None:
            seed_positions.append(self.x_max)
        else:
            seed_positions=[self.x_max] 
        return seed_positions

    def distribute_points(self, seed_positions, n_points_total):
        n_seeds = len(seed_positions)        
        total_range = self.x_range
        current_left = self.x_min
        point_sets = []
        cumulative_total = 0
        #print("seed positions: {}".format(seed_positions))
        for i_seed in range(n_seeds):
            current_right = seed_positions[i_seed]
            #print("i_seed: {}, current_left: {}, current_right: {}".format(i_seed, current_left, current_right))
            current_range = current_right-current_left
            n_points = np.max([int((current_range/total_range)*n_points_total),3])
            #print("n_points fraction: {}".format((current_range/total_range)*n_points_total))
            n_points = n_points - n_points % 2 + 1
            #print("n_points: {}".format(n_points))
            if i_seed == 0:
                point_set = np.linspace(current_left, current_right, n_points)
            else:
                point_set = np.linspace(current_left, current_right, n_points)[1:]
            cumulative_total += point_set.size
            point_sets.append(point_set)
            current_left = current_right        
        return point_sets, cumulative_total

    def init_intervals(self, seed_positions=None):
        power_of_two = np.floor(np.log2(self.n_parallel-1))
        #print("n_parallel: {}".format(self.n_parallel))
        #print("power of two: {}".format(power_of_two))
        n_initial_points = np.min([int(np.power(2, power_of_two)+1), self.max_func])
        seed_positions = self._init_seeds(seed_positions)
        n_seeds = len(seed_positions)
        n_initial_points = np.max([n_initial_points, 3*(n_seeds)])
        #print("n_intial_points: {}".format(n_initial_points))        
        point_sets, n_initial_points = self.distribute_points(seed_positions, n_initial_points)       
        #print("n_intial_points: {}".format(n_initial_points))       

            
        x_init = np.hstack(point_sets)
        #print("x_init: {}".format(x_init))
        #x_init = np.array([self.x_min, 0.5*(self.x_min+self.x_max),
        #                   self.x_max])
        y_init = self.function(x_init)
        self.intervals = []
        self.errors = []
        for ii in range(0, x_init.size-2, 2):
            #print(ii)
            #print(x_init[ii:ii+3])
            current_interval = Interval(x_init[ii:ii+3], y_init[ii:ii+3])
            self.intervals.append(current_interval)
            self.errors.append(current_interval.estimate_error())
        #self.intervals = [Interval(x_init, y_init)]
        #self.errors = [self.intervals[0].estimate_error()]

    def evaluate_function(self, x):
        return self.function(x)

    def refine_intervals(self):
        n_parallel = self.n_parallel
        n_evaluations = len(self.intervals)*2+1
        #print("n initial evaluations: {}".format(n_evaluations))
        remaining_budget = self.max_func - n_evaluations
        #print("initial remaining budget: {}".format(remaining_budget))
        step = 0
        while remaining_budget > 0:
            error_array = np.array(self.errors)            
            sort_args = np.argsort(error_array)[::-1]
            self.intervals = np.array(self.intervals)[sort_args].tolist()
            new_evaluations = np.min([len(self.intervals)*2, n_parallel, remaining_budget])
            #print("new_evaluations: {}".format(new_evaluations))            
            to_evaluate = []
            for i in range(0, new_evaluations, 2):
                interval = self.intervals.pop(0)
                new_intervals = interval.split()                
                self.intervals.extend(new_intervals)
                to_evaluate.append(new_intervals[0].x[1])
                to_evaluate.append(new_intervals[1].x[1])
            #print("to evaluate: {}".format(to_evaluate))
            y_vals = self.evaluate_function(np.array(to_evaluate))
            ii = 0
            for interval in self.intervals:
                if interval.contains_nan():
                    interval.y[1] = y_vals[ii]
                    ii += 1
            errors = np.zeros(len(self.intervals))
            for i_int, interval in enumerate(self.intervals):
                errors[i_int] = interval.estimate_error()
            self.errors = errors
            step += 1
            n_evaluations += len(to_evaluate)
            remaining_budget -= len(to_evaluate)            
            #print("n_evaluations: {}".format(n_evaluations))
            #print("remaining budget: {}".format(remaining_budget))
            if remaining_budget == 1:
                #print("terminating due to simulation budget")
                remaining_budget = 0
                break
            if np.max(errors) < self.tol:
                #print("terminating due to error tolerance")
                #print(self.tol)
                #print(errors)
                break
                    
    def get_grid(self):
        left = np.zeros(len(self.intervals))
        for ii, interval in enumerate(self.intervals):
            left[ii] = interval.x[0]
        sort_indices = np.argsort(left)        
        self.intervals = np.array(self.intervals)[sort_indices].tolist()
        grid = np.zeros(len(self.intervals)*2+1)
        f_vals = np.zeros(len(self.intervals)*2+1)
        i_grid = 0
        for i_int in range(len(self.intervals)):
            interval = self.intervals[i_int]
            grid[i_grid] = interval.x[0]
            grid[i_grid+1] = interval.x[1]
            
            f_vals[i_grid] = interval.y[0]
            f_vals[i_grid+1] = interval.y[1]
            
            i_grid += 2
        # last point is not shared with another interval so add as a special case
        grid[i_grid] = interval.x[2]
        f_vals[i_grid] = interval.y[2]
        return grid, f_vals    
