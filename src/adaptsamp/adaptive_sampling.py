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

class Triangle():
    
    def __init__(self, x, y):
        """
        x, np.array(3,) float:
          the x positions
        y, np.array(3,) float:
          the y positions
        """
        self.x = x
        self.y = y

    def area(self):
        edge0 = np.array([self.x[1]-self.x[0], self.y[1]-self.y[0]])
        edge1 = np.array([self.x[2]-self.x[0], self.y[2]-self.y[0]])
        area = abs( edge0[0] * edge1[1] - edge0[1] * edge1[0] ) / 2
        return area

    def mid_points(self):
        mids = np.zeros((3,2))
        mids[0,:] = np.array([np.mean(self.x[[0, 1]]), np.mean(self.y[[0, 1]])])
        mids[1,:] = np.array([np.mean(self.x[[1, 2]]), np.mean(self.y[[1, 2]])])
        mids[2,:] = np.array([np.mean(self.x[[0, 2]]), np.mean(self.y[[0, 2]])])
        return mids

    def central_point(self):
        center = np.array([np.mean(self.x), np.mean(self.y)])
        return center

    def contains(self, point):
        trial_x = point[0]
        trial_y = point[1]
        area0 = self.area()
        areas = []
        for i in range(3):
            x_sub = np.array([trial_x, self.x[i], self.x[(i+1)%3]])
            y_sub = np.array([trial_y, self.y[i], self.y[(i+1)%3]])
            areas.append(Triangle(x_sub, y_sub).area())
        return np.isclose(area0,np.sum(np.array(areas)))
    
class Interval2D():

    def __init__(self, x, y, f):
        """
        evaluate the trapz and simpsons error on a seven point triangle

        x, np.array(7,) float:
          the x positions
        y, np.array(7,) float:
          the y positions
        f, np.array(7,) float:
          the evaluated function values

        2
        ##
        ####
        #######
        ######### 4
        5 ###########
        ####### 6 #####
        ##################
        #####################
        0 ######## 3 ########## 1
        """
        self.x = x
        self.y = y
        self.f = f
        self.hash_points()
        #self.triangles = self.triangulate()

    @classmethod
    def from_outer_triangle(interval, x, y, f):
        points = np.vstack([x, y]).T
        t = Triangle(points[:,0], points[:,1])
        mid_points =t.mid_points()
        center = t.central_point()
        all_points = np.vstack([points, mid_points, center])
        f = np.concatenate([f, np.full(4, np.nan)])
        return interval(all_points[:,0], all_points[:,1], f)

    def contains(self, point):
        vertex_indices = self.big_triangle()
        x = self.x[vertex_indices[0,:]]
        y = self.y[vertex_indices[0,:]]
        return Triangle(x, y).contains(point)

    def approximate_linear(self, point):
        
            

    def small_triangles(self):
        return np.array([[0, 3, 5],
                         [3, 1, 4],
                         [3, 4, 5],
                         [5, 4, 2]])

    def seven_point_weights(self):
        return np.array([3., 3., 3., 8., 8., 8., 27.])/60.

    def big_triangle(self):
        return np.array([[0, 1, 2]])

    def hash_points(self):
        hash_map = {}
        for row in range(self.x.size):
            point = np.array([self.x[row], self.y[row]])
            str_val = "{}".format(point.tolist())
            hash_map[hash(str_val)] = row
        self.hash_map = hash_map

    def estimate_three_point(self):
        estimate = 0.
        triangles = self.small_triangles()
        for row in range(triangles.shape[0]):
            t_index = triangles[row, :]
            triangle = Triangle(self.x[t_index], self.y[t_index])            
            area = triangle.area()
            estimate += np.sum(self.f[t_index])/3. *area
        return estimate

    def estimate_seven_point(self):
        estimate = 0.
        t_index = self.big_triangle()[0,:]
        triangle = Triangle(self.x[t_index], self.y[t_index])            
        area = triangle.area()
        weights = self.seven_point_weights()
        for row in range(self.x.shape[0]):            
            estimate += self.f[row]*weights[row]*area
        return estimate

    def estimate_error(self):
        return  np.abs(self.estimate_seven_point()-self.estimate_three_point())

    def split(self):
        intervals = []
        triangles = self.small_triangles()
        for row in range(triangles.shape[0]):
            t_index = triangles[row, :]
            x = self.x[t_index]
            y = self.y[t_index]
            f = self.f[t_index]            
            intervals.append(Interval2D.from_outer_triangle(x, y, f))            
            
        return intervals

    def contains_nan(self):
        return np.any(np.isnan(self.f))
    
    def __repr__(self):
        string = "("
        for row in range(self.x.shape[0]):
            string += "({},{},{}), ".format(self.x[row], self.y[row], self.f[row])
        string = string[:-2] + ")"            
        return string
    
    
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
                             
        

class AdaptiveSampler2D():

    def __init__(self, function, x_min, x_max, y_min, y_max, tol=1e-3, max_func=100,
                 n_parallel=1):
        self.function = function                             
        self.x_min = x_min
        self.x_max = x_max                             
        self.x_range = x_max-x_min
        self.y_min = y_min
        self.y_max = y_max
        self.y_range = y_max-y_min
        self.tol = tol
        if max_func < 11:
            raise ValueError("Sampler requires at least 11 function evaluations")
        self.max_func = max_func
        
        self.n_parallel = n_parallel
        self.intervals = []
        self.n_evaluations = 0
        self.global_hash_map = {}
        self.point_array = np.empty((0, 2))
        self.f = np.empty((0,))
        #self.init_intervals()
        #self.refine_intervals()

    def get_unique_points(self, intervals):
        hashed_points = {}
        for interval in intervals:
            for hash_val, row in interval.hash_map.items():
                hashed_points[hash_val] = np.array([interval.x[row], interval.y[row]])
        #for hash_val, point in hashed_points.items():
        #    print(hash_val, point)
        points = []
        hash_list = []
        for hash_val, point in hashed_points.items():
            if hash_val not in self.global_hash_map:
                points.append(point[:2])
                hash_list.append(hash_val)
        if len(points) > 0:
            points = np.array(points)
        else:
            points = np.empty((0, 2))
        return hash_list, points

    def update_global_maps(self, hash_list, points, f_vals):
        for hash_val in hash_list:
            self.global_hash_map[hash_val] = len(self.global_hash_map)
        self.point_array = np.vstack([self.point_array, points])

        self.f = np.hstack([self.f, f_vals])

    def update_intervals(self, intervals, hash_list, f_vals):
        #print("intervals: {}".format(len(intervals)))
        #print("hash_list: {}".format(hash_list))
        #print("f_vals: {}".format(f_vals))
        
        for interval in intervals:
            #print(interval.hash_map)
            for hash_val, interval_row in interval.hash_map.items():
                if np.isnan(interval.f[interval_row]):
                    global_row = self.global_hash_map[hash_val]
                    interval.f[interval_row] = self.f[global_row]
            #print(interval.f)        

    def init_intervals(self):
        points = np.array([[self.x_min, self.y_min],
                           [self.x_max, self.y_min],
                           [self.x_min, self.y_max]])
        f = np.full(3, np.nan)
        interval0 = Interval2D.from_outer_triangle(points[:,0], points[:,1], f)

        points = np.array([[self.x_max, self.y_min],
                           [self.x_max, self.y_max],
                           [self.x_min, self.y_max]])
        interval1 = Interval2D.from_outer_triangle(points[:,0], points[:,1], f)

        n_initial_points = 7*2 - 3 #3 points are shared between the two

        hash_list, init_points = self.get_unique_points([interval0, interval1])
        
        #print(hash_list)
        #print(init_points.shape)        
        f = self.evaluate_function(init_points[:,0], init_points[:,1])        
        #print(f)
        #print(f.shape)
        self.update_global_maps(hash_list, init_points, f)
        
        
        self.n_evaluations += f.shape[0]
        self.intervals = []
        self.errors = []

                
        self.update_intervals([interval0, interval1], hash_list, f)
        for interval in [interval0, interval1]:
            self.intervals.append(interval)
            self.errors.append(interval.estimate_error())

        #self.intervals = [Interval(x_init, y_init)]
        #self.errors = [self.intervals[0].estimate_error()]

    def evaluate_function(self, x, y):
        return self.function(x, y)

    def refine_intervals(self):
        n_parallel = self.n_parallel
        self.get_grid()
        n_evaluations = len(self._all_hashed_vals)
        #print("n initial evaluations: {}".format(n_evaluations))
        remaining_budget = self.max_func - n_evaluations
        #print("initial remaining budget: {}".format(remaining_budget))
        step = 0
        while remaining_budget > 0:
            print("########## Step {} #############".format(step))
            error_array = np.array(self.errors)
            #print("error array: {}".format(error_array))
            sort_args = np.argsort(error_array)[::-1]
            self.intervals = np.array(self.intervals)[sort_args].tolist()
            max_evaluations = np.min([n_parallel, remaining_budget])
            #print("max_evaluations: {}".format(max_evaluations))
            total_hash_list = []
            total_to_evaluate = np.empty((0,2))
            total_new_intervals = []
            for i_int in range(len(self.intervals)):
                #print("i : {}".format(i_int))
                interval = self.intervals[i_int]
                #interval = self.intervals.pop(0)
                new_intervals = interval.split()
                #self.intervals.extend(new_intervals)
                hash_list, to_evaluate = self.get_unique_points(new_intervals)
                #print("new_points_required: {}, max_evals: {}".format(len(total_hash_list) + len(hash_list), max_evaluations))
                if len(total_hash_list) + len(hash_list) > max_evaluations:
                    i_int -= 1
                    #print("cannot split interval, too few evaluations")
                    break
                total_new_intervals += new_intervals
                total_hash_list += hash_list
                total_to_evaluate = np.vstack([total_to_evaluate, to_evaluate])
                if len(total_hash_list) >= max_evaluations:                    
                    break
            if len(total_hash_list) == 0:
                print("not enough parallel evaluations allowed, need at least {} for refinemnet".format(len(hash_list)))                
                break
            #print("to evaluate: {}".format(to_evaluate))
            print("# new evaluations: {}".format(total_to_evaluate.shape[0]))
            #print("# new evaluations: {}".format(len(total_hash_list)))          
            if remaining_budget - len(total_hash_list) <0:
                print("terminating due to simulation budget mid step")
                break
            else:
                for j in range(i_int+1):
                    self.intervals.pop(0)
                #print("# new intervals: {}".format(len(total_new_intervals)))
                self.intervals.extend(total_new_intervals)
            #to_evaluate = np.array(to_evaluate)
            f_vals = self.evaluate_function(total_to_evaluate[:,0], total_to_evaluate[:,1])
            self.n_evaluations += f_vals.shape[0]

            self.update_global_maps(total_hash_list, total_to_evaluate, f_vals)            
            self.update_intervals(total_new_intervals, total_hash_list, f_vals)
            
            errors = np.zeros(len(self.intervals))
            for i_int, interval in enumerate(self.intervals):
                errors[i_int] = interval.estimate_error()
            self.errors = errors
            step += 1
            n_evaluations += len(total_to_evaluate)
            remaining_budget -= len(total_to_evaluate)            
            #print("n_evaluations: {}".format(n_evaluations))
            #print("remaining budget: {}".format(remaining_budget))
            if int(remaining_budget/13.) == 0:
                print("terminating due to simulation budget end step")
                remaining_budget = 0
                break
            if np.max(errors) < self.tol:
                #print("terminating due to error tolerance")
                #print(self.tol)
                #print(errors)
                break

   
    def get_grid(self, grid_type='six_point'):
        if grid_type == 'three_point':
            max_row = 3
        elif grid_type == 'six_point':
            max_row = 5
        elif grid_type == 'seven_point':
            max_row = 7
        points = []
        values = []
        global_triangles = []
        vertex_index = 0
        hashed_vals = {}
        all_hashed_vals = set()
        for i_int, interval in enumerate(self.intervals):
            #print("########## Interval {} ########".format(i_int))
            if grid_type == 'three_point':
                local_triangles = interval.big_triangle()
                tri_list = [[]]
            elif grid_type == 'six_point':
                local_triangles = interval.small_triangles()
                tri_list = [[],[],[],[]]
            elif grid_type == 'seven_point':
                local_grid = np.vstack([interval.x, interval.y]).T
                local_triangles = scipy.spatial.Delaunay(local_grid).simplices
                tri_list = [[] for _ in range(local_triangles.shape[0])]
            for hash_val, row in interval.hash_map.items():
                all_hashed_vals.add(hash_val)
                if row > max_row:
                    continue
                
                if hash_val not in hashed_vals:
                    point = interval.x[row], interval.y[row]
                    points.append(point)
                    values.append(interval.f[row])
                    hashed_vals[hash_val] = vertex_index
                    for t_row in range(local_triangles.shape[0]):
                        if row in local_triangles[t_row, :]:
                            tri_list[t_row].append(vertex_index)
                    vertex_index += 1
                else:
                    for t_row in range(local_triangles.shape[0]):
                        if row in local_triangles[t_row, :]:
                            tri_list[t_row].append(hashed_vals[hash_val])
            global_triangles += tri_list
        self._all_hashed_vals = all_hashed_vals
        grid = np.array(points)
        f_vals = np.array(values)
        if grid_type == 'seven_point':
            triangles = scipy.spatial.Delaunay(grid).simplices
        else:
            triangles = np.array(global_triangles)
        return grid, f_vals, triangles                                       
