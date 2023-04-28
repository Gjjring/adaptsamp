import numpy as np
import itertools
import scipy.spatial

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
    
    def barycentric(self, point):
        A = np.concatenate([np.ones((3, 1)), self.x.reshape(3,1), self.y.reshape(3,1)], axis=1)
        detA = np.linalg.det(A)
        A1 =  A.copy()
        A1[0, 1:] = point
        detA1 = np.linalg.det(A1)
        
        A2 =  A.copy()
        A2[1, 1:] = point
        detA2 = np.linalg.det(A2)
        
        A3 =  A.copy()
        A3[2, 1:] = point
        detA3 = np.linalg.det(A3)
        
        bpoint = np.array([detA1/detA, detA2/detA, detA3/detA])
        return bpoint
        
    
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
        hash_map, dict:
          hash of the x,y points for faster comparisons

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
        pass
            

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
        t_index = self.big_triangle()[0,:]
        triangle = Triangle(self.x[t_index], self.y[t_index])            
        area = triangle.area()
        estimate = (np.sum(self.f[t_index])/3)*area
        return estimate
        
    def estimate_six_point(self):
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
    
    def interpolate(self, point):
    
        bpoints = self.barycentric(point)
        f_val = 0.
        b0 = bpoints[0]
        b1 = bpoints[1]
        b2 = bpoints[2]
        if False:
            f_val = self.f[0]*b0 + self.f[1]*b1 + self.f[2]*b2
        elif False:
            for i in range(3):
                f_val += bpoints[i]*(2*bpoints[i]-1)*self.f[i]
            for ij, k in zip(itertools.combinations([0, 1, 2], 2), [3, 5, 4]):
                i = ij[0]
                j = ij[1]
                f_val += 4*bpoints[i]*bpoints[j]*self.f[k]
        else:
            for i in range(3):
                f_val += (bpoints[i]*(2*bpoints[i]-1) + 3*b0*b1*b2)*self.f[i]
            for ij, k in zip(itertools.combinations([0, 1, 2], 2), [3, 5, 4]):
                i = ij[0]
                j = ij[1]
                f_val += (4*bpoints[i]*bpoints[j]-12*b0*b1*b2)*self.f[k]
            f_val += 27*b0*b1*b2*self.f[6]
        return f_val
        
        
        
    def barycentric(self, point):
        t_index = self.big_triangle()[0,:]
        triangle = Triangle(self.x[t_index], self.y[t_index])
        bpoint = triangle.barycentric(point)        
        return bpoint
    
    def __repr__(self):
        string = "("
        for row in range(self.x.shape[0]):
            string += "({},{},{}), ".format(self.x[row], self.y[row], self.f[row])
        string = string[:-2] + ")"            
        return string
    
    

                             
        

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
        self.errors = []
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

    def init_intervals_triangle(self):
        points = np.array([[self.x_max, self.y_min],
                           [self.x_max, self.y_max],
                           [self.x_min, self.y_min]])
        f = np.full(3, np.nan)
        interval0 = Interval2D.from_outer_triangle(points[:,0], points[:,1], f)

        n_initial_points = 7 

        hash_list, init_points = self.get_unique_points([interval0])
        print(len(hash_list))
        
        f = self.evaluate_function(init_points[:,0], init_points[:,1])        
        self.update_global_maps(hash_list, init_points, f)        
        print(len(self.global_hash_map))
        self.n_evaluations += f.shape[0]
        self.intervals = []
        self.errors = []
                
        self.update_intervals([interval0], hash_list, f)
        for interval in [interval0]:
            self.intervals.append(interval)
            self.errors.append(interval.estimate_error())


    def init_from_intervals(self, interval_arrays):
        points = []
        f_vals = []
        for int_array in interval_arrays:
            x = int_array[:, 0]
            y = int_array[:, 1]
            xy = int_array[:, :2]
            #points.append(xy)
            f_vals = int_array[:, 2]
            #f_vals.append(f_vals)
            #points = np.vstack(points)
            #f_vals = np.array(f_vals)
            interval = Interval2D(x, y, f_vals)
            hash_list, new_points = self.get_unique_points([interval])
            sliced_f_vals = f_vals[list(interval.hash_map.values())]
            
            
            self.update_global_maps(hash_list, new_points, sliced_f_vals)
            self.update_intervals([interval], hash_list, sliced_f_vals)
            self.intervals.append(interval)
            self.errors.append(interval.estimate_error())

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
        debug = False
        while remaining_budget > 0:
            print("########## Step {} #############".format(step))
            error_array = np.array(self.errors)
            if debug:
                print("error array: {}".format(error_array))
                print("n intervals: {}".format(len(self.intervals)))                
            sort_args = np.argsort(error_array)[::-1]
            self.intervals = np.array(self.intervals)[sort_args].tolist()
            max_evaluations = np.min([n_parallel, remaining_budget])
            if debug:
                print("max_evaluations: {}".format(max_evaluations))
            total_hash_list = []
            total_to_evaluate = np.empty((0,2))
            total_new_intervals = []
            for i_int in range(len(self.intervals)):
                if debug:
                    print("i : {}".format(i_int))
                interval = self.intervals[i_int]
                #interval = self.intervals.pop(0)
                new_intervals = interval.split()
                #self.intervals.extend(new_intervals)
                if debug:
                    print("len(new_intervals): {}".format(len(new_intervals)))
                candidate_new_intervals = total_new_intervals + new_intervals
                hash_list, to_evaluate = self.get_unique_points(candidate_new_intervals)
                #print("hash list: {}".format(hash_list))
                for hash_val in hash_list:
                    if hash_val in self.global_hash_map:
                        print("doubled hash value: {}".format(hash_val))
                if debug:
                    print("new_points_required: {}, max_evals: {}".format(len(total_hash_list) + len(hash_list), max_evaluations))
                if len(total_hash_list) + len(hash_list) > max_evaluations:
                    i_int -= 1
                    print("cannot split interval, too few evaluations")
                    break
                total_new_intervals = candidate_new_intervals
                total_hash_list = hash_list                
                total_to_evaluate = to_evaluate
                if len(total_hash_list) > max_evaluations:
                    i_int -= 1
                    print("hash list larger than max evaluations")
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
                print("len(total_new_intervals): {}".format(len(total_new_intervals)))                
                self.intervals.extend(total_new_intervals)
            #to_evaluate = np.array(to_evaluate)
            f_vals = self.evaluate_function(total_to_evaluate[:,0], total_to_evaluate[:,1])
            self.n_evaluations += f_vals.shape[0]
            if debug:
                print("len(global_hash): {}".format(len(self.global_hash_map)))                
                print("len total hash list: {}".format(len(total_hash_list)))
            self.update_global_maps(total_hash_list, total_to_evaluate, f_vals)
            if debug:
                print("len(global_hash): {}".format(len(self.global_hash_map)))
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
    
    def interpolate(self, x, y):
        values = np.zeros(x.size)
        for ix in range(x.size):
            x_val = x[ix]
            y_val = y[ix]
            xy = np.array([x_val, y_val])
            for i_int, interval in enumerate(self.intervals):
                if interval.contains(xy):
                    values[ix] = interval.interpolate(xy)
                    break
        return values

    def integrate(self):
        value = 0.
        for i_int, interval in enumerate(self.intervals):
            value += interval.estimate_seven_point()
        return value
        
    def get_error_distribution(self):        
        points = []
        error_vals = []
        global_triangles = []
        vertex_index = 0
        hashed_vals = {}
        all_hashed_vals = set()
        max_row = 3
        for i_int, interval in enumerate(self.intervals):
            #print("########## Interval {} ########".format(i_int))            
            local_triangles = interval.big_triangle()
            tri_list = [[]]            
            for hash_val, row in interval.hash_map.items():
                all_hashed_vals.add(hash_val)
                if row > max_row:
                    continue                
                if hash_val not in hashed_vals:
                    point = interval.x[row], interval.y[row]
                    points.append(point)
                    #values.append(interval.f[row])
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
            error_vals.append(interval.estimate_error())
        self._all_hashed_vals = all_hashed_vals
        grid = np.array(points)
        error = np.array(error_vals)
        #f_vals = np.array(values)        
        triangles = np.array(global_triangles)
        return grid, error, triangles                                       
