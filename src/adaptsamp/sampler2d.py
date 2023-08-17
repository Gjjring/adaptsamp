import numpy as np
import itertools
import scipy.spatial
from rtree import index as rindex

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

    def __init__(self, x, y, f, error_func=None):
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
        self.error_func = error_func
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

    def bounding_box(self):
        x_min = np.amin(self.x)
        x_max = np.amax(self.x)
        y_min = np.amin(self.y)
        y_max = np.amax(self.y)
        return (x_min, y_min, x_max, y_max)

    def hash_points(self):
        hash_map = {}
        for row in range(self.x.size):
            point = np.array([self.x[row], self.y[row]])
            str_val = "{}".format(point.tolist())
            hash_map[hash(str_val)] = row
        self.hash_map = hash_map

    def area(self):
        t_index = self.big_triangle()[0,:]
        triangle = Triangle(self.x[t_index], self.y[t_index])
        area = triangle.area()
        return area

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
        if self.error_func is None:
            return self._estimate_error()
        else:
            return self.error_func(self)

    def _estimate_error(self):
        return np.abs(self.estimate_seven_point()-self.estimate_three_point())

    def split(self):
        intervals = []
        triangles = self.small_triangles()
        for row in range(triangles.shape[0]):
            t_index = triangles[row, :]
            x = self.x[t_index]
            y = self.y[t_index]
            f = self.f[t_index]
            interval = Interval2D.from_outer_triangle(x, y, f)
            interval.error_func = self.error_func
            intervals.append(interval)

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
                 n_parallel=1, error_func=None, strategy='mix'):
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
        self.strategy = strategy
        self.n_parallel = n_parallel
        self.intervals = []
        self.errors = []
        self.areas = []
        self.n_evaluations = 0
        self.global_hash_map = {}
        self.point_array = np.empty((0, 2))
        self.f = np.empty((0,))
        self.error_func = error_func
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

    def update_intervals(self, intervals):
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
        interval0.error_func = self.error_func

        points = np.array([[self.x_max, self.y_min],
                           [self.x_max, self.y_max],
                           [self.x_min, self.y_max]])
        interval1 = Interval2D.from_outer_triangle(points[:,0], points[:,1], f)
        interval1.error_func = self.error_func

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
        self.areas = []

        self.update_intervals([interval0, interval1])
        for interval in [interval0, interval1]:
            self.intervals.append(interval)
            self.errors.append(interval.estimate_error())
            self.areas.append(interval.area())

        #self.intervals = [Interval(x_init, y_init)]
        #self.errors = [self.intervals[0].estimate_error()]

    def init_intervals_triangle(self):
        points = np.array([[self.x_max, self.y_min],
                           [self.x_max, self.y_max],
                           [self.x_min, self.y_min]])
        f = np.full(3, np.nan)
        interval0 = Interval2D.from_outer_triangle(points[:,0], points[:,1], f)
        interval0.error_func = self.error_func
        n_initial_points = 7

        hash_list, init_points = self.get_unique_points([interval0])
        #print(len(hash_list))

        f = self.evaluate_function(init_points[:,0], init_points[:,1])
        self.update_global_maps(hash_list, init_points, f)
        #print(len(self.global_hash_map))
        self.n_evaluations += f.shape[0]
        self.intervals = []
        self.errors = []
        self.areas = []
        self.update_intervals([interval0])
        for interval in [interval0]:
            self.intervals.append(interval)
            self.errors.append(interval.estimate_error())
            self.areas.append(interval.area())


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
            interval.error_func = self.error_func
            hash_list, new_points = self.get_unique_points([interval])
            sliced_f_vals = []
            for hash_val in hash_list:
                interval_index = interval.hash_map[hash_val]
                sliced_f_vals.append(interval.f[interval_index])
            sliced_f_vals = np.array(sliced_f_vals)
            #sliced_f_vals = f_vals[list(interval.hash_map.values())]
            self.update_global_maps(hash_list, new_points, sliced_f_vals)
            self.update_intervals([interval])
            self.intervals.append(interval)
            self.errors.append(interval.estimate_error())
            self.areas.append(interval.area())

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
        debug = True
        info = False
        while remaining_budget > 0:
            if info:
                print("########## Step {} #############".format(step))
            error_array = np.array(self.errors)
            areas_array = np.array(self.areas)

            largest_area = np.amax(areas_array)
            median_area = np.median(areas_array)
            #if largest_area > median_area*25:
            #    refinement_strategy = 'area'
            #    sort_args = np.argsort(areas_array)[::-1]
            #else:
            #    refinement_strategy = 'error'
            #    sort_args = np.argsort(error_array)[::-1]
            refinement_strategy = self.strategy
            #refinement_strategy= 'error'
            if refinement_strategy == 'mix':
                sort_args = np.argsort(np.sqrt(areas_array)*error_array)[::-1]
            elif refinement_strategy == 'error':
                sort_args = np.argsort(error_array)[::-1]

            mean_error = np.mean(error_array)
            std_dev_error = np.std(error_array)
            rms_error = np.sqrt(np.mean(error_array**2))
            if info:
                print("area ratio: {:.7e}".format(largest_area/median_area))
                print("rms error: {:.7e}".format(rms_error))
                print("std error: {:.7e}".format(std_dev_error/mean_error))
            if debug:
                print("refinement step: {}".format(refinement_strategy))
                print("error array: {}".format(error_array))
                print("n intervals: {}".format(len(self.intervals)))

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
                if interval.estimate_error() <= self.tol:
                    i_int -= 1
                    break
                new_intervals = interval.split()
                if debug:
                    print("len(new_intervals): {}".format(len(new_intervals)))
                candidate_new_intervals = total_new_intervals + new_intervals
                hash_list, to_evaluate = self.get_unique_points(candidate_new_intervals)
                if debug:
                    print("new_points_required: {}, max_evals: {}".format(len(hash_list), max_evaluations))
                if len(hash_list) > max_evaluations:
                    i_int -= 1
                    if debug:
                        print("cannot split interval, too few evaluations")
                    break
                total_new_intervals = candidate_new_intervals
                total_hash_list = hash_list
                total_to_evaluate = to_evaluate
                #if len(total_hash_list) > max_evaluations:
                #    i_int -= 1
                #    print("hash list larger than max evaluations")
                #    break
            if len(total_hash_list) == 0:
                if debug:
                    print("not enough parallel evaluations allowed, need at least {} for refinemnet".format(len(hash_list)))
                break
            if debug:
                print("# new evaluations: {}".format(total_to_evaluate.shape[0]))

            #to_evaluate = np.array(to_evaluate)
            if remaining_budget - len(total_hash_list) <0:
                print("terminating due to simulation budget mid step")
                break
            f_vals = self.evaluate_function(total_to_evaluate[:,0], total_to_evaluate[:,1])
            if debug:
                print("n intervals before pop: {}".format(len(self.intervals)))
                print("poping {} intervals".format(i_int+1))
            for j in range(i_int+1):
                self.intervals.pop(0)
            if debug:
                print("n intervals after pop: {}".format(len(self.intervals)))
            #print("# new intervals: {}".format(len(total_new_intervals)))
            if debug:
                print("len(total_new_intervals): {}".format(len(total_new_intervals)))
            self.intervals.extend(total_new_intervals)

            self.n_evaluations += f_vals.shape[0]
            if debug:
                print("len(global_hash): {}".format(len(self.global_hash_map)))
                print("len total hash list: {}".format(len(total_hash_list)))
            self.update_global_maps(total_hash_list, total_to_evaluate, f_vals)
            if debug:
                print("len(global_hash): {}".format(len(self.global_hash_map)))
            self.update_intervals(total_new_intervals)

            errors = np.zeros(len(self.intervals))
            areas = np.zeros(len(self.intervals))
            for i_int, interval in enumerate(self.intervals):
                errors[i_int] = interval.estimate_error()
                areas[i_int] = interval.area()
            self.errors = errors
            self.areas = areas
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
                print("terminating due to error tolerance")
                #print(self.tol)
                #print(errors)
                break


    def get_grid(self, grid_type='six_point'):
        if grid_type == 'three_point':
            max_row = 3
        elif grid_type == 'six_point':
            max_row = 6
        elif grid_type == 'seven_point':
            max_row = 7
        points = []
        values = []
        weights = []
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
                if row >= max_row:
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
            triangles = np.vstack(global_triangles)
        return grid, f_vals, triangles

    def interpolate(self, x, y):
        values = np.zeros(x.size)
        idx = rindex.Index()
        for ii, interval in enumerate(self.intervals):
            idx.insert(ii, interval.bounding_box())

        for ix in range(x.size):
            x_val = x[ix]
            y_val = y[ix]
            xy = np.array([x_val, y_val])
            query = [x_val, y_val, x_val, y_val]
            indices = list(idx.intersection(query))
            for index in indices:
                interval = self.intervals[index]
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
                if row >= max_row:
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
            error_vals.append(interval.estimate_error()*np.sqrt(interval.area()))
        self._all_hashed_vals = all_hashed_vals
        grid = np.array(points)
        error = np.array(error_vals)
        #f_vals = np.array(values)
        triangles = np.array(global_triangles)
        return grid, error, triangles

    def get_grid_weights(self, grid_type='six_point'):
        if grid_type == 'three_point':
            max_row = 3
            weight_fraction=1./3.
        elif grid_type == 'six_point':
            max_row = 6
            weight_fraction=1./6.
        elif grid_type == 'seven_point':
            max_row = 7
        points = []
        weights = []
        global_triangles = []
        vertex_index = 0
        hashed_vals = {}
        all_hashed_vals = set()

        global_area = (self.x_max-self.x_min)*(self.y_max-self.y_min)*0.5

        for i_int, interval in enumerate(self.intervals):
            #print("########## Interval {} ########".format(i_int))
            interval_area = interval.area()
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
                if row >= max_row:
                    continue

                if hash_val not in hashed_vals:
                    point = interval.x[row], interval.y[row]
                    points.append(point)
                    weights.append(interval_area*weight_fraction/global_area)
                    hashed_vals[hash_val] = vertex_index
                    for t_row in range(local_triangles.shape[0]):
                        if row in local_triangles[t_row, :]:
                            tri_list[t_row].append(vertex_index)
                    vertex_index += 1
                else:
                    for t_row in range(local_triangles.shape[0]):
                        if row in local_triangles[t_row, :]:
                            tri_list[t_row].append(hashed_vals[hash_val])
                    current_vertex_index = hashed_vals[hash_val]
                    weights[current_vertex_index] += interval_area*weight_fraction/global_area
            global_triangles += tri_list
        self._all_hashed_vals = all_hashed_vals
        #self._hashed_vals = hashed_vals
        grid = np.array(points)
        weights = np.array(weights)
        if grid_type == 'seven_point':
            triangles = scipy.spatial.Delaunay(grid).simplices
        else:
            triangles = np.vstack(global_triangles)
        return grid, weights, triangles
