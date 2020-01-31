# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 00:16:48 2017

@author: Kedar
"""

from scipy import optimize as opt
import numpy as np

#-----------------------------------------------------------------------------#
def my_function(input_vars):
    '''
    function to minimize
    '''
    x = float(input_vars[0])
    y = float(input_vars[1])
    f = (x-3)**2 + (y-3)**2 + 3
    return f
#-----------------------------------------------------------------------------#
def my_function1(input_vars):
    '''
    function to minimize
    '''
    x = float(input_vars[0])
    f = np.sin(x)
    return f
#-----------------------------------------------------------------------------#
def generate_initial_simplex(initial_guess, c=1):
    '''
    given an n-dimensional initial guess of the a function minimum, this 
    subroutine creates a nelder-mead simplex with sides of length one and 
    returns a list of the simplex vertices. the supplied initial guess will be
    the last vertex.
    '''
    # convert the initial guess to a numpy array, just for uniformity
    x_0 = np.array(initial_guess)
    # figure how dimension of the function domain (based on the initial guess)
    n = len(x_0)
    # generate a simplex composed at n+1 points with side-lengths c
    b = (c/(n*np.sqrt(2)))*(np.sqrt(n+1)-1)
    a = b + c/np.sqrt(2)
    simplex_vertices = []
    for i in range(1,n+1):
        i_th_addition = np.array([b]*n)
        i_th_addition[i-1] = a
        i_th_vertex = x_0 + i_th_addition
        simplex_vertices.append(i_th_vertex)
    simplex_vertices.append(x_0)
    return simplex_vertices
#-----------------------------------------------------------------------------#
def shrink_simplex(simplex_vertices):
    '''
    takes in a list of simplex vertices in which the best vertex is first.
    shrinks the simplex by holding the best vertex constant and bringing all
    other vertices halfway closer to the best one. returns the shrunken 
    simplex.
    '''
    rho = 0.5
    best_vertex = simplex_vertices[0]
    shrunken_simplex = [best_vertex]
    for vertex in simplex_vertices[1:]:
        shrunken_vertex = best_vertex + rho*(vertex-best_vertex)
        shrunken_simplex.append(shrunken_vertex)
    return shrunken_simplex
#-----------------------------------------------------------------------------#
def print_simplex_info(simplex_vertices, obj_values, std_simplex, counter):
    '''
    takes the list of arrays of vertices, the list of objective values, and the
    standard deviation of the objective values and prints them nicely to the
    screen.
    '''
    simplex_string = ''
    n = len(simplex_vertices)-1
    simplex_string += '\n\nsimplex #'+str(counter)+':\n'
    print('\n\nsimplex #'+str(counter)+':\n')
    vertex_counter = 1
    for vertex in simplex_vertices:
        simplex_string += '\tvertex #'+str(vertex_counter)+':\t( '
        print('\tvertex #'+str(vertex_counter)+':\t( ', end='')
        point_counter = 1
        for point in vertex:
            simplex_string += str(round(point,5))
            print(str(round(point,5)), end='')
            if point_counter==n:
                simplex_string += ' )\n'
                print(' )')
            else:
                simplex_string += ', '
                print(', ', end='')
            point_counter += 1
        vertex_counter += 1
    simplex_string += '\n'
    print()
    counter = 1
    for value in obj_values:
        simplex_string += '\tobjective at vertex #'+str(counter)+':\t'+str(round(value,7))+'\n'
        print('\tobjective at vertex #'+str(counter)+':\t'+str(round(value,7)))
        counter += 1
    simplex_string += '\n\tst.dev. of objective at the vertices: '+ str(std_simplex)
    print('\n\tst.dev. of objective at the vertices: '+ str(std_simplex))
    return simplex_string
#-----------------------------------------------------------------------------#
def print_vertex(vertex):
    '''
    given an array of vertices, print them nicely
    '''
    vertex_string = ''
    n = len(vertex)
    point_counter = 1
    vertex_string += '( '
    print('( ', end='')
    for point in vertex:
        vertex_string += str(round(point,5))
        print(str(round(point,5)), end='')
        if point_counter==n:
            vertex_string +=' )'
            print(' )', end='')
        else:
            vertex_string +=', '
            print(', ', end='')
        point_counter += 1
    return vertex_string
#-----------------------------------------------------------------------------#
def my_nelder_mead(obj_function, initial_guess, std_conv=1e-6, 
                   initial_side_length=1, verbose=False):
    '''
    this function uses the nelder-mead simplex algorithm to find the minimum of
    a given objective function without using gradients. based on alonso's aa222
    notes, ch. 6
    Input:
        - obj_function:     name of a python subroutine that evaluates a 
                            scalar-valued function given input that looks like
                            initial guess. the call obj_function(initial_guess)
                            should return a scalar value. initial guess can be
                            a vector.
        - initial_guess:    user-supplied domain value(s) that might be the 
                            close to the miniumum (numpy array)
        - std_conv:         the standard deviation of the vertex objective
                            values must be below this for the problem to be 
                            considered converged
    Ouput:
        - x_min:            the local minimum found using nelder-mead
    '''
    # print header
    if verbose:
        my_file = open('nelder_mead_output.txt', 'w')
        print('\n\t\t*** Nelder-Mead Simplex Process ***')
        my_file.write('\n\t\t*** Nelder-Mead Simplex Process ***')
        
    # generate a simplex composed at n+1 points with equal side lengths
    simplex_vertices = generate_initial_simplex(initial_guess, c=initial_side_length)
    
    # evaluate the objection at the simplex vertices and identify the best 
    # point (lowest objective value), the second-worst point (second-highest
    # objective value), and the worst point (highest objective value)
    obj_values = []
    for vertex in simplex_vertices:
        if verbose: 
            print('\n\tevaluating vertex: ', end='')
            my_file.write('\n\tevaluating vertex: ')
            vertex_string = print_vertex(vertex)
            my_file.write(vertex_string)
        obj_values.append(obj_function(vertex))
    
    # compute the standard deviation of the objective values for this simplex
    std_simplex = np.std(obj_values)
    
    # print information about this simplex to the screen
    if verbose:
        simplex_string = print_simplex_info(simplex_vertices, obj_values, std_simplex, counter=0)
        my_file.write(simplex_string)
        
    # initialize the counter
    simplex_counter = 0
    
    # loop until standard deviation criterion reached
    while std_simplex > std_conv:
        
        # zip together the vertices with their corresponding objective values
        vertex_obj_pairs = list(zip(simplex_vertices,obj_values))
        # sort this zipped list by objective value
        vertex_obj_pairs.sort(key=lambda pair: pair[1])
        # find the worst pair
        worst_pair = vertex_obj_pairs[-1]
        # find the second-worst pair
        second_worst_pair = vertex_obj_pairs[-2]
        # find the best pair
        best_pair = vertex_obj_pairs[0]
        
        # create a sorted list of just the vertices
        simplex_vertices = [pair[0] for pair in vertex_obj_pairs]
        
        # excluding the worst vertex, find the location of the average vertex
        n = len(initial_guess)
        summed_vertices = np.array([0.0]*n)
        for pair in vertex_obj_pairs[:-1]:
            summed_vertices += pair[0]
        average_vertex = (1/n)*summed_vertices       
        
        # compute the location of the reflected point
        alpha = 1
        worst_vertex = worst_pair[0]
        reflected_point = average_vertex + alpha*(average_vertex-worst_vertex)
        # compute the value of the objective function at the reflected point
        if verbose: 
            print('\n\tevaluating reflected point: ', end='')
            vertex_string = print_vertex(reflected_point)
            my_file.write('\n\tevaluating reflected point: ')
            my_file.write(vertex_string)
        obj_value_reflected = obj_function(reflected_point)
        
        # if the reflected point yields a better objective value that the best
        # vertex, then keep moving in that direction and define an expansion point
        obj_value_best = best_pair[1]
        obj_value_second_worst = second_worst_pair[1]
        if obj_value_reflected < obj_value_best:
            gamma = 1
            expansion_point = reflected_point+gamma*(reflected_point-average_vertex)
            # if the objective value at the expansion point is lower than the
            # objective value at the best vertex, then we accept the expansion
            # point, otherwise, we accept the reflected point
            if verbose: 
                print('\n\tevaluating expansion point: ', end='')
                vertex_string = print_vertex(expansion_point)
                my_file.write('\n\tevaluating expansion point: ')
                my_file.write(vertex_string)
            obj_value_expansion = obj_function(expansion_point)
            if obj_value_expansion < obj_value_best:
                accepted_point = expansion_point
                simplex_vertices[-1] = accepted_point
            else:
                accepted_point = reflected_point
                simplex_vertices[-1] = accepted_point
        elif obj_value_reflected <= obj_value_second_worst:
            accepted_point = reflected_point
            simplex_vertices[-1] = accepted_point
        else:
            # if the reflected point yields a worse objective value than the worst 
            # vertex, then find an inside-contraction point
            obj_value_worst = worst_pair[1]
            beta = 0.5
            if obj_value_reflected > obj_value_worst:
                inside_contraction_point = average_vertex-beta*(average_vertex-worst_vertex)
                if verbose: 
                    print('\n\tevaluating inside-contraction point: ', end='')
                    vertex_string = print_vertex(inside_contraction_point)
                    my_file.write('\n\tevaluating inside-contraction point: ')
                    my_file.write(vertex_string)
                obj_value_inside_contraction = obj_function(inside_contraction_point)
                if obj_value_inside_contraction < obj_value_worst:
                    accepted_point = inside_contraction_point
                    simplex_vertices[-1] = accepted_point
                else:
                    # shrink the simplex
                    simplex_vertices = shrink_simplex(simplex_vertices)
            else:            
                # if the reflected point is better than the worst vertex but still 
                # worse than the second-worst vertex, find an outside-contraction 
                # point
                outside_contraction_point = average_vertex+beta*(average_vertex-worst_vertex)
                if verbose: 
                    print('\n\tevaluating outside-contraction point: ', end='')
                    vertex_string = print_vertex(outside_contraction_point)
                    my_file.write('\n\tevaluating outside-contraction point: ')
                    my_file.write(vertex_string)
                obj_value_outside_contraction = obj_function(outside_contraction_point)
                if obj_value_outside_contraction <= obj_value_reflected:
                    accepted_point = outside_contraction_point
                    simplex_vertices[-1] = accepted_point
                else:
                    # shrink the simplex
                    simplex_vertices = shrink_simplex(simplex_vertices)
        
        # evaluate the objective at the simplex vertices and identify the best 
        # point (lowest objective value), the second-worst point (second-highest
        # objective value), and the worst point (highest objective value)
        obj_values = []
        if verbose: 
            print()
            my_file.write('\n')
        for vertex in simplex_vertices:
            if verbose:
                print('\n\tevaluating vertex: ', end='')
                vertex_string = print_vertex(vertex)
                my_file.write('\n\tevaluating vertex: ')
                my_file.write(vertex_string)
            obj_values.append(obj_function(vertex))
        
        # compute the standard deviation of the objective values for the simplex
        std_simplex = np.std(obj_values)        
        
        # update simplex counter
        simplex_counter += 1
        
        # print information about this simplex to the screen
        if verbose:
            simplex_string = print_simplex_info(simplex_vertices, obj_values, std_simplex, simplex_counter)
            my_file.write(simplex_string)
            
    # zip together the vertices with their corresponding objective values
    vertex_obj_pairs = list(zip(simplex_vertices,obj_values))
    # sort this zipped list by objective value
    vertex_obj_pairs.sort(key=lambda pair: pair[1])
    # create a sorted list of just the vertices
    simplex_vertices = [pair[0] for pair in vertex_obj_pairs]
    # print the best vertex to the screen
    min_vertex = simplex_vertices[0].astype(float)
    if verbose:
        print('\n\n\t\t*** local minimum found at: ', end='')
        vertex_string = print_vertex(min_vertex)
        print(' ***\n')
        my_file.write('\n\n\t\t*** local minimum found at: ')
        my_file.write(vertex_string)
        my_file.write(' ***\n')
        my_file.close()
    # return the best vertex
    return min_vertex
#-----------------------------------------------------------------------------#
def main():
    '''
    main function, executed when run as a standalone file
    '''
    # practice using scipy's nelder-mead simplex function
    initial_guess = [5,5]
    result = opt.minimize(my_function, initial_guess, method='nelder-mead')
    
    #print('minimum = ', round(result.x[0]/np.pi,3), 'pi')
    print('minimum = ', result.x)
    
    # using my nelder-mead simplex algorithm on a 2D function 
    initial_guess = [5,5]
    best_vertex2 = my_nelder_mead(my_function, initial_guess, std_conv=1e-6, verbose=True)
    print('min at:',best_vertex2)
    
    # using my nelder-mead simplex algorithm on a 1D function 
    initial_guess = [5]
    best_vertex1 = my_nelder_mead(my_function1, initial_guess, std_conv=1e-6, 
                                  initial_side_length=2, verbose=True)
    print('min at:', round(best_vertex1[0]/np.pi,3), 'pi')
#-----------------------------------------------------------------------------#    
# standard boilerplate to call the main() function
if __name__ == '__main__':
    main()