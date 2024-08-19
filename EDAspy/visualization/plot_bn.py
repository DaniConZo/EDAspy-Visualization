#!/usr/bin/env python
# coding: utf-8

import ipywidgets as widgets
from ipywidgets import interact
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from EDAspy.optimization import EdaResult
from typing import Union


def arcs2adj_mat(arcs: list, n_variables: int) -> np.array:
    """
    This function transforms the list of arcs in the BN structure to an adjacency matrix.

    :param arcs: list of arcs in the BN structure.
    :param n_variables: number of variables.
    :type arcs: list
    :type n_variables: int
    :return: adjacency matrix
    :rtype: np.array
    """

    matrix = np.zeros((n_variables, n_variables))
    for arc in arcs:
        matrix[int(arc[0]), int(arc[1])] = 1

    return matrix

def bn_report(input_data: EdaResult=None, importance_threshold: float = 0.2, print_total_adjacency_matrix: bool = False):
    """This function displays a summary, in the form of a heat-map, of the variable relationships learned in the BN strucutre during the EDA run. It also prints a report where 
       the most frequent arcs, given an importance threshold specified by the user, are enumerated.
       By default, arcs that appear in at least 20% of iterations are enumerated.

       :param input_data: EdaResult object
       :param importance_threshold: float. Proportion [0-1] of total iterations
       :param print_total_adjacency_matrix: bool if True the total adjacency matrix is returned and can be saved into a variable

       :type input_data: EdaResult
       :type importance_threshold: float
       :type print_total_adjacency_matrix: bool
    
    """
    
    assert isinstance(input_data, EdaResult), 'Input object is not EdaResult class'
    assert type(importance_threshold) is float, 'importance_threshold must be float [0-1]' 
    assert 0 <= importance_threshold <=1, 'importance_threshold must be float [0-1]'
    assert type(print_total_adjacency_matrix) is bool, 'print_total_adjacency matrix must be a boolean'

    n_variables = len(input_data.sel_inds_hist[0,0,:-1])
    #Sum of all adjacency matrices
    M = np.zeros((n_variables,n_variables))
    for gen in range(len(input_data.sel_inds_hist)):
        M+=arcs2adj_mat(arcs=input_data.prob_clone_hist[gen].arcs(), n_variables=n_variables) #input_data.prob_mod_hist[gen].print_structure() if .pm.clone() is used

    ## Matrix plot (heat-map)
    ##TODO pasar opciones del plot a los argumentos de la función para que puedan ser modificados por el usuario.
    #TODO nombres de las variables
    nodes = input_data.prob_clone_hist[0].nodes()

    plt.imshow(M, cmap='plasma')
    plt.xticks(np.arange(len(nodes)), labels=nodes)
    plt.yticks(np.arange(len(nodes)), labels=nodes)
    plt.colorbar()
    plt.title('BN report: arc appearances')


    tobesorted=[]
    counter=0
    for i in range(n_variables):
        for j in range(n_variables):
            if M[i,j] >= importance_threshold*len(input_data.sel_inds_hist):
                tobesorted.append((M[i, j],(i, j)))
                #print(f'Arc ({i}, {j}) is present at least {100*importance_threshold}% of total iterations. ({M[i,j]} times)')
            else:
                counter += 1
    sortedelements = sorted(tobesorted, key=lambda x: x[0], reverse=True)
    for elements in sortedelements:
        print(f'Arc {elements[1]} is present {elements[0]} times. {round(elements[0]*100/len(input_data.sel_inds_hist),2)}% iterations')

    if counter==n_variables**2:
        print(f'No arc appears {100*importance_threshold}% of total iterations')

    if print_total_adjacency_matrix:
        return(M)


def _noise(n_variables: int, size: float) -> np.array:
    h_noise = np.zeros(n_variables)
    h_noise[::2] = size*2
    return h_noise - size


def _set_positions(variables: list) -> dict:
    n_variables = len(variables)
    n_cols = int(np.sqrt(n_variables))
    n_rows = int(np.ceil(n_variables / n_cols))

    pos_list = []
    for row in range(n_rows):
        for col in range(n_cols):
            pos_list.append([col, -row])

    '''if noise:
        noise_list = _noise(len_variables, size)
        for i in range(len_variables):
            pos_list[i][0] += noise_list[i]
            pos_list[i][1] += noise_list[i]'''

    return {variables[i]: pos_list[i] for i in range(n_variables)}


def plt_bn(arcs: list, var_names: list, pos: dict = None, curved_arcs: bool = True,
            curvature: float = -0.3, node_size: int = 500, node_color: str = 'red',
            edge_color: str = 'black', arrow_size: int = 15, node_transparency: float = 0.9,
            edge_transparency: float = 0.9, node_line_widths: float = 2, title: str = None,
            output_file: str = None):
    """
    This function Plots a BN structure as a directed acyclic graph.

    :param arcs: Arcs in the BN structure.
    :param var_names: List of variables.
    :param pos: Positions in the plot for each node.
    :param curved_arcs: True if curved arcs are desired.
    :param curvature: Radians of curvature for edges. By default, -0.3.
    :param node_size: Size of the nodes in the graph. By default, 500.
    :param node_color: Color set to nodes. By default, 'red'.
    :param edge_color: Color set to edges. By default, 'black'.
    :param arrow_size: Size of arrows in edges. By default, 15.
    :param node_transparency: Alpha value [0, 1] that defines the transparency of the node. By default, 0.9.
    :param edge_transparency: Alpha value [0, 1] that defines the transparency of the edge. By default, 0.9.
    :param node_line_widths: Width of the nodes contour lines. By default, 2.0.
    :param title: Title for Figure. By default, None.
    :param output_file: Path to save the figure locally.
    :type arcs: list(tuple)
    :type var_names: list
    :type pos: dict {name of variables: tuples with coordinates}
    :type curved_arcs: bool
    :type curvature: float
    :type node_size: int
    :type node_color: str
    :type edge_color: str
    :type arrow_size: int
    :type node_transparency: float
    :type edge_transparency: float
    :type node_line_widths: float
    :type title: str
    :type output_file: str
    :return: Figure.
    """

    g = nx.DiGraph()
    g.add_nodes_from(var_names)
    g.add_edges_from(arcs)

    if not pos:
        pos = _set_positions(var_names)

    nx.draw_networkx_nodes(g, pos, node_size=node_size, node_color=node_color, alpha=node_transparency,
                           linewidths=node_line_widths)

    if curved_arcs:
        nx.draw_networkx_edges(g, pos, edgelist=g.edges(), edge_color=edge_color,
                               connectionstyle="arc3,rad=" + str(curvature), arrowsize=arrow_size,
                               alpha=edge_transparency)
    else:
        nx.draw_networkx_edges(g, pos, edgelist=g.edges(), edge_color=edge_color, arrowsize=arrow_size)

    nx.draw_networkx_labels(g, pos)

    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)

    #plt.show()


def plot_bn(input_data: EdaResult = None , fig_size: tuple = (8,5), var_names: Union[list, None] = None, pos: Union[dict, None] = None, curved_arcs: bool = True,
            curvature: float = -0.3, node_size: int = 500, node_color: str = 'red',
            edge_color: str = 'black', arrow_size: int = 15, node_transparency: float = 0.9,
            edge_transparency: float = 0.9, node_line_widths: float = 2., plt_suptitle: str = 'BN structure evolution'):
    
    '''This function returns a interactive plot with BN structure evolution during the algorithm run.
       This function calls plt_bn function which given a set of arcs plots a single BN. 

        :param input_data: eda_result object
        :param fig_size: tuple wiht figsize passed to plt.figure(figsize)
        :param plt_suptitle: string with sup title
        :param arcs: Arcs in the BN structure.
        :param var_names: List of variables.
        :param pos: Positions in the plot for each node.
        :param curved_arcs: True if curved arcs are desired.
        :param curvature: Radians of curvature for edges. By default, -0.3.
        :param node_size: Size of the nodes in the graph. By default, 500.
        :param node_color: Color set to nodes. By default, 'red'.
        :param edge_color: Color set to edges. By default, 'black'.
        :param arrow_size: Size of arrows in edges. By default, 15.
        :param node_transparency: Alpha value [0, 1] that defines the transparency of the node. By default, 0.9.
        :param edge_transparency: Alpha value [0, 1] that defines the transparency of the edge. By default, 0.9.
        :param node_line_widths: Width of the nodes contour lines. By default, 2.0.
        :param sup_title: Sup Title for Figure.

        :type input_data: EdaResult
        :type fig_size: tuple
        :type plt_suptitle: str
        :type arcs: list(tuple)
        :type var_names: list
        :type pos: dict {name of variables: tuples with coordinates}
        :type curved_arcs: bool
        :type curvature: float
        :type node_size: int
        :type node_color: str
        :type edge_color: str
        :type arrow_size: int
        :type node_transparency: float
        :type edge_transparency: float
        :type node_line_widths: float
        :type plt_suptitle: str
        :return: Figure.'''
    
    assert isinstance(input_data, EdaResult), 'Input object is not EdaResult class'
    assert type(fig_size)==tuple, 'fig_size must be a tuple'
    assert isinstance(var_names, Union[list, None]), 'var must be a list of strings'
    assert isinstance(pos, Union[dict, None]), 'pos must be a dict {name of variables: tuples with coordinates}'
    assert type(curved_arcs)==bool, 'curved_arcs must be boolean'
    assert type(curvature)==float, 'curvature must be float. Radians of curvature for edges'
    assert type(node_size)==int, 'node_size must be an integer'
    assert type(node_color) is str, 'node_color must be a string'
    assert type(edge_color)  is str, 'edge_color must be a string'
    assert type(arrow_size) is int, 'arrow_size must be an integer'
    assert type(node_transparency) is float, 'node_transparency must be a float [0, 1]'
    assert type(edge_transparency) is float, 'edge_transparency must be a float [0, 1]'
    assert type(node_line_widths) is float, 'node_line_widths must be a float'
    assert type(plt_suptitle) is str, 'plt_suptitle must be a string'
    
    #Creation of var_names list if not provided
    if var_names is None:
        var_names = []
        for i in range(len(input_data.sel_inds_hist[0,0,:-1])):
            var_names.append(str(i))

    def update_plot(gen):
            #print(f'{gen} Iteration')
            plt.figure(figsize=fig_size)
            plt.suptitle(plt_suptitle, fontsize=20)
            plt.subplots_adjust(top=0.5)
            plt.title(f'Generation {gen}')
            plt_bn(arcs = input_data.prob_clone_hist[gen].arcs(), var_names=var_names, pos=pos, ##input_data.prob_mod_hist[gen].print_structure() or .arcs() if pm.clone() is used
                     curved_arcs=curved_arcs, curvature=curvature, node_size=node_size, node_color=node_color, edge_color=edge_color,
                     arrow_size=arrow_size, node_transparency=node_transparency, edge_transparency=edge_transparency, node_line_widths=node_line_widths)
            
    def save_plot(_):
        save_location = save_location_text.value#+".pdf"
        update_plot(generation_slider.value)
        plt.savefig(save_location)  # Save the plot using the provided location
        plt.clf()
        plt.close()
            

    generation_slider = widgets.IntSlider(value=0, min=0, max=len(input_data.sel_inds_hist)-1,
                                                step=1, description='Generation', layout=widgets.Layout(width='100%'))
    
    save_location_text = widgets.Text(placeholder='Enter save location here...', description='Save Loc:', layout=widgets.Layout(width='50%'))
    save_button = widgets.Button(description='Save Plot')
    save_button.on_click(save_plot)
     
    interact(update_plot, gen=generation_slider)
    display(save_location_text, save_button)
