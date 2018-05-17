# -*- coding: utf-8 -*-
from graphviz import Source


class TreeVisualizer:
	# Constructor
	def __init__(self):
		self.node_to_graphviz_counter = 0
		self.edge_to_graphviz_counter = 0

	# Generates data for a graphical representation of a node in a public tree.
	# @param node the node to generate data for
	# @return a table containing `name`, `label`, and `shape` fields for graphviz
	# @local
	def node_to_graphviz(self, node):
		out = {'label': '"<f0>' + str(node.current_player)}

		# 1.0 label
		out['label'] += '| spent: '

		for i in range(2):
			out['label'] += str(node.bets[i]) + " "

		out['label'] += '"'

		# 2.0 name
		out['name'] = '"node' + str(self.node_to_graphviz_counter) + '"'
		# 3.0 shape
		out['shape'] = '"record"'
		self.node_to_graphviz_counter += 1

		return out

	# Generates data for graphical representation of a public tree action as an
	# edge in a tree.
	# @param from the graphical node the edge comes from
	# @param to the graphical node the edge goes to
	# @param node the public tree node before at which the action is taken
	# @param child_node the public tree node that results from taking the action
	# @return a table containing fields `id_from`, `id_to`, `id` for graphviz and
	# a `strategy` field to use as a label for the edge
	# @local
	def nodes_to_graphviz_edge(self, fro, to, node, child_node):
		out = {'id_from': fro['name'], 'id_to': to['name'], 'id': self.edge_to_graphviz_counter}

		index = node.children.index(child_node)
		action = node.actions[index]

		if action == -2:
			action = "FOLD"
		elif action == -1:
			action = "CCALL"
		else:
			action = 'R' + str(action)
		out['label'] = action

		self.edge_to_graphviz_counter += 1
		return out

	# Recursively generates graphviz data from a public tree.
	# @param node the current node in the public tree
	# @param nodes a table of graphical nodes generated so far
	# @param edges a table of graphical edges generated so far
	# @local
	def graphviz_dfs(self, node, nodes, edges):
		gv_node = self.node_to_graphviz(node)
		nodes.append(gv_node)

		for child_node in node.children:
			gv_node_child = self.graphviz_dfs(child_node, nodes, edges)
			gv_edge = self.nodes_to_graphviz_edge(gv_node, gv_node_child, node, child_node)
			edges.append(gv_edge)

		return gv_node

	# Generates `.dot` and `.svg` image files which graphically represent
	# a game's public tree.
	#
	# Each node in the image lists the acting player, the number of chips
	# committed by each player, the current betting round, public cards,
	# and the depth of the subtree after the node, as well as any probabilities
	# or values stored in the `ranges_absolute`, `cf_values`, or `cf_values_br`
	# fields of the node.
	#
	# Each edge in the image lists the probability of the action being taken
	# with each private card.
	#
	# @param root the root of the game's public tree
	# @param filename a name used for the output files
	def graphviz(self, root, filename):
		filename = filename or 'unnamed.dot'

		out = 'digraph g {  graph [ rankdir = "LR"];node [fontsize = "16" shape = "ellipse"]; edge [];'

		nodes = []
		edges = []
		self.graphviz_dfs(root, nodes, edges)

		for i in range(len(nodes)):
			node = nodes[i]
			node_text = node['name'] + '[' + 'label=' + node['label'] + ' shape = ' + node['shape'] + '];'

			out = out + node_text

		for i in range(len(edges)):
			edge = edges[i]
			edge_text = str(edge['id_from']) + ':f0 -> ' + str(edge['id_to']) + ':f0 [ id = ' + str(
				edge['id']) + ' label = "' + edge['label'] + '"];'
			out += edge_text

		out += '}'

		with open('../Data/Visual/%s.dot' % filename, 'w') as fout:
			fout.write(str(out))
