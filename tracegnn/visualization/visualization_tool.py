import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KDTree
from graphviz import Digraph
import os
from tracegnn.data import TraceGraphIDManager
from datetime import datetime, timedelta
from typing import *
from loguru import logger
import snappy
import json


class WebViewerUtil:
    def __init__(self,
                 case_idx: int = 0):
        # Load data from pickle file
        logger.info(f"[WebViewerUtil] Using case: {case_idx}")
        logger.info(f"[WebViewerUtil] Loading self.result from: tracegnn/visualization/sample_cases/model_dat/case_{case_idx}.pkl")
        self.result = pickle.loads(snappy.decompress(open(f'tracegnn/visualization/sample_cases/model_dat/case_{case_idx}.pkl', 'rb').read()))
        logger.info("[WebViewerUtil] Loading finished.")

        # Load id_manager
        self.id_manager = TraceGraphIDManager('tracegnn/visualization/sample_cases/id_manager')
        self.nll_info = json.load(open('tracegnn/visualization/sample_cases/model_dat/nll_info.json', 'rt'))

    def _load_between(self, start_ts: int, end_ts: int, before_start_ts: int):
        anomaly_x, anomaly_y = [], []
        before_x, before_y = [], []
        anomaly_idx, before_idx = [], []
        anomaly_scores = {}

        # Load from data
        for i in range(len(self.result['start_time'])):
            if start_ts <= self.result['start_time'][i] < end_ts:
                anomaly_idx.append(i)
                anomaly_x.append(self.result['encoded'][i])
                anomaly_y.append(self.result['graph_latency_nll'][i] / self.nll_info['graph_latency_nll_p99'] + self.result['graph_structure_nll'][i] / self.nll_info['graph_structure_nll_p99'])

                for j in range(len(self.result['node_nll'][i])):
                    anomaly_scores.setdefault(self.result['operation_id'][i][j], [])
                    anomaly_scores[self.result['operation_id'][i][j]].append(self.result['node_nll'][i][j] / self.nll_info['node_nll_p99'])
            elif before_start_ts <= self.result['start_time'][i] < start_ts:
                before_idx.append(i)
                before_x.append(self.result['encoded'][i])
                before_y.append(self.result['graph_latency_nll'][i] / self.nll_info['graph_latency_nll_p99'] + self.result['graph_structure_nll'][i] / self.nll_info['graph_structure_nll_p99'])

        anomaly_x = np.stack(anomaly_x, axis=0)
        before_x = np.stack(before_x, axis=0)
        before_y = np.array(before_y).astype(float)
        anomaly_y = (np.array(anomaly_y) > 1.0).astype(int)
        anomaly_idx, before_idx = np.array(anomaly_idx), np.array(before_idx)
        logger.info(f"{len(anomaly_y)=}, {anomaly_y.sum()=}")

        return (before_x, before_y, before_idx), (anomaly_x, anomaly_y, anomaly_idx)

    def _query_kd_tree(self, before_x, before_y, anomaly_x, normal_threshold: float=0.5):
        normal_idx = [i for i in range(len(before_x)) if before_y[i] < normal_threshold]
        normal_x = [before_x[i] for i in normal_idx]
        kd_tree = KDTree(normal_x)
        _, result = kd_tree.query(anomaly_x)
        result = [normal_idx[i[0]] for i in result]

        return result

    def get_node_scores(self, start_ts: int, end_ts: int, before_start_ts: int, score_threshold: float=1.0):
        _, (_, _, anomaly_idx) = self._load_between(start_ts, end_ts, before_start_ts)

        # Use the JI value to rank the nodes
        op_struct_anomaly_dict = {}
        op_latency_anomaly_dict = {}
        op_cnt_dict = {}
        total_struct_anomaly_cnt, total_latency_anomaly_cnt = 0, 0

        for idx in anomaly_idx:
            current_struct_anomaly, current_latency_anomaly = 0, 0
            op_set = set(self.result['operation_id'][idx]) | {self.id_manager.num_operations}
            reconst_op_set = set([item[0] for item in self.result['reconstruct_nodes'][idx]])
            
            # Structural anomalies
            for op in reconst_op_set - op_set:
                op_struct_anomaly_dict.setdefault(op, 0.0)
                op_struct_anomaly_dict[op] += self.result['graph_structure_nll'][idx] / self.nll_info['graph_structure_nll_p99']
                current_struct_anomaly = 1
            # Latency anomalies
            for i in range(len(self.result['operation_id'][idx])):
                op = self.result['operation_id'][idx][i]
                op_latency_anomaly_dict.setdefault(op, 0.0)
                op_latency_anomaly_dict[op] += self.result['node_nll'][idx][i] / self.nll_info['node_nll_p99']
                if self.result['node_nll'][idx][i] / self.nll_info['node_nll_p99'] > score_threshold:
                    current_latency_anomaly = 1

            for op in (reconst_op_set | op_set):
                op_cnt_dict.setdefault(op, 0)
                op_cnt_dict[op] += 1

            total_struct_anomaly_cnt += current_struct_anomaly
            total_latency_anomaly_cnt += current_latency_anomaly

        # Calculate Struct & Latency JI Value
        op_struct_ji_dict = {}
        for op in op_struct_anomaly_dict:
            if op not in op_cnt_dict:
                continue
            x_and_y = op_struct_anomaly_dict[op]
            x_or_y = total_struct_anomaly_cnt + op_cnt_dict[op]
            op_struct_ji_dict[op] = x_and_y / (x_or_y + 1e-7)

        op_latency_ji_dict = {}
        for op in op_latency_anomaly_dict:
            if op not in op_cnt_dict:
                continue
            x_and_y = op_latency_anomaly_dict[op]
            x_or_y = total_latency_anomaly_cnt + op_cnt_dict[op]
            op_latency_ji_dict[op] = x_and_y / (x_or_y + 1e-7)

        return op_struct_ji_dict, op_latency_ji_dict

    def plot_at(self, op: int, op_struct_ji_dict, op_latency_ji_dict, start_ts: int, end_ts: int, before_start_ts: int, normal_threshold: float=0.5, struct_normal_threshold: float=0.3, pa_depth: int=2, child_depth: int=2):
        logger.info(f"[WebViewerUtil] Processing with API: {op}")
        (before_x, before_y, before_idx), (anomaly_x, anomaly_y, anomaly_idx) = self._load_between(start_ts, end_ts, before_start_ts)

        # Find original trees with kd_tree
        origin_trees = self._query_kd_tree(before_x, before_y, anomaly_x, normal_threshold=normal_threshold)
        
        # Build tree with given op (root nodes, pa nodes, child nodes)
        root_nodes: Set[int] = set()
        nodes: Set[int] = {op}
        edges: Set[Tuple[int, int]] = set()
        edges_nll_list: Dict[Tuple[int, int], List[float]] = {}

        for idx in origin_trees:
            for item in self.result['graph_key'][idx].split(';'):
                s, e = map(int, item.split(','))
                if s == 0:
                    root_nodes.add(e)

            # Up BFS
            last_nodes: Set[int] = {op}
            for depth in range(pa_depth):
                cur_nodes: Set[int] = set()
                for i, item in enumerate(self.result['graph_key'][idx].split(';')):
                    s, e = map(int, item.split(','))
                    if e in last_nodes:
                        cur_nodes.add(s)
                        nodes.add(s)
                        edges.add((s, e))
                        edges_nll_list.setdefault((s, e), [])
                        edges_nll_list[(s, e)].append(self.result['node_nll'][idx][i] / self.nll_info['node_nll_p99'])
                last_nodes = cur_nodes
            
            # Down BFS
            last_nodes: Set[int] = {op}
            for depth in range(child_depth):
                cur_nodes: Set[int] = set()
                for i, item in enumerate(self.result['graph_key'][idx].split(';')):
                    s, e = map(int, item.split(','))
                    if s in last_nodes:
                        cur_nodes.add(s)
                        nodes.add(e)
                        edges.add((s, e))
                        edges_nll_list.setdefault((s, e), [])
                        edges_nll_list[(s, e)].append(self.result['node_nll'][idx][i] / self.nll_info['node_nll_p99'])
                last_nodes = cur_nodes

        logger.info("[WebViewerUtil] Rendering graph...")
        def score_to_color(x: float) -> str:
            """
                x: 0 ~ 1
            """
            green = np.array([0.0, 255.0, 0.0])
            yellow = np.array([255.0, 255.0, 0.0])
            red = np.array([255.0, 0.0, 0.0])

            if x <= 0.5:
                color = yellow * x * 2 + (1.0 - x * 2) * green
            else:
                color = red * (x - 0.5) * 2 + (1.0 - (x - 0.5) * 2) * yellow

            s = '#'
            for i in range(3):
                c = int(max(min(color[i], 255.0), 0.0))
                s += f'{c:02x}'

            return s

        # Calculate edges color (normalized)
        edges_color = {}
        for k, v in edges_nll_list.items():
            edges_color[k] = score_to_color(np.mean(v))

        # Calculate nodes fillcolor (normalized)
        op_fill_color = {}
        for k in set(op_struct_ji_dict) | set(op_latency_ji_dict):
            op_fill_color[k] = score_to_color(op_struct_ji_dict.get(k, 0.0) + op_latency_ji_dict.get(k, 0.0))

        # Plot tree with graphviz
        g = Digraph(str(op), node_attr={'shape': 'rect', 'style': 'filled', 'width': '1.0', 'height': '0.3'}, graph_attr={'ranksep': '0.25'})
        for nd in nodes:
            g.node(name=str(nd), label="API " + str(nd) + ("*" if nd == op else ""), fillcolor=op_fill_color.get(nd, '#00ff00') + '88', color=score_to_color(op_struct_ji_dict.get(nd, 0.0) / struct_normal_threshold))
        for s, e in edges:
            g.edge(str(s), str(e))

        g.render(filename=f"tracegnn/visualization/assets/visualization_case", format='svg', cleanup=True)

        logger.info("[WebViewerUtil] Finished.")



if __name__ == '__main__':
    time_dict = {
        0: {'start_ts': 1650523740, 'end_ts': 1650523980, 'before_start_ts': 1650523440},
        1: {'start_ts': 1650523200, 'end_ts': 1650523440, 'before_start_ts': 1650522900}
    }

    # Plot case 0
    web_viewer_util = WebViewerUtil(0)
    start_ts, end_ts, before_start_ts = time_dict[0]['start_ts'], time_dict[0]['end_ts'], time_dict[0]['before_start_ts']

    op_struct_anomaly_dict, op_latency_anomaly_dict = web_viewer_util.get_node_scores(start_ts, end_ts, before_start_ts, score_threshold=1.0)
    web_viewer_util.plot_at(2, op_struct_anomaly_dict, op_latency_anomaly_dict, start_ts, end_ts, before_start_ts, normal_threshold=0.5, pa_depth=2, child_depth=2)

    # Plot case 1
    web_viewer_util = WebViewerUtil(1)
    start_ts, end_ts, before_start_ts = time_dict[1]['start_ts'], time_dict[1]['end_ts'], time_dict[1]['before_start_ts']
    op_struct_anomaly_dict, op_latency_anomaly_dict = web_viewer_util.get_node_scores(start_ts, end_ts, before_start_ts, score_threshold=1.0)
    web_viewer_util.plot_at(61, op_struct_anomaly_dict, op_latency_anomaly_dict, start_ts, end_ts, before_start_ts, normal_threshold=0.5, pa_depth=2, child_depth=2)
