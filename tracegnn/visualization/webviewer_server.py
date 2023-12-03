from flask import request, Flask, render_template
from tracegnn.visualization.visualization_tool import WebViewerUtil
import pickle
import json
import os

app = Flask(__name__, template_folder='assets/graph_html', static_folder='assets')


@app.route('/<int:case_id>')
def default_index(case_id: int):
    return index(case_id, {
        0: 2,
        1: 61
    }[case_id])

@app.route('/<int:case_id>/<int:select_op>')
def index(case_id: int, select_op: int):
    time_dict = {
        0: {'start_ts': 1650523740, 'end_ts': 1650523980, 'before_start_ts': 1650523440},
        1: {'start_ts': 1650523200, 'end_ts': 1650523440, 'before_start_ts': 1650522900}
    }

    web_viewer_util = WebViewerUtil(case_id)
    start_ts, end_ts, before_start_ts = time_dict[case_id]['start_ts'], time_dict[case_id]['end_ts'], time_dict[case_id]['before_start_ts']

    op_struct_anomaly_dict, op_latency_anomaly_dict = web_viewer_util.get_node_scores(start_ts, end_ts, before_start_ts, score_threshold=1.0)

    # Calculate anomaly score rank
    op_anomaly_score = {}
    for k in set(op_struct_anomaly_dict) | set(op_latency_anomaly_dict):
        op_anomaly_score[k] = op_struct_anomaly_dict.get(k, 0.0) + op_latency_anomaly_dict.get(k, 0.0)

    web_viewer_util.plot_at(select_op, op_struct_anomaly_dict, op_latency_anomaly_dict, start_ts, end_ts, before_start_ts, normal_threshold=0.5, pa_depth=2, child_depth=2)

    return render_template(
        template_name_or_list='webviewer.html',
        graph_size=(1000, 600),
        page_title='Webviewer',
        class_data=sorted(op_anomaly_score.items(), key=lambda x: x[1], reverse=True),
        select_op=select_op,
        case_id=case_id
    )

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=12312,
        debug=True
    )
