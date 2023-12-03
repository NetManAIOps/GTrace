import pandas as pd
import numpy as np
import os


class SpeedWriter:
    def __init__(self, hyper_params):
        self.hyper_params = hyper_params

        self.path = os.path.join('speed_data', hyper_params['dataset_name'])
        self.ts, self.cnt, self.speed = [0.0], [0], [0.0]
        self.current_ts = 0.0
        self.total_cnt = 0

    def add(self, time_consume, count, eps=1e-7):
        speed = count / (time_consume + 1e-7)

        self.current_ts += time_consume
        self.total_cnt += count

        self.ts.append(self.current_ts)
        self.cnt.append(self.total_cnt)
        self.speed.append(speed)

        # Write to file
        df = pd.DataFrame({
            'ts': self.ts,
            'cnt': self.cnt,
            'speed': self.speed
        })

        os.makedirs(self.path, exist_ok=True)
        file_path = os.path.join(self.path, self.hyper_params['method'] + ".csv")
        df.to_csv(file_path, index=False)
