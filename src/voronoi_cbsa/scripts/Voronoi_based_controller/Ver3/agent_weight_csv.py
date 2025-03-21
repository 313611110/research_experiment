#!/usr/bin/env python3
import rospy
from voronoi_cbsa.msg import WeightArray
import pandas as pd
import os

class WeightLogger:
    def __init__(self):
        self.weight_data = []
        self.save_path = "~/research_ws/src/voronoi_cbsa/scripts/Voronoi_based_controller/Ver3/agent_weight.csv"

        for i in range(6):  # 假設你有 6 個 agent
            rospy.Subscriber(f"/agent_{i}/visualize/sensor_weights", WeightArray, self.weight_callback, callback_args=i)

    def weight_callback(self, msg, agent_id):
        for weight in msg.weights:
            data = {
                'Agent_ID': agent_id,
                'Target_ID': weight.event_id,
                'Sensor_Type': weight.type,
                'Weight': weight.score
            }
            self.weight_data.append(data)

    def run(self):
        rate = rospy.Rate(1)  # 每秒寫入一次 CSV
        while not rospy.is_shutdown():
            if self.weight_data:
                df = pd.DataFrame(self.weight_data)
                if not os.path.exists(os.path.dirname(self.save_path)):
                    os.makedirs(os.path.dirname(self.save_path))

                if os.path.isfile(self.save_path):
                    df.to_csv(self.save_path, mode='a', header=False, index=False)
                else:
                    df.to_csv(self.save_path, index=False)

                self.weight_data = []  # 清空已儲存的資料
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node('agent_weight_logger', anonymous=True)
    logger = WeightLogger()
    logger.run()
