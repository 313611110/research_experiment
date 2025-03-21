#! /usr/bin/env python3
import rospy
import pygame
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os
import imageio
import pandas as pd

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray, Float32MultiArray, Float64MultiArray, Float64, String, Int16
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Pose
from voronoi_cbsa.msg import ExchangeData, NeighborInfoArray, TargetInfoArray, SensorArray, ValidSensors, WeightArray,ExchangeDataArray
from scipy.stats import multivariate_normal
from matplotlib.animation import FuncAnimation
import itertools

class Visualize2D():
    def __init__(self):
        self.total_agents   = rospy.get_param('/total_agents', 1)
        map_width           = rospy.get_param("/map_width", 24)
        map_height          = rospy.get_param("/map_height", 24)
        self.map_size       = np.array([map_height, map_width])
        grid_size           = rospy.get_param("/grid_size", 0.1)
        self.grid_size      = np.array([grid_size, grid_size])
        self.size           = (self.map_size/self.grid_size).astype(np.int64)
        self.sensor_pool = ['camera', 'manipulator', 'smoke_detector']
        self.agent_scores           = {}
        self.agent_sensor_scores    = {}
        self.agent_valid_sensors    = {}
        self.agent_pos              = {}
        self.agent_per              = {}
        self.start                  = 0
        self.tmp                    = [-1 for i in range(self.total_agents)]
        self.switch                 = [0 for i in range(self.total_agents)]
        self.controller             = [1 for i in range(self.total_agents)]
        self.t                      = 0
        self.tmp_score              = 0
        self.plt_utl                = []
        self.plt_time               = []
        self.target_received        = False  
        self.plot_role              = "All"      # camera, smoke_detector, manipulator, all
        self.plot_type              = "voronoi"     # voronoi,  footprint
        self.plot_link              = "cooepration" # cooperation, coordination, communication
        self.color                  = {}
        self.event_density          = {}
        self.agent_info             = {}
        self.agent_camera_info      = {}
        self.agent_manipulator_info = {}
        self.agent_smoke_detector_info = {}
        self.total_score            = []
        self.agent_score_plt        = {}
        self.frame                  = []
        self.cnt                    = 0
        self.agent_sensor_weights   = {}
        self.agent_failure          = {}
        #self.FetchAgentInfo()
        
        color_pool = [(255, 0, 0), (255, 128, 0), (255,255,0), (0,255,0), (0,255,255), (0,0,255), (178,102,255), (255,0,255), (13, 125, 143)]
        
        rospy.Subscriber("/target", TargetInfoArray, self.TargetCallback)

        for i in range(self.total_agents):
            self.agent_sensor_scores[i]     = {}
            self.agent_sensor_weights[i]    = {}
            self.agent_score_plt[i]         = []
            self.agent_failure[i]           = False
            
            try:
                self.color[i] = color_pool[i]
            except:
                self.color[i] = list(np.random.choice(range(255),size=3))
            
            rospy.Subscriber("/agent_"+str(i)+"/visualize/sensor_weights", WeightArray, self.WeightCB(i))           
            rospy.Subscriber("/agent_"+str(i)+"/visualize/valid_sensors", ValidSensors, self.ValidSensorCB(i))
            rospy.Subscriber("/agent_"+str(i)+"/visualize/sensor_scores", WeightArray, self.SensorScoresCB(i))
            rospy.Subscriber("/agent_"+str(i)+"/visualize/total_score", Float64, self.TotalScoreCB(i))
            rospy.Subscriber("/agent_"+str(i)+"/visualize/pose", Pose, self.PoseCB(i))
            rospy.Subscriber("/agent_"+str(i)+"/failure", Int16, self.FailureCB(i))
                    
        self.window_size = self.size*2
        self.display = pygame.display.set_mode(self.window_size)
        self.display.fill((0,0,0))
        self.blockSize = int(self.window_size[0]/self.size[0])

        # define the codec and create a video writer object
        self.cnt = 0
        self.images = []
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        self.prefix = '~/research_ws/src/voronoi_cbsa/result/'
        if not os.path.exists(self.prefix + self.timestr):
            os.makedirs(self.prefix + self.timestr)
        
    def FetchAgentInfo(self):

        for agent in range(self.total_agents):
            # General Agent's Settings
            prefix = "agent_" + str(agent) + "/Controller"
            id                      = rospy.get_param(prefix+"/id", default=0)
            camera_valid            = rospy.get_param(prefix+"/camera", default=0)
            manipulator_valid       = rospy.get_param(prefix+"manipulator", default=0)
            smoke_detector_valid    = rospy.get_param(prefix+"/smoke_detector", default=0)
            valid_sensor            = {'camera'         : camera_valid,
                                        'manipulator'    : manipulator_valid,
                                        'smoke_detector' : smoke_detector_valid}
            
            if camera_valid:
                # Camera Settings
                per_x               = rospy.get_param("~per_x", default=0)
                per_y               = rospy.get_param("~per_y", default=0)
                init_perspective    = np.array([float(per_x), float(per_y)])
                angle_of_view       = rospy.get_param("~angle_of_view")
                range_limit         = rospy.get_param("~desired_range")
                camera_variance     = rospy.get_param("~camera_variance", default=1)
                
                camera_info         = { 'perspective'    : init_perspective,
                                        'angle of view'  : angle_of_view,
                                        'desired range'  : range_limit,
                                        'variance'       : camera_variance}
            else:
                camera_info = None
            
            self.agent_camera_info[agent] = camera_info
            
            if manipulator_valid:
                # Manipulator Settings
                arm_length          = rospy.get_param("~arm_length", default = 1)
                approx_param        = rospy.get_param("~approx_param", default=20)
                
                manipulator_info    = { 'arm length'     : arm_length,
                                        'k'              : approx_param}
            else:
                manipulator_info = None
            
            self.agent_manipulator_info[agent] = manipulator_info

            if smoke_detector_valid:
                # Smoke Detector Settings
                smoke_variance      = rospy.get_param("~smoke_variance", default=1)
                smoke_detector_info = { 'variance'       : smoke_variance } 
            else:
                smoke_detector_info = None
                
            self.agent_smoke_detector_info[agent] = smoke_detector_info
            
            self.agent_info[agent] = {'camera_info'         : camera_info,
                                      'manipulator_info'    : manipulator_info,
                                      'smoke_detector_info' : smoke_detector_info}
            
    def TargetCallback(self, msg):
        self.target_received = True
        self.targets = []

        for target in msg.targets:
            pos_x = target.position.x
            pos_y = self.map_size[1] - target.position.y  # 反轉 Y 軸
            pos = np.array([pos_x, pos_y])

            std = target.standard_deviation
            weight = target.weight

            vel_x = target.velocity.linear.x
            vel_y = target.velocity.linear.y
            vel = np.array([vel_x, -vel_y])  # 速度的 Y 軸也需要反轉

            requirements = [target.required_sensor[i] for i in range(len(target.required_sensor))]
            self.targets.append([pos, std, weight, vel, target.id, requirements])

            
    def FailureCB(self, id):
        def callback(msg):
            if msg.data == 1:
                self.agent_failure[id] = True
        
        return callback
    
    def SensorScoresCB(self, id):
        def callback(msg):
            # msg 是 WeightArray 型別，裡面只有 weights[]，沒有 sensors[]
            for w in msg.weights:
                # w.type, w.score 分別代表感測器類型 (camera, manipulator等) 與分數
                self.agent_sensor_scores[id][w.type] = w.score
        return callback

    
    def TotalScoreCB(self, id):
        def callback(msg):
            self.agent_scores[id] = msg.data

        return callback
    
    def ValidSensorCB(self, id):
        def callback(msg):
            self.agent_valid_sensors[id] = []
            for data in msg.data:
                self.agent_valid_sensors[id].append(data) 

        return callback
    
    def PoseCB(self, id):
        def callback(msg):

            self.agent_pos[id] = np.asarray([msg.position.x, self.map_size[1] - msg.position.y])
            self.agent_per[id] = np.asarray([msg.orientation.x, msg.orientation.y])

        return callback
    
    def WeightCB(self, id):
        def callback(msg):
            for sensor in msg.weights:
                self.agent_sensor_weights[id][sensor.type] = {}
            
            for sensor in msg.weights:
                self.agent_sensor_weights[id][sensor.type][sensor.event_id] = sensor.score
                
        return callback
    
    def Update(self):
        def ComputeEventDensity(map_size, target, grid_size, role):

            event = np.zeros(self.size)

            for i in range(len(target)):
                if role in target[i][5] or role == 'All':
                    x, y = np.mgrid[0:map_size[0]:grid_size[0], 0:map_size[1]:grid_size[1]]
                    xy = np.column_stack([x.flat, y.flat])
                    mu = np.array(target[i][0])
                    sigma = np.array([target[i][1], target[i][1]])
                    covariance = np.diag(sigma**2)
                    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
                    event += z.reshape(x.shape)
             
            return event
            #return np.ones(self.size)
        
        def ComputeSensorVoronoi(role, agents_pos, global_event):
            x_coords, y_coords = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]), indexing='ij')

            if role == "All":
                
                total_cost = np.full(self.size, np.inf)
                sensor_voronoi = np.full(self.size, -1)

                for agent in agents_pos.keys():
                    if not self.agent_failure[agent]:
                        pos_self = agents_pos[agent]
                        grid_size = self.grid_size

                        cost = (((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2))*global_event
                        sensor_voronoi = np.where(cost < total_cost, agent, sensor_voronoi)
                        total_cost = np.where(cost < total_cost, cost, total_cost)
                    
            else:
                
                total_cost = np.full(self.size, np.inf)
                sensor_voronoi = np.full(self.size, -1)
                    
                for agent in agents_pos.keys():
                    if agent in self.agent_valid_sensors.keys() and not self.agent_failure[agent]:
                        if role in self.agent_valid_sensors[agent]:
                            pos_self = agents_pos[agent]
                            grid_size = self.grid_size
                            
                            weight  = 0
                            if role in self.agent_sensor_weights[agent]:
                                for event in self.agent_sensor_weights[agent][role].keys():
                                    weight += self.agent_sensor_weights[agent][role][event]
                            
                            cost = (((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2)
                                    - weight**2)*global_event#self.ComputeCost(role, pos_self, grid_size, x_coords, y_coords)
                            sensor_voronoi = np.where(cost < total_cost, agent, sensor_voronoi)
                            total_cost = np.where(cost < total_cost, cost, total_cost)

            return sensor_voronoi.transpose()
        
        def ComputeSensorFootprint(role, agent_pos, global_event, agent_info):
            for agent in self.agent_pos.keys():
                pass
            
        self.display.fill((0,0,0))
        
        if self.target_received:
            event = ComputeEventDensity(self.map_size, self.targets, self.grid_size, self.plot_role)
            event_plt = ((event - event.min()) * (1/(event.max() - event.min()) * 255)).astype('uint8')
            
            if self.plot_type == 'voronoi':
                voronoi_plt = ComputeSensorVoronoi(self.plot_role, self.agent_pos.copy(), event.copy())
                for x_map,x in enumerate(range(0, self.window_size[0], self.blockSize)):
                    for y_map,y in enumerate(range(0, self.window_size[1], self.blockSize)):
                        
                        id = voronoi_plt[y_map, x_map]
                        rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
                        pygame.draw.rect(self.display, (event_plt[x_map,y_map],event_plt[x_map,y_map],event_plt[x_map,y_map]), rect, 1)
                        
                        if id != -1:
                            rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
                            w = 0.3
                            color = [0, 0, 0]
                            color[0] = w*self.color[id][0] + (1-w)*event_plt[x_map,y_map]
                            color[1] = w*self.color[id][1] + (1-w)*event_plt[x_map,y_map]
                            color[2] = w*self.color[id][2] + (1-w)*event_plt[x_map,y_map]
                            pygame.draw.rect(self.display, color, rect, 1)
                            
            font = pygame.font.Font('freesansbold.ttf', 30)
            text = font.render(self.plot_role, True, (255,255,255))
            textRect = text.get_rect()
            textRect.center = (100, 50)
            self.display.blit(text, textRect)
            name_pool = ['\u03b1', '\u03b2', '\u03b4']
            for i in range(len(self.targets)):
                if self.plot_role in self.targets[i][5] or self.plot_role == 'All': 
                    center =  np.array(self.targets[i][0])/self.grid_size*self.blockSize
                    offset = 15
                    pygame.draw.polygon(self.display, (0, 0, 0), 
                                                    ((int(center[0]), int(center[1]) - offset),  # <-- 轉換為 int
                                                        (int(center[0]) - offset, int(center[1]) + offset),
                                                        (int(center[0]) + offset, int(center[1]) + offset)))

                    
                    font = pygame.font.Font('freesansbold.ttf', 20)
                    context_sensor = str(i) + ": " + "{"
                    for k,sensor in enumerate(self.sensor_pool):
                        if sensor in self.targets[i][5]:
                            context_sensor += name_pool[k] + " "
                    context_sensor += "}"
                            
                    text = font.render(context_sensor, True, (0,0,0))
                    textRect = text.get_rect()
                    textRect.center = (int(center[0]), int(center[1]) - 30)  # <-- 轉換為 int
                    self.display.blit(text, textRect)
            
            #total_score = 0
            #score_ready = True
            for id in self.agent_pos.keys():
                if not self.agent_failure[id]:
                    if id in self.agent_scores.keys():
                        self.agent_score_plt[id].append(self.agent_scores[id])
                    
                    # 計算 agent 的中心座標
                    center = self.agent_pos[id] / self.grid_size * self.blockSize

                    # 將 center 數組的分量轉換為純 Python int
                    try:
                        center_x = int(center[0].item())
                    except AttributeError:
                        center_x = int(center[0])
                    try:
                        center_y = int(center[1].item())
                    except AttributeError:
                        center_y = int(center[1])

                    # 繪製 sensor 信息
                    font = pygame.font.Font('freesansbold.ttf', 15)
                    context_sensor = "{"
                    if id in self.agent_valid_sensors.keys():
                        for k, sensor in enumerate(self.sensor_pool):
                            if sensor in self.agent_valid_sensors[id]:
                                context_sensor += name_pool[k] + " "
                    context_sensor += "}"
                    text = font.render(context_sensor, True, self.color[id])

                    # 計算文本顯示位置：向上偏移 20 個像素
                    upper = center_y - 20 if center_y - 20 > 0 else center_y + 20
                    # 確保 new_center 是一個包含兩個 int 的元組
                    new_center = (center_x, int(upper))
                    ## print("new_center:", new_center)  # 可用來檢查輸出
                    textRect = text.get_rect(center=new_center)
                    self.display.blit(text, textRect)


                    # 绘制 agent 的 id 信息
                    text = font.render(str(id), True, self.color[id])
                    upper2 = center_y - 40 if center_y - 40 > 0 else center_y + 40
                    new_center2 = (center_x, int(upper2))
                    textRect = text.get_rect(center=new_center2)
                    self.display.blit(text, textRect)

                    
                    # Draw agent's position
                    if id in self.agent_valid_sensors.keys():

                        if self.plot_role in self.agent_valid_sensors[id]:
                            width = 0
                        elif self.plot_role == "All":
                            width = 0
                        else:
                            width = 2
                        # 計算中心座標，並將其轉為整數或正確的 tuple
                        center_pos = self.agent_pos[id] / self.grid_size * self.blockSize
                        center_pos = (int(center_pos[0]), int(center_pos[1]))

                        # 使用位置參數呼叫 circle()
                        pygame.draw.circle(self.display, self.color[id], center_pos, 10, int(width))

                                            
                        if 'camera' in self.agent_valid_sensors[id]:
                            # Draw agent's perspective
                            pos = self.agent_pos[id]/self.grid_size*self.blockSize
                            per = self.agent_per[id]/self.grid_size*self.blockSize
                            angle = np.arctan2(self.agent_per[id][1], self.agent_per[id][0])  # 計算方向角    #############有改#############
                            pygame.draw.line(self.display, self.color[id], pos, 
                                                (pos[0] + 20 * np.cos(angle), pos[1] - 20 * np.sin(angle)), 2)

                else:
                    # Draw agent's position
                    if id in self.agent_valid_sensors.keys():
                        width = 1
                        if self.plot_role in self.agent_valid_sensors[id]:
                            width = 0
                        elif self.plot_role == "All":
                            width = 0
                        else:
                            width = 2
                        pygame.draw.circle(self.display, (125, 125, 125), 
                            self.agent_pos[id]/self.grid_size*self.blockSize, radius=10, width=width) 
                    
            # if score_ready:
            #     font = pygame.font.Font('freesansbold.ttf', 32)
            #     text = font.render(str(np.round(total_score, 3)), True, (255,255,255))
            #     textRect = text.get_rect()
            #     textRect.center = (100, 80)
            #     self.display.blit(text, textRect)

            #     self.total_score.append(total_score)
            #     self.frame.append(self.cnt)
            #     self.cnt += 1
                
        pygame.display.flip()
        
        # transform the pixels to the format used by open-cv
        pixels = cv2.rotate(pygame.surfarray.pixels3d(self.display), cv2.ROTATE_90_CLOCKWISE)
        pixels = cv2.flip(pixels, 1)

        self.images.append(pixels)
        #self.plot()
        
    def plot(self):
        plt.plot(self.frame, self.total_score, color = 'b')
        plt.title("Total score")
        plt.xlabel('Frame id')
        plt.ylabel('Score')
        
        plt.draw()  
        plt.pause(0.01)  
    
    #def plot_graph(self):
        
        
    def End(self):
        pass
        #imageio.mimsave(self.prefix+'4_CB.gif', self.images)
 
def KillCB(msg):
    global kill
    if msg.data == 1:
        kill = True  
         
if __name__=="__main__":
    global kill
    kill = False
    
    rospy.init_node('visualizer', anonymous=False, disable_signals=True)
    
    total_agents = rospy.get_param('/total_agents', '1')
    rospy.Subscriber("/kill", Int16, KillCB)
    record_clk_pub = rospy.Publisher("/record_clk", Int16, queue_size=1)
    
    pygame.init()

    visualizer = Visualize2D()
    
    msg = Int16()
    clk = 1
    target1_pos_x = []
    target1_pos_y = []
    target2_pos_x = []
    target2_pos_y = []
    target3_pos_x = []
    target3_pos_y = []
    target4_pos_x = []
    target4_pos_y = []
    
    frame = []
    cnt = 0
    while not rospy.is_shutdown():
        if kill:
            visualizer.End()
            break
        
        visualizer.Update()
        
        if visualizer.target_received and len(visualizer.targets) >= 1:
            clk *= -1
            msg.data = clk
            record_clk_pub.publish(msg)
            frame.append(cnt)
            cnt += 1

            # 假設 targets[i][0] 為位置向量 (x, y)
            target1_pos_x.append( visualizer.targets[0][0][0] )
            target1_pos_y.append( visualizer.targets[0][0][1] )
            
            
            
           
        
    pygame.quit()   
    plt_dict = {'frame_id'              : frame,
                "target0_pos_x"                 : target1_pos_x,
                "target0_pos_y"                 : target1_pos_y,
                #"target1_pos_x"                 : target2_pos_x,
                #"target1_pos_y"                 : target2_pos_y,
                #"target2_pos_x"                 : target3_pos_x,
                #"target2_pos_y"                 : target3_pos_y,
                #"target4_pos_x"                 : target4_pos_x,
                #"target4_pos_y"                 : target4_pos_y
                }
        
    save_path = "/home/heish/research_ws/src/voronoi_cbsa/result/10/"
    isExist = os.path.exists(save_path)
    if not isExist:
        os.makedirs(save_path)
        
    df = pd.DataFrame.from_dict(plt_dict) 
    df.to_csv (save_path + "target_path.csv", index=False, header=True)
    # plt_dict = {'frame_id': visualizer.frame,
    #             'total score': visualizer.total_score}
    
    # for id in range(total_agents):
    #     if len(visualizer.agent_score_plt[id]) > 0:
    #         plt_dict[str(id)+"'s score"] = visualizer.agent_score_plt[id]
        
    # df = pd.DataFrame.from_dict(plt_dict) 
    # df.to_csv (r'~/research_ws/src/voronoi_cbsa/result/non-coop-3.csv', index=False, header=True)