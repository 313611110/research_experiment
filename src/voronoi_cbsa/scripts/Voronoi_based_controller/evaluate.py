import os 
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Circle
from scipy.stats import multivariate_normal
import cv2

class Evaluator:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 24)
        self.ax.set_ylim(0, 24)
        pass
    
    def SinglePlotCoop(self, paths, agent_id = -1):
        global type
        def getint(name):
            _, num = name.split('/'+type+'/')
            num, _ = num.split('.')
            return int(num)
        
        fig  = plt.figure()
        ax1 = fig.add_subplot()
        
        color_pool = {'coop': 'r', 'non-coop': 'g'}
        
        if agent_id != -1:
            for type  in paths.keys():
                df = pd.read_csv(paths[type]+type+"_"+str(agent_id)+".csv")
                col = str(agent_id)+"'s score"
                data = np.array((df['frame_id'].values, df[col].values))
                label = type
                ax1.plot(data[0][:], np.log10(data[1][:]), color = color_pool[type], label=label)
        
        else:
            for type in paths.keys():
                frames = []
                scores = {}
                max_len = 0
                
                files = sorted(glob.glob(os.path.join(paths[type], "*.csv")),key=getint)
                for i, file in enumerate(files):
                    print(file)
                    df = pd.read_csv(file)
                    col = str(i)+"'s score"
                    data = np.array((df['frame_id'].values, df[col]))
                    scores[i] = (data[1][:])
                    
                    if len(data[0][:]) > max_len:
                        max_len = len(data[0][:])
                        frames = data[0][:]
                
                total_mat = []  
                for i in scores.keys():
                    tmp = np.squeeze(np.zeros((max_len, 1)))
                    tmp[0:len(scores[i][:])] = scores[i][:]
                    #ax1.scatter(len(scores[i][:]), scores[i][-1], label = str(i)+"'s failure")
                    total_mat = tmp if i == 0 else np.dstack([total_mat, tmp])
                print(total_mat)
                result = np.squeeze(np.sum(total_mat, axis=2))

                label = type
                ax1.plot(frames, result, color = color_pool[type], label=label)
                
                
        title = "Score of agent " + str(agent_id) if agent_id != -1 else "Exp3 Average score of 10 trials"
        ax1.set_xlabel("frame id")
        ax1.set_ylabel("utility")
        ax1.set_title(title)
        
        plt.legend() 
        plt.show()
    
    def average_score(self):
        def getint(name):
            _, num = name.split('/res/')
            num, _ = num.split('.')
            return int(num)
        
        fig  = plt.figure()
        ax1 = fig.add_subplot()
        data_length = 97
        trials = [0]
        
        color_pool = {'coop': 'r', 'non-coop': 'g'}
        results = {'coop': np.zeros(data_length), 'non-coop': np.zeros(data_length)}
        for trial_id in trials:
            paths = {'coop'     : "/home/andrew/research_ws/src/voronoi_cbsa/result/andre/coop/res/", 
                     'non-coop' : "/home/andrew/research_ws/src/voronoi_cbsa/result/andre/non-coop/res/"}
            
            for type in paths.keys():
                    frames = []
                    scores = {}
                    max_len = 0
                    
                    files = sorted(glob.glob(os.path.join(paths[type], "*.csv")), key=getint)
                    for i, file in enumerate(files):
                        print(file)
                        df = pd.read_csv(file)
                        col = str(i)+"'s score"
                        data = np.array((df['frame_id'].values, df[col]))
                        scores[i] = (data[1][:])
                        
                        if len(data[0][:]) > max_len:
                            max_len = len(data[0][:])
                            frames = data[0][:]
                    
                    total_mat = []  
                    for i in scores.keys():
                        tmp = np.squeeze(np.zeros((max_len, 1)))
                        tmp[0:len(scores[i][:])] = scores[i][:]
                        total_mat = tmp if i == 0 else np.dstack([total_mat, tmp])

                    result = np.squeeze(np.sum(total_mat, axis=2))
                    results[type] += result[:data_length]
        label = type
        ax1.plot(frames[:data_length], results['coop']/len(trials), "-", color = color_pool['coop'], label='coop')
        label = type
        ax1.plot(frames[:data_length], results['non-coop']/len(trials), "--", color = color_pool['non-coop'], label='non-coop')
        
        non_10 = 0
        non_90 = 0
        coop_10 = 0
        coop_90 = 0
        for i,val in enumerate(results['non-coop']/len(trials)):
            if val >= np.max(results['non-coop']/len(trials))*0.1:
                non_10 = i
                break
        
        for i,val in enumerate(results['non-coop']/len(trials)):
            if val >= np.max(results['non-coop']/len(trials))*0.9:
                non_90 = i
                break
        
        for i,val in enumerate(results['coop']/len(trials)):
            if val >= np.max(results['coop']/len(trials))*0.1:
                coop_10 = i
                break
        
        for i,val in enumerate(results['coop']/len(trials)):
            if val >= np.max(results['coop']/len(trials))*0.9:
                coop_90 = i
                break
        
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 24

        title = "Simulation 3"
        ax1.set_xlabel("frame id", fontsize=14)
        ax1.set_ylabel("H(p(t))", fontsize=14)
        ax1.set_title(title, fontsize=20)
        
        idx = max([coop_90,non_90])
        print(idx)
        print("Transient:")
        print("Coop: " + str(np.sum(results['coop'][:idx])) + " Non: " + str(np.sum(results['non-coop'][:idx])))
        print(np.sum(results['coop'][:idx]/len(trials))/np.sum(results['non-coop'][:idx]/len(trials)))
        print("Steady-state:")
        print("Coop: " + str(np.sum(results['coop'][-1:])) + " Non: " + str(np.sum(results['non-coop'][-1:])))
        print(np.mean(results['coop'][-1:]/len(trials))/np.mean(results['non-coop'][-1:]/len(trials)))
        print("Transient:")
        print("Coop: " + str(np.sum(results['coop'])) + " Non: " + str(np.sum(results['non-coop'])))
        print(np.mean(results['coop']/len(trials))/np.mean(results['non-coop']/len(trials)))
        plt.legend() 
        plt.show()
        
    def MultiPlotCoop(self, paths):
        files = {'coop'     : sorted(glob.glob(os.path.join(paths['coop'], "*.csv"))),
                 'non-coop' : sorted(glob.glob(os.path.join(paths['non-coop'], "*.csv")))}
        
        agent_num = len(files['coop'])
        fig, ax  = plt.subplots(math.ceil(agent_num/2), 2, constrained_layout = True)
        color_pool = {'coop': 'r', 'non-coop': 'g'}
        
        for type in paths.keys():
            for i, file in enumerate(files[type]):
                df = pd.read_csv(file)
                col = str(i)+"'s score"
                data = np.array((df['frame_id'].values, df[col].values))
                
                label = type
                title = "Score of agent " + str(i)
                ax[i % math.ceil(agent_num/2), i // math.ceil(agent_num/2)].plot(data[0][:], (data[1][:]), color = color_pool[type], label=label)
                ax[i % math.ceil(agent_num/2), i // math.ceil(agent_num/2)].legend(loc="upper left")
                ax[i % math.ceil(agent_num/2), i // math.ceil(agent_num/2)].set_xlabel("frame id")
                ax[i % math.ceil(agent_num/2), i // math.ceil(agent_num/2)].set_ylabel("utility")
                ax[i % math.ceil(agent_num/2), i // math.ceil(agent_num/2)].set_title(title)
        
        plt.legend() 
        plt.show()
    
    def PlotTrajectory(self, paths):
        files = {'coop'     : sorted(glob.glob(os.path.join(paths['coop'], "*.csv"))),
                 'non-coop' : sorted(glob.glob(os.path.join(paths['non-coop'], "*.csv")))}
        
        fig, ax  = plt.subplots(1, 2, constrained_layout = True)
        color_pool = [np.array((255, 0, 0)), np.array((255, 128, 0)), np.array((255,255,0)),
                      np.array((0,255,0)), np.array((0,255,255)), np.array((0,0,255)),
                      np.array((178,102,255)), np.array((255,0,255)), np.array((13, 125, 143))]
        for j, type in enumerate(paths.keys()):
            for i, file in enumerate(files[type]):
                df = pd.read_csv(file)
                col_x = "pos_x"
                col_y = "pos_y"
                data = np.array((df[col_x].values, df[col_y].values))
                
                label = "agent " + str(i)
                ax[j].scatter(data[1][0], data[0][0], marker="s",s=50,color = color_pool[i]/255*0.5, label=label+" start", zorder=2)
                ax[j].scatter(data[1][-1], data[0][-1], marker="^",s=50, color = color_pool[i]/255*0.5, label=label+" end", zorder=2)
                ax[j].plot(data[1][:], (data[0][:]), color = color_pool[i]/255, label=label, zorder=1)
                
            
            title = "Trajectories of " + type
            ax[j].legend(loc="upper left")
            ax[j].set_xlabel("x(m)")
            ax[j].set_ylabel("y(m)")
            ax[j].set_title(title)
        
        plt.legend() 
        plt.show()
    
    
    def Animation(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 24)
        self.ax.set_ylim(0, 24)
        
        ani = animation.FuncAnimation(self.fig, self.PlotVoronoi, frames = 398, interval=10)
        print('saving')
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'))
        ani.save("Sim_non_coop.mp4", writer=writer)
    
    def PlotVoronoi(self, data): #(self,data)
        global type
        voronoi = True
        type='non-coop'
        trial_id = '6'
        paths = {'coop'     : "/home/andrew/research_ws/src/voronoi_cbsa/result/4_static_targets/"+trial_id+"/coop/unbalance/", 
                'non-coop' : "/home/andrew/research_ws/src/voronoi_cbsa/result/4_static_targets/"+trial_id+"/non-coop/unbalance/"}
        
        # paths = {'coop'     : "/home/andrew/Downloads/andre/coop/", 
        #      'non-coop' : "/home/andrew/Downloads/andre/non-coop/"}
        
        def getint(name):
            _, num = name.split('/'+'unbalance'+'/')
            num, _ = num.split('.')
            return int(num)
        
        files = sorted(glob.glob(os.path.join(paths[type], "*.csv")), key=getint)
        
        total_S = [[0,2],[0,1],[0,2],[0,1]]
        agent_S = [[0],[0],[0],[0],[0],[0],[0,1],[0,2],[2],[2],[1],[1]]
        color_pool = ['green', 'blue', 'orange']
        name_pool = ['camera', 'smoke_detector', 'manipulator']
        finished_sensor = []
        target_pos = [(5,5),(19,19),(5,19),(19,5)]
        target_sigma = [0.8,0.8,0.8,0.8]
        
        self.ax.clear()
        frame = data
        # print(frame)
        # target_pos = [np.array([6*np.cos(((-1)**0)*(frame*2.15)/180*np.pi) + 12, 6*np.sin(((-1)**0)*(frame*2.15)/180*np.pi) + 12]), np.array([6*np.cos(((-1)**1)*(frame*2.15)/180*np.pi) + 12, 6*np.sin(((-1)**1)*(frame*2.15)/180*np.pi) + 12])]
        for target_id in [0,1,2,3]:
            for target_sensor in total_S[target_id]:
                if target_sensor not in finished_sensor:
                    finished_sensor.append(target_sensor)
                    points = []
                    for i, file in enumerate(files):
                        if target_sensor in agent_S[i]:
                            df = pd.read_csv(file)
                            col_x = "pos_x"
                            col_y = "pos_y"
                            
                            data = np.array((df[col_x].values, df[col_y].values))
                            
                            points.append(np.array([data[0][frame], data[1][frame]]))
                            self.ax.scatter(data[0][frame], data[1][frame], marker="s",color = 'black', s=25, zorder=2)
                            
                            if target_sensor == 0:
                                radius = 5
                                angle = 60
                
                                dx = df["per_x"].values[frame]
                                dy = df["per_y"].values[frame]
                                direction_radians = np.arctan2(dy, dx)
                                direction_degrees = np.degrees(direction_radians)
                                if direction_degrees < 0:
                                    direction_degrees += 360
                                theta1 = np.radians(direction_degrees - angle / 2)
                                theta2 = np.radians(direction_degrees + angle / 2)
                                theta = np.linspace(theta1, theta2, 100)
                                cx = data[0][frame]
                                cy = data[1][frame]
                                x = radius * np.cos(theta) + data[0][frame]
                                y = radius * np.sin(theta) + data[1][frame]
                                sector_x = np.concatenate(([cx], x, [cx]))
                                sector_y = np.concatenate(([cy], y, [cy]))

                                # Use the fill function to create the filled area of the FoV
                                self.ax.fill(sector_x, sector_y, color=color_pool[target_sensor], alpha=0.2, edgecolor='none')                    

                                self.ax.set_aspect('equal', 'box')  # Maintain aspect ratio
                            
                            if target_sensor == 1:
                                radius = 2
                                angle = 360
                                cx = data[0][frame]
                                cy = data[1][frame]
                                circle = Circle((cx, cy), radius, color=color_pool[target_sensor], alpha=0.2)

                                # Add the circle to the axes
                                self.ax.add_patch(circle)

                                # Set aspect of the plot to be equal
                                self.ax.set_aspect('equal', 'box')

                                # Set limits to make sure the whole circle is visible
                                # ax.set_xlim([cx - radius - 1, cx + radius + 1])
                                # ax.set_ylim([cy - radius - 1, cy + radius + 1])
                                
                            if target_sensor == 2:
                                radius = 1
                                angle = 360

                                cx = data[0][frame]
                                cy = data[1][frame]
                                circle = Circle((cx, cy), radius, color=color_pool[target_sensor], alpha=0.2, edgecolor='none')

                                # Add the circle to the axes
                                self.ax.add_patch(circle)

                                # Set aspect of the plot to be equal
                                self.ax.set_aspect('equal', 'box')

                                # Set limits to make sure the whole circle is visible
                                # ax.set_xlim([cx - radius - 1, cx + radius + 1])
                                # ax.set_ylim([cy - radius - 1, cy + radius + 1])
                    
                    labeled = True
                    if len(points) >= 3 and voronoi:
                        self.ax.set_xlim([0,24])
                        self.ax.set_ylim([0,24])
                        vor = Voronoi(points)
                        regions, vertices = voronoi_finite_polygons_2d(vor, radius=150)
                        for region in regions:
                            polygon = vertices[region]
                            # Ensure it's a closed polygon by repeating the first vertex at the end
                            polygon_closed = np.vstack((polygon, polygon[0]))
                            self.ax.plot(polygon_closed[:, 0], polygon_closed[:, 1], ':', color = color_pool[target_sensor], label = name_pool[target_sensor], alpha=0.7) if not labeled\
                                                else self.ax.plot(polygon_closed[:, 0], polygon_closed[:, 1], ':', color = color_pool[target_sensor], alpha=0.7)
                            labeled = True
                                    
                        # voronoi_plot_2d(vor, show_vertices=False, line_colors=color_pool[target_sensor],
                        #              line_width=2, line_alpha=0.6, point_size=2, ax=ax)

            ComputeEventDensity(target_pos[target_id], target_sigma[target_id], total_S[target_id], self.ax)
                
        self.ax.legend(loc="upper left")
        self.ax.set_facecolor('gray')
        self.ax.set_xlabel("x(m)")
        self.ax.set_ylabel("y(m)")
        self.ax.set_title("Simulation 2 Start Configuration")  
        plt.show()
        
def ComputeEventDensity(target_pos, target_sigma, target_sensor, ax):
        x, y = np.mgrid[0:24:0.01, 0:24:0.01]
        xy = np.column_stack([x.flat, y.flat])
        mu = np.array(target_pos)
        sigma = np.array([target_sigma,target_sigma])
        covariance = np.diag(sigma**2)
        z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
        event = z.reshape(x.shape)
            
        #return np.ones(self.size)
        ax.contour(x, y, event, alpha=0.3)
        color_pool = ['green', 'blue', 'orange']
        icon_pool = ['^','o','*']
        name_pool = ['camera', 'smoke_detector', 'manipulator']
        cnt = [(-0.3,-0.3),(0,0.3),(0.3,-0.3)]
        for s in target_sensor:
            ax.scatter(target_pos[0] + cnt[s][0],target_pos[1]+cnt[s][1],  marker=icon_pool[s],color = color_pool[s], s=50, zorder=2, edgecolors='black', linewidth=1)

        return 
           
# Function to create a bounded Voronoi diagram
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # Finite region
            new_regions.append(vertices)
            continue

        # Reconstruct a Voronoi region with one or more points at infinity
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # Finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # Sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # Finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

if __name__ == "__main__":
    eval = Evaluator()
    
    trial_id = '0'
    paths = {'coop'     : "/home/andrew/research_ws/src/voronoi_cbsa/result/1_dynamic_target/"+trial_id+"/non-coop/unbalance/", 
             'non-coop' : "/home/andrew/research_ws/src/voronoi_cbsa/result/1_dynamic_target/"+trial_id+"/coop/unbalance/"}
    # paths = {'coop'     : "/home/andrew/Downloads/andre/coop/", 
    #          'non-coop' : "/home/andrew/Downloads/andre/non-coop/"}
    #eval.SinglePlotCoop(paths, -1)
    eval.average_score()
    #eval.MultiPlotCoop(paths)
    #eval.PlotTrajectory(paths)
    #eval.PlotVoronoi(data=0)
    #eval.Animation()


    
    
    