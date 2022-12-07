
import argparse
import os
import cv2
from pathlib import Path
from dreamerv2.envs.navigation.engine import Engine
from tqdm import trange
import numpy as np

def get_action(env):
    world_config = env.world_config_dict
    positions = []
    # Get position of all objects in environment
    for k, v in world_config['objects'].items():
        # print(k, v['pos'])
        positions.append(v['pos'])
    for k, v in world_config['geoms'].items():
        # print(k, v['pos'])
        positions.append(v['pos'])

    positions = np.asarray(positions)

    # Select a random position from positions as the goal
    goal = positions[np.random.randint(len(positions))]
    goal = goal[:2]

    # Get the current position of the agent
    ego_pos = env.robot_pos[:2]

    # Get vector from ego_pos to goal
    action = goal - ego_pos

    # Normalize the action vector
    action = action / np.linalg.norm(action) * np.random.uniform(0.1, 0.75)

    # Add noise to the action
    action += np.random.normal(0, 0.05, size=action.shape)

    return action


def main(args):
    config = {
        'robot_base': 'xmls/point2.xml',
        # 'robot_base': 'xmls/point.xml',
        'task': 'goal',
        'observe_goal_lidar': True,
        'observe_box_lidar': True,
        'observe_hazards': True,
        'observe_vases': True,
        'constrain_hazards': True,
        'lidar_max_dist': 3,
        'lidar_num_bins': 16,
        'hazards_num': 2,
        'vases_num': 2,
        'observation_flatten': False,
        'render_lidar_markers': False,
        'vases_size': 0.2,  # Half-size (radius) of vase object
        'robot_rot': 0,  # Override robot starting angle
    }

    env = Engine(config)

    # Run a random policy and visualize the results on the screen
    for ep in trange(1000):
        env.reset()
        done = False
        t = 0
        ep_folder = Path('./generated_dataset/ep_%06d/' % ep)
        if not os.path.exists(ep_folder):
            os.makedirs(ep_folder)

        while not done:
            if t == 0 or t % 20 == 0:
                action = get_action(env)

            next_observation, reward, done, info = env.step(action)

            # pprint(next_observation)
            print(reward)
            # print(done)
            # print(info)

            if args.save:
                if t % 5 == 0:
                    img = env.render(mode='rgb_array', camera_id=1)

                    # plt.imshow(img)
                    # plt.show()

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(str(ep_folder / ('obs_%06d.png' % (t // 5))), img)
            else:
                env.render(mode='human')

            t += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action="store_true",
                        help="If set, the rendered imges will be saved as dataset, otherwise they will be shown on the screen")

    args = parser.parse_args()
    main(args)