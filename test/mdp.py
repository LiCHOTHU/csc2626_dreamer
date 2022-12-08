import wandb
import argparse
import os
import torch
import numpy as np
import gym
from dreamerv2.utils.wrapper import GymMinAtar, OneHotAction
from dreamerv2.training.config import MinAtarConfig
from dreamerv2.training.slot_config import SlotMinAtarConfig, SlotSafetyConfig
from dreamerv2.training.slot_config_1slot import SlotMinAtarConfig_1slot, SlotSafetyConfig_1slot
from dreamerv2.training.trainer import Trainer
from dreamerv2.training.evaluator import Evaluator

def main(args):
    wandb.login()
    env_name = args.env

    '''make dir for saving results'''
    exp_name = env_name
    if args.exp_suffix is not None:
        exp_name += "_{}".format(args.exp_suffix)
    
    result_dir = os.path.join(args.result_root, exp_name)
    model_dir = os.path.join(result_dir, 'r{}_models'.format(args.id))  #dir to save learnt models
    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device:
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
    print('using :', device)

    if env_name == "safety":
        from dreamerv2.envs.navigation.engine import Engine
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
        env.reset()
    else:
        env = OneHotAction(GymMinAtar(env_name))
        
    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]
    obs_dtype = bool
    action_dtype = np.float32
    batch_size = args.batch_size
    seq_len = args.seq_len

    if args.slot:
        from dreamerv2.training.slot_config import SlotMinAtarConfig, SlotSafetyConfig
        if env_name == "safety":
            CFG = SlotSafetyConfig
        else:
            CFG = SlotMinAtarConfig
    elif args.slot_1slot:
        if env_name == "safety":
            CFG = SlotSafetyConfig_1slot
        else:
            CFG = SlotMinAtarConfig_1slot
    else:
        CFG = MinAtarConfig
    config = CFG(
        env=env_name,
        obs_shape=obs_shape,
        action_size=action_size,
        obs_dtype = obs_dtype,
        action_dtype = action_dtype,
        seq_len = seq_len,
        batch_size = batch_size,
        model_dir=model_dir, 
        eval_render=False
    )

    config_dict = config.__dict__

    if args.slot or args.slot_1slot:
        from dreamerv2.training.slot_trainer import SlotTrainer
        TRN = SlotTrainer
    else:
        TRN = Trainer
    if args.slot or args.slot_1slot:
        from dreamerv2.training.slot_evaluator import SlotEvaluator
        EVL = SlotEvaluator
    else:
        EVL = Evaluator
        
    trainer = TRN(config, device)
    evaluator = EVL(config, device)

    with wandb.init(
            project = 'mastering MinAtar with world models', 
            config = config_dict, 
            name = exp_name, 
            dir = result_dir+"/wandb"
        ):
        """training loop"""
        print('...training...')
        train_metrics = {}
        trainer.collect_seed_episodes(env)
        obs, score = env.reset(), 0
        done = False
        prev_rssmstate = trainer.RSSM._init_rssm_state(1)
        prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
        episode_actor_ent = []
        scores = []
        best_mean_score = 0
        best_save_path = os.path.join(model_dir, 'models_best.pth')
        
        for iter in range(1, trainer.config.train_steps):  
            if iter%trainer.config.train_every == 0:
                train_metrics = trainer.train_batch(train_metrics)
            if iter%trainer.config.slow_target_update == 0:
                trainer.update_target()       
                
            if args.exp_suffix == 'debug':
                trainer.config.save_every = 100
                
            if iter%trainer.config.save_every == 0:
                save_path = os.path.join(model_dir, 'models_%d.pth' % iter)
                trainer.save_model(iter)

                # Evaluate the saved model and log the results
                eval_score = evaluator.eval_saved_agent(env, save_path)
                wandb.log({'eval_score': eval_score})

            with torch.no_grad():
                embed = trainer.ObsEncoder(torch.tensor(obs, dtype=torch.float32)[None, None].to(trainer.device))
                embed = embed.squeeze(0) 
                _, posterior_rssm_state = trainer.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                model_state = trainer.RSSM.get_model_state(posterior_rssm_state)
                action, action_dist = trainer.ActionModel(model_state)
                action = trainer.ActionModel.add_exploration(action, iter).detach()
                action_ent = torch.mean(action_dist.entropy()).item()
                episode_actor_ent.append(action_ent)

            next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
            score += rew

            if done:
                trainer.buffer.add(obs, action.squeeze(0).cpu().numpy(), rew, done)
                train_metrics['train_rewards'] = score
                train_metrics['action_ent'] =  np.mean(episode_actor_ent)
                wandb.log(train_metrics, step=iter)
                scores.append(score)
                if len(scores)>100:
                    scores.pop(0)
                    current_average = np.mean(scores)
                    if current_average>best_mean_score:
                        best_mean_score = current_average 
                        print('saving best model with mean score : ', best_mean_score)
                        save_dict = trainer.get_save_dict()
                        torch.save(save_dict, best_save_path)
                
                obs, score = env.reset(), 0
                done = False
                prev_rssmstate = trainer.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
                episode_actor_ent = []
            else:
                trainer.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), rew, done)
                obs = next_obs
                prev_rssmstate = posterior_rssm_state
                prev_action = action

    '''evaluating probably best model'''
    evaluator.eval_saved_agent(env, best_save_path)

if __name__ == "__main__":

    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help='mini atari env name')

    # For naming the experiment
    parser.add_argument("--exp_suffix", type=str, default=None, help='experiment name, appended to env name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument("--id", type=str, default=None, help='Experiment ID')

    # For saving chcckpoints
    parser.add_argument('--result_root', type=str, default='models', help='Directory to save models')
    
    parser.add_argument('--device', default='cuda', help='CUDA or CPU')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=50, help='Sequence Length (chunk length)')
    parser.add_argument('--slot', action='store_true')
    parser.add_argument('--slot_1slot', action='store_true')
    args = parser.parse_args()
    if args.slot_1slot:
        assert not args.slot, 'can only specify 1 args'
    main(args)
