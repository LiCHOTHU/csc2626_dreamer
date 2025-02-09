import argparse
import os
import torch
import numpy as np
import gym
from dreamerv2.utils.wrapper import GymMinAtar, OneHotAction, breakoutPOMDP, space_invadersPOMDP, seaquestPOMDP, asterixPOMDP, freewayPOMDP
from dreamerv2.training.config import MinAtarConfig
from dreamerv2.training.slot_config import SlotFreewayConfig, SlotAsterixConfig, SlotSafetyConfig
from dreamerv2.training.slot_config_1slot import SlotFreewayConfig_1slot, SlotAsterixConfig_1slot, SlotSafetyConfig_1slot
from dreamerv2.training.evaluator import Evaluator
from dreamerv2.training.slot_evaluator import SlotEvaluator

pomdp_wrappers = {
    'breakout':breakoutPOMDP,
    'seaquest':seaquestPOMDP,
    'space_invaders':space_invadersPOMDP,
    'asterix':asterixPOMDP,
    'freeway':freewayPOMDP,
}

def main(args):
    print(args)
    env_name = args.env
    if args.pomdp==1:
        exp_id = args.id + '_pomdp'
        PomdpWrapper = pomdp_wrappers[env_name]
        env = PomdpWrapper(OneHotAction(GymMinAtar(env_name)))
        print('using partial state info')
    else:
        exp_id = args.id
        env = OneHotAction(GymMinAtar(env_name))
        print('using complete state info')
    
    if args.eval_episode == 1:
        eval_render = True
    else:
        eval_render = False

    if torch.cuda.is_available() and args.device:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('using :', device)  
    
    result_dir = os.path.join('results', '{}_{}'.format(env_name, exp_id))
    model_dir = os.path.join(result_dir, 'models')                           

    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]
    obs_dtype = bool 
    action_dtype = np.float32

    if args.slot:
        if env_name == "safety":
            CFG = SlotSafetyConfig
        elif 'freeway' in env_name.lower():
            CFG = SlotFreewayConfig
        elif 'asterix' in env_name.lower():
            CFG = SlotAsterixConfig
        else:
            raise NotImplementedError
    elif args.slot_1slot:
        if env_name == "safety":
            CFG = SlotSafetyConfig_1slot
        elif 'freeway' in env_name.lower():
            CFG = SlotFreewayConfig_1slot
        elif 'asterix' in env_name.lower():
            CFG = SlotAsterixConfig_1slot
        else:
            raise NotImplementedError
    else:
        CFG = MinAtarConfig
    config = CFG(
        env=env_name,
        obs_shape=obs_shape,
        action_size=action_size,
        obs_dtype = obs_dtype,
        action_dtype = action_dtype,
        model_dir=model_dir, 
        eval_episode=args.eval_episode,
        eval_render=eval_render
    )

    if args.slot or args.slot_1slot:
        EVL = SlotEvaluator
    else:
        EVL = Evaluator
    evaluator = EVL(config, device)
    best_score = 0
    
    # for f in sorted(os.listdir(model_dir)):
    for f in ['models_best.pth']:
        print('evaluating model : ', f)
        eval_score = evaluator.eval_saved_agent(env,  os.path.join(model_dir, f))
        if eval_score > best_score:
            print('..saving model number')
            best_score=eval_score

    print('best mean evaluation score amongst stored models is : ', best_score)

if __name__ == "__main__":

    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help='mini atari env name')
    parser.add_argument('--eval_episode', type=int, default=10, help='number of episodes to eval')
    parser.add_argument("--id", type=str, default='0', help='Experiment ID')
    parser.add_argument("--eval_render", type=int, help='to render while evaluation')
    parser.add_argument("--pomdp", type=int, help='partial observation flag')
    parser.add_argument('--device', default='cuda', help='CUDA or CPU')
    parser.add_argument('--slot', action='store_true')
    parser.add_argument('--slot_1slot', action='store_true')
    args = parser.parse_args()
    if args.slot_1slot:
        assert not args.slot, 'can only specify 1 args'
    main(args)
