import os
import json
import argparse
import logging
from dotenv import load_dotenv
from datetime import datetime

from reinforcement import _version
from reinforcement.environment.envs import optionEnvironment
from reinforcement.training.trainer import DDPG
from reinforcement.utils.logging_utils import initialise_logger

project_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
env_file = os.path.join(project_path, "config.env")
load_dotenv(env_file)


logger = logging.getLogger("root")

def main():

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        "--experiment_no",
        type=str, 
        required=True,
        help="Please provide the experiment number")

    # Required arguments
    parser.add_argument(
        "--flow", 
        default= os.getenv("FLOW"), 
        type=str, 
        required=True,
        help="Choose between train, evaluate or train_evaluate")

    # Optional arguments
    parser.add_argument(
        "--risk_aversion", 
        default= os.getenv("RISK_AVERSION"), 
        type=bool, 
        required=False,
        help="Write it")

    # Optional arguments
    parser.add_argument(
        "--continuous_action", 
        default= os.getenv("CONT_ACTION"), 
        type=bool, 
        required=False,
        help="Write it")

    parser.add_argument(
        "--sabr_action", 
        default= os.getenv("SABR_ACTION"), 
        type=bool, 
        required=False,
        help="Write it")

    parser.add_argument(
        "--random_seed", 
        default= os.getenv("RANDOM_SEED"), 
        type=int, 
        required=False,
        help="Write it")

    parser.add_argument(
        "--num_contract", 
        default=os.getenv("NO_CONTRACT"), 
        type=int, 
        required=False,
        help="Write it")
    
    parser.add_argument(
        "--ttm", 
        default=os.getenv("TT_MATURITY"), 
        type=int,
        required=False,
        help="Write it")

    parser.add_argument(
        "--num_sim_train", 
        default=os.getenv("NO_OF_SIMUL_TRAIN"), 
        type=int, 
        required=False,
        help="Write it")

    parser.add_argument(
        "--num_sim_test", 
        default=os.getenv("NO_OF_SIMUL_TEST"), 
        type=int, 
        required=False,
        help="Write it")

    parser.add_argument(
        "--trad_freq", 
        default=os.getenv("TRADE_FREQ"), 
        type=int, 
        required=False,
        help="Write it")

    parser.add_argument(
        "--num_episode_train", 
        default=os.getenv("NO_EPISODE_TRAIN"), 
        type=int, 
        required=False,
        help="Write it")
    
    parser.add_argument(
        "--num_episode_test", 
        default=os.getenv("NO_EPISODE_TEST"), 
        type=int, 
        required=False,
        help="Write it")

    parser.add_argument(
        "--spread", 
        default=os.getenv("SPREAD"), 
        type=float, 
        required=False,     
        help="Write it'")

    args = parser.parse_args()

    config = {'experiment_no': args.experiment_no,
              'flow': args.flow,
              'continuous_action': args.continuous_action,
              'sabr_action': args.sabr_action,
              'random_seed': args.random_seed,
              'num_contract': args.num_contract,
              'risk_aversion': args.risk_aversion,
              'ttm': args.ttm,
              'num_sim_train': args.num_sim_train,
              'num_sim_test': args.num_sim_test,
              'trad_freq': args.trad_freq,
              'num_episode_train': args.num_episode_train,
              'num_episode_test': args.num_episode_test,
              'spread': args.spread,
             }

    # set experiment number in environment variable
    os.environ['EXPERIMENT_NO'] = config["experiment_no"]

    # initialize logger
    global logger
    logger = initialise_logger(log_level=logging.INFO)
    logger.info(f"Experiment started at {datetime.now()} with flow: {config['flow']}")


    # setup for training
    # use init_ttm, spread and other arguments to train for different scenarios
    env = optionEnvironment(continuous_action_flag=config["continuous_action"],
                            sabr_flag=config["sabr_action"], 
                            dg_random_seed= config["random_seed"], 
                            init_ttm=config["ttm"], 
                            trade_freq=config["trad_freq"], 
                            spread=config["spread"], 
                            num_contract=config["num_contract"],
                            num_sim=config["num_sim_train"])

    logger.info(f"The option price environment was initialized successfully")
    ddpg_trainer = DDPG(env, config["risk_aversion"])

    # saving configurations for the experiment
    model_metadata = {
                    'version': _version.__version__,
                    'major_version': _version.MAJOR_VERSION,
                    'minor_version': _version.MINOR_VERSION,
                    'patch_version': _version.PATCH_VERSION,
                    'algorithm': 'DDPG',
                    'model_date': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                }

    model_metadata = dict(model_metadata, **config)
    with open(f"experiments/{os.getenv('EXPERIMENT_NO')}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, ensure_ascii=False, indent=4)

    # for second round training, specify the tag of weights to load
    # ddpg.load(tag="50")

    # for second round training, may want to start with a specific value of epsilon
    # ddpg.epsilon = 0.1

    # episode for training: 0 to 50000 inclusive
    # cycle through available data paths if number of episode for training > number of sim paths
    history = ddpg_trainer.train(config["num_episode_train"])
    ddpg_trainer.save_history(history, "ddpg.csv")

    # setup for testing; use another instance for testing
    env_test = optionEnvironment(continuous_action_flag=config["continuous_action"],
                                sabr_flag=config["sabr_action"], 
                                dg_random_seed= config["random_seed"], 
                                init_ttm=config["ttm"], 
                                trade_freq=config["trad_freq"], 
                                spread=config["spread"], 
                                num_contract=config["num_contract"],
                                num_sim=config["num_sim_test"])
    ddpg_test = DDPG(env_test, config["risk_aversion"])
    ddpg_test.load()

    # episode for testing: 0 to 100000 inclusive
    ddpg_test.test(config["num_episode_test"])

if __name__ == "__main__":
    main()

